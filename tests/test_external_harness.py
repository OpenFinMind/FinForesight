"""Offline contract tests for the FinanceHarness external boundary."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import textwrap
import time
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark_harness.config import (
    FINANCEHARNESS_PROTOCOL,
    DatasetConfig,
    DeepResearchConfig,
    ExecutionConfig,
    ExperimentConfig,
    ExternalHarnessConfig,
    ModelConfig,
    ResolvedModelProfile,
    SearchConfig,
    TraceConfig,
    load_experiment_config,
)
from benchmark_harness.contracts import (
    TEMPORAL_FILTER_NETWORK_CLASS,
    AdapterResult,
    NetworkEvent,
    RunContext,
    TaskInput,
    ToolCall,
    TraceEvent,
)
from benchmark_harness.external_harnesses.checkout import (
    ExternalHarnessError,
    ExternalHarnessManager,
)
from benchmark_harness.external_harnesses.financeharness.adapter import (
    DeepModelBroker,
    DeepRunOutput,
    FinanceHarnessAdapter,
    FinanceHarnessBoundaryError,
    _validated_service_receipts,
)
from benchmark_harness.external_harnesses.financeharness.policy import DeepResearchPolicyGuard
from benchmark_harness.external_harnesses.ipc import (
    IPC_PROTOCOL,
    Envelope,
    IpcChannel,
    IpcProtocolError,
    build_child_environment,
    encode_envelope,
)
from benchmark_harness.profile_registry import get_profile, normalize_baseline
from benchmark_harness.runner import BenchmarkRunner
from benchmark_harness.search import CutoffSearchGateway, VolcHttpResponse
from benchmark_harness.search_service.api import SearchSessionDescriptor, ServiceLimits
from benchmark_harness.trace.jsonl_sink import JsonlSink


def test_deep_example_and_registry_are_locked() -> None:
    config = load_experiment_config("configs/deep-research-financeharness.example.toml")
    assert config.is_deep_research
    assert config.deep_research == DeepResearchConfig()
    assert config.external_harnesses["financeharness"].protocol == FINANCEHARNESS_PROTOCOL
    assert get_profile("deep_research").allowed_baselines == frozenset({"financeharness"})
    assert normalize_baseline("finance-harness") == "financeharness"


def test_ipc_rejects_unknown_fields_and_nonmonotonic_frames() -> None:
    frame = encode_envelope(
        Envelope(
            type="hello", request_id="r", task_id="t", seq=0, payload={"protocol": IPC_PROTOCOL}
        )
    )
    channel = IpcChannel(BytesIO(frame), BytesIO(), task_id="t")
    assert channel.read().type == "hello"
    duplicate_channel = IpcChannel(BytesIO(frame + frame), BytesIO(), task_id="t")
    assert duplicate_channel.read().type == "hello"
    with pytest.raises(IpcProtocolError):
        duplicate_channel.read()


def test_child_environment_strips_parent_credentials() -> None:
    environment = build_child_environment(
        "/tmp/financeharness",
        "/tmp/benchmark",
        base_environment={
            "PATH": "/bin",
            "ALIYUN_API_KEY": "secret",
            "VOLC_SECRET_ACCESS_KEY": "secret",
            "OPENAI_API_KEY": "secret",
            "LANG": "C.UTF-8",
        },
    )
    assert "ALIYUN_API_KEY" not in environment
    assert "VOLC_SECRET_ACCESS_KEY" not in environment
    assert "OPENAI_API_KEY" not in environment
    assert environment["PYTHONUNBUFFERED"] == "1"


def test_financeharness_validates_child_receipt_handoff_chain() -> None:
    descriptor = SearchSessionDescriptor(
        session_id="a" * 32,
        task_token="task-token",
        run_id="handoff-run",
        task_id="A1-handoff",
        profile="deep_research",
        baseline="financeharness",
        cutoff_date="2024-06-01",
        policy_version="deep_research/v2",
        expires_at="2099-01-01T00:00:00+00:00",
        instance_id="instance",
        limits=ServiceLimits(8, 80, 240, 1860),
        api_contract_sha256="api",
        implementation_manifest_sha256="implementation",
        capability_sha256="capability",
        service_cohort_sha256="cohort",
    )

    class ChildSession:
        def __init__(self) -> None:
            self.descriptor = descriptor

    first_hash = hashlib.sha256(b"receipt-1").hexdigest()
    second_hash = hashlib.sha256(b"receipt-2").hexdigest()
    first = {
        "receipt_index": 1,
        "service_request_id": "request-1",
        "receipt_sha256": first_hash,
        "previous_receipt_sha256": None,
        "call_index": 1,
        "replayed": False,
        "api_contract_sha256": "api",
        "implementation_manifest_sha256": "implementation",
        "capability_sha256": "capability",
        "service_cohort_sha256": "cohort",
    }
    second = {
        **first,
        "receipt_index": 2,
        "service_request_id": "request-2",
        "receipt_sha256": second_hash,
        "previous_receipt_sha256": first_hash,
        "call_index": 2,
    }

    receipts = _validated_service_receipts([first, second], service_session=ChildSession())
    assert [receipt.receipt_sha256 for receipt in receipts] == [first_hash, second_hash]

    broken = [first, {**second, "previous_receipt_sha256": "0" * 64}]
    with pytest.raises(
        FinanceHarnessBoundaryError,
        match="child_search_service_receipt_chain_invalid",
    ):
        _validated_service_receipts(broken, service_session=ChildSession())


def test_external_sync_requires_license_ack_before_side_effect(tmp_path: Path) -> None:
    config = ExternalHarnessConfig(checkout_root=tmp_path)
    manager = ExternalHarnessManager(config)
    with pytest.raises(ExternalHarnessError, match="UNDECLARED"):
        manager.sync()
    assert not (tmp_path / "financeharness").exists()


def test_deep_policy_skips_missing_d_e_cutoff_without_activity() -> None:
    task = TaskInput(task_id="D-1", prompt="question", cutoff_date=None)
    decision = DeepResearchPolicyGuard().check(task, None, skipped_no_cutoff=True)
    assert decision.comparable is True
    assert decision.model_call_count == 0
    assert decision.child_network_call_count == 0


def test_parent_model_broker_retries_and_ledger_counts_each_attempt() -> None:
    profile = ResolvedModelProfile(
        mode="mock",
        endpoint="mock://local",
        model_name="mock-model",
        api_key=None,
        temperature=0.0,
        max_tokens=128,
        timeout_seconds=1.0,
        retries=1,
        seed=None,
    )
    context = RunContext(
        run_id="run",
        framework="financeharness",
        adapter_version="test",
        model=profile,
        trace_content_mode="hash_only",
    )
    attempts = 0

    def responder(_request: dict[str, object]) -> dict[str, object]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("synthetic timeout")
        return {"choices": [{"message": {"content": "answer"}}]}

    broker = DeepModelBroker(profile, context, max_calls=4, responder=responder)
    response = broker.request({"messages": [{"role": "user", "content": "q"}]})
    assert response["choices"]
    assert broker.call_count == 2
    assert len(broker.network_events) == 2
    assert broker.network_events[0].error == "TimeoutError"
    assert all(request["model"] == "mock-model" for request in broker.requests)


class _DeepFakeSource:
    def __init__(self, records: list[dict[str, object]]) -> None:
        self.records = records

    def preflight(self) -> dict[str, object]:
        return self.provenance()

    def provenance(self) -> dict[str, object]:
        return {"source": "injected-test-source", "dataset_revision": "test-v1"}

    def load_records(
        self, *, categories: list[str] | None = None, status: str | None = "ready"
    ) -> list[dict[str, object]]:
        records = [
            record for record in self.records if status is None or record["status"] == status
        ]
        if categories:
            wanted = {category.upper() for category in categories}
            records = [record for record in records if record["category"] in wanted]
        return records


class _DeepFakeEvaluation:
    def parse_prediction(self, category: str, raw_text: str) -> dict[str, object]:
        return {"format_valid": bool(raw_text), "category": category, "raw": raw_text}

    def evaluate_prediction(
        self, category: str, parsed: dict[str, object], record: dict[str, object]
    ) -> dict[str, object]:
        del record
        return {"format_valid": parsed["format_valid"], "category": category}

    def summarize_results(
        self,
        predictions: list[dict[str, object]],
        skipped: list[dict[str, object]],
        *,
        agent: str,
        model: str,
    ) -> dict[str, object]:
        return {
            "agent": agent,
            "model": model,
            "evaluated": len(predictions),
            "skipped": len(skipped),
        }


class _DeepSpyEvaluation(_DeepFakeEvaluation):
    def __init__(self) -> None:
        self.parse_calls = 0
        self.evaluate_calls = 0
        self.summary_inputs: list[list[dict[str, object]]] = []

    def parse_prediction(self, category: str, raw_text: str) -> dict[str, object]:
        self.parse_calls += 1
        return super().parse_prediction(category, raw_text)

    def evaluate_prediction(
        self, category: str, parsed: dict[str, object], record: dict[str, object]
    ) -> dict[str, object]:
        self.evaluate_calls += 1
        return super().evaluate_prediction(category, parsed, record)

    def summarize_results(
        self,
        predictions: list[dict[str, object]],
        skipped: list[dict[str, object]],
        *,
        agent: str,
        model: str,
    ) -> dict[str, object]:
        self.summary_inputs.append(list(predictions))
        return super().summarize_results(predictions, skipped, agent=agent, model=model)


class _DeepFakeVolcTransport:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def post(
        self,
        url: str,
        *,
        json: dict[str, object],
        headers: dict[str, str],
        timeout_seconds: float,
    ) -> VolcHttpResponse:
        del url, headers, timeout_seconds
        self.queries.append(str(json["Query"]))
        return VolcHttpResponse(
            status_code=200,
            request_id="deep-fake-search",
            payload={
                "Result": {
                    "WebResults": [
                        {
                            "Url": "https://source.example/pre-cutoff",
                            "Title": "Historical evidence",
                            "Content": "Evidence available before the cutoff.",
                            "PublishTime": "2024-01-01T00:00:00Z",
                        }
                    ]
                }
            },
        )


def test_filter_calls_remain_audited_but_do_not_consume_agent_budget(tmp_path: Path) -> None:
    task = TaskInput(
        task_id="A1-BUDGET",
        prompt="Research the historical evidence.",
        cutoff_date="2024-06-01",
    )
    gateway = CutoffSearchGateway(transport=_DeepFakeVolcTransport(), test_only=True)
    outcome = gateway.search(
        task_id=task.task_id,
        framework="financeharness",
        query="historical evidence",
        cutoff_date=task.cutoff_date,
    )
    agent_events = tuple(
        NetworkEvent(
            host="mock.parent.model",
            method="POST",
            path="/v1/chat/completions",
            body_hash=f"agent-{index}",
            status_code=200,
            latency_seconds=0.0,
            is_model_endpoint=True,
        )
        for index in range(32)
    )
    filter_event = NetworkEvent(
        host="mock.parent.model",
        method="POST",
        path="/v1/chat/completions",
        body_hash="filter-1",
        status_code=200,
        latency_seconds=0.0,
        is_model_endpoint=True,
        network_class=TEMPORAL_FILTER_NETWORK_CLASS,
    )
    result = AdapterResult(
        raw_text="grounded answer",
        usage={},
        latency_seconds=0.0,
        canonical_input_hash=task.input_hash,
        effective_input_hash=task.input_hash,
        effective_messages=tuple(task.messages),
        effective_request={
            "deep_research": {
                "termination": "answer",
                "rounds": 1,
                "filter_model_call_count": 1,
            }
        },
        model_call_count=len(agent_events),
        tool_calls=(ToolCall(name="search"),),
        network_events=(*agent_events, filter_event),
        trace_events=(),
    )
    sink = JsonlSink(tmp_path, content_mode="full")
    sink.write_network_events(
        run_id="budget-run",
        task_id=task.task_id,
        framework="financeharness",
        events=result.network_events,
    )
    sink.write_trace_events(
        (
            TraceEvent(
                run_id="budget-run",
                task_id=task.task_id,
                framework="financeharness",
                adapter_version="financeharness-external/v1",
                event_seq=1,
                event_type="model_response",
                timestamp="2026-08-12T00:00:00+00:00",
                control_flow_node="financeharness_parent_temporal_filter",
                network={
                    "is_model_endpoint": True,
                    "network_class": TEMPORAL_FILTER_NETWORK_CLASS,
                },
            ),
        )
    )
    decision = DeepResearchPolicyGuard().check(
        task,
        result,
        search_outcomes=(outcome,),
        network_ledger_rows=(outcome.network_ledger(profile="deep_research"),),
        research_audit_rows=(
            {"task_id": task.task_id, "event_type": "task_started", "network_call_count": 0},
            {"task_id": task.task_id, "event_type": "task_finished", "network_call_count": 0},
        ),
        citations=("https://source.example/pre-cutoff",),
    )

    assert len(agent_events) + 1 > DeepResearchConfig().max_model_calls
    assert result.model_call_count == DeepResearchConfig().max_model_calls
    assert decision.comparable is True
    assert decision.model_call_count == DeepResearchConfig().max_model_calls
    assert decision.filter_model_call_count == 1
    assert result.network_events[-1].network_class == TEMPORAL_FILTER_NETWORK_CLASS
    network_rows = [
        json.loads(line) for line in sink.network_path.read_text(encoding="utf-8").splitlines()
    ]
    trace_rows = [
        json.loads(line) for line in sink.trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(row.get("network_class") == TEMPORAL_FILTER_NETWORK_CLASS for row in network_rows)
    assert any(
        row.get("control_flow_node") == "financeharness_parent_temporal_filter"
        and row.get("network", {}).get("network_class") == TEMPORAL_FILTER_NETWORK_CLASS
        for row in trace_rows
    )


def test_deep_runner_fake_e2e_preserves_parent_search_and_time_band(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmark_harness import runner as runner_module

    external_config = ExternalHarnessConfig(checkout_root=tmp_path / "external")
    manifest = {
        "schema_version": "1",
        "adapter": "financeharness",
        "adapter_version": "v1",
        "repo_url": external_config.repo_url,
        "revision": external_config.revision,
        "tree_sha": external_config.tree_sha,
        "uv_lock_sha256": external_config.uv_lock_sha256,
        "pyproject_sha256": external_config.pyproject_sha256,
        "python_version": "3.12",
        "protocol": FINANCEHARNESS_PROTOCOL,
        "license_status": "UNDECLARED",
        "license_acknowledged": True,
        "source_relative": "data/external-harnesses/financeharness",
    }

    class FakeManager:
        config = external_config

        def verify(self) -> dict[str, object]:
            return manifest

    manager = FakeManager()
    transport = _DeepFakeVolcTransport()
    gateway = CutoffSearchGateway(transport=transport, test_only=True)

    class FakeAdapter:
        adapter_version = "financeharness-external/v1"

        def run_deep(self, task: TaskInput, context: RunContext) -> DeepRunOutput:
            outcome = gateway.search(
                task_id=task.task_id,
                framework="financeharness",
                query="model selected historical evidence",
                cutoff_date=task.cutoff_date,
            )
            context.emit("model_request", control_flow_node="fake_parent_model")
            context.emit("model_response", control_flow_node="fake_parent_model")
            context.emit("tool_call", tool_name="search", control_flow_node="fake_worker")
            context.emit("task_finished", control_flow_node="fake_worker")
            effective = task.messages
            model_event = NetworkEvent(
                host="mock.parent.model",
                method="POST",
                path="/v1/chat/completions",
                body_hash="fake-model-request",
                status_code=200,
                latency_seconds=0.0,
                is_model_endpoint=True,
            )
            result = AdapterResult(
                raw_text="grounded answer",
                usage={},
                latency_seconds=0.0,
                canonical_input_hash=task.input_hash,
                effective_input_hash=task.input_hash,
                effective_messages=tuple(effective),
                effective_request={
                    "messages": effective,
                    "messages_sha256": task.input_hash,
                    "prompt_sha256": "prompt",
                    "deep_research": {"termination": "answer", "rounds": 1},
                },
                model_call_count=1,
                tool_calls=(ToolCall(name="search"),),
                network_events=(model_event,),
                trace_events=context.events,
            )
            return DeepRunOutput(
                result=result,
                search_outcomes=(outcome,),
                search_ledger_rows=(outcome.network_ledger(profile="deep_research"),),
                research_audit_rows=(
                    {
                        "task_id": task.task_id,
                        "event_type": "task_started",
                        "network_call_count": 0,
                    },
                    {
                        "task_id": task.task_id,
                        "event_type": "task_finished",
                        "network_call_count": 0,
                    },
                ),
                citations=("https://source.example/pre-cutoff",),
                external_manifest=manifest,
            )

    config = ExperimentConfig(
        dataset=DatasetConfig(),
        model=ModelConfig(mode="mock"),
        search=SearchConfig(test_only=True),
        execution=ExecutionConfig(input_policy="deep_research"),
        trace=TraceConfig(content_mode="hash_only", results_dir=tmp_path / "results-deep-research"),
        deep_research=DeepResearchConfig(),
        external_harnesses={"financeharness": external_config},
    )
    source = _DeepFakeSource(
        [
            {
                "task_id": "A1-00001",
                "category": "A1",
                "status": "ready",
                "time_band": "T1",
                "prompt": "Research the historical evidence.",
                "seed": {"cutoff_date": "2024-06-01"},
            }
        ]
    )
    monkeypatch.setattr(runner_module, "_make_external_harness_manager", lambda _config: manager)
    runner = BenchmarkRunner(
        config,
        dataset_source_factory=lambda _config: source,
        evaluation_engine=_DeepFakeEvaluation(),
        search_gateway_factory=lambda _config: gateway,
        external_harness_adapter_factory=lambda **_kwargs: FakeAdapter(),
    )
    result = runner.run(baselines=["financeharness"], run_id="deep-fake", isolated=False)
    baseline = result["baselines"][0]
    output_dir = tmp_path / "results-deep-research" / "deep-fake" / "financeharness"
    summary = json.loads((output_dir / "metrics_summary.json").read_text())
    prediction = json.loads((output_dir / "predictions.jsonl").read_text())
    assert baseline["comparable_task_count"] == 1
    assert prediction["comparable"] is True
    assert summary["by_time_band"] == {"T1": 1}
    assert transport.queries == ["model selected historical evidence"]
    assert (output_dir / "research_audit.jsonl").is_file()
    assert (output_dir / "external_harness_manifest.json").is_file()


def test_deep_runner_excludes_skipped_rows_from_frozen_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from benchmark_harness import runner as runner_module

    external_config = ExternalHarnessConfig(checkout_root=tmp_path / "external")
    manifest = {
        "schema_version": "1",
        "adapter": "financeharness",
        "adapter_version": "v1",
        "repo_url": external_config.repo_url,
        "revision": external_config.revision,
        "tree_sha": external_config.tree_sha,
        "uv_lock_sha256": external_config.uv_lock_sha256,
        "pyproject_sha256": external_config.pyproject_sha256,
        "python_version": "3.12",
        "protocol": FINANCEHARNESS_PROTOCOL,
        "license_status": "UNDECLARED",
        "license_acknowledged": True,
        "source_relative": "data/external-harnesses/financeharness",
    }

    class FakeManager:
        config = external_config

        def verify(self) -> dict[str, object]:
            return manifest

    class SpyAdapter:
        adapter_version = "financeharness-external/v1"

        def __init__(self) -> None:
            self.calls = 0

        def run_deep(self, task: TaskInput, context: RunContext) -> DeepRunOutput:
            del task, context
            self.calls += 1
            raise AssertionError("skipped deep tasks must not invoke the adapter")

    manager = FakeManager()
    adapter = SpyAdapter()
    evaluator = _DeepSpyEvaluation()
    source = _DeepFakeSource(
        [
            {
                "task_id": "D-00001",
                "category": "D",
                "status": "ready",
                "time_band": "T1",
                "prompt": "D no cutoff",
                "seed": {},
            },
            {
                "task_id": "E-00001",
                "category": "E",
                "status": "ready",
                "time_band": "T2",
                "prompt": "E no cutoff",
                "seed": {},
            },
            {
                "task_id": "B-00001",
                "category": "B",
                "status": "ready",
                "time_band": "T3",
                "prompt": "B cutoff after event",
                "seed": {"cutoff_date": "2024-02-01", "event_date": "2024-01-01"},
            },
        ]
    )
    transport = _DeepFakeVolcTransport()
    gateway = CutoffSearchGateway(transport=transport, test_only=True)
    config = ExperimentConfig(
        dataset=DatasetConfig(),
        model=ModelConfig(mode="mock"),
        search=SearchConfig(test_only=True),
        execution=ExecutionConfig(input_policy="deep_research"),
        trace=TraceConfig(content_mode="hash_only", results_dir=tmp_path / "results-deep-research"),
        deep_research=DeepResearchConfig(),
        external_harnesses={"financeharness": external_config},
    )
    monkeypatch.setattr(runner_module, "_make_external_harness_manager", lambda _config: manager)
    runner = BenchmarkRunner(
        config,
        dataset_source_factory=lambda _config: source,
        evaluation_engine=evaluator,
        search_gateway_factory=lambda _config: gateway,
        external_harness_adapter_factory=lambda **_kwargs: adapter,
    )

    result = runner.run(baselines=["financeharness"], run_id="deep-skip-boundary", isolated=False)
    baseline = result["baselines"][0]
    output_dir = tmp_path / "results-deep-research" / "deep-skip-boundary" / "financeharness"
    summary = json.loads((output_dir / "metrics_summary.json").read_text())
    predictions = {
        row["task_id"]: row
        for row in map(json.loads, (output_dir / "predictions.jsonl").read_text().splitlines())
    }

    assert adapter.calls == 0
    assert transport.queries == []
    assert evaluator.parse_calls == 0
    assert evaluator.evaluate_calls == 0
    assert evaluator.summary_inputs == [[]]
    assert baseline["comparable_task_count"] == 0
    assert summary["harness"]["comparable_task_count"] == 0
    assert summary["upstream_summary"]["evaluated"] == 0
    assert summary["upstream_summary"]["skipped"] == 3
    for task_id, row in predictions.items():
        assert row["parsed"] == {}
        assert row["metrics"] == {}
        assert row["model_call_count"] == 0
        assert row["filter_model_call_count"] == 0
        assert row["tool_call_count"] == 0
        assert row["non_model_network_call_count"] == 0
        assert row["search_call_count"] == 0
        assert row["visit_count"] == 0
        assert row["child_network_call_count"] == 0
        if task_id in {"D-00001", "E-00001"}:
            assert row["skipped_no_cutoff"] is True
        else:
            assert row["skipped_no_cutoff"] is False


def test_host_adapter_rejects_legacy_search_frame_over_ipc(tmp_path: Path) -> None:
    task_id = "A1-IPC"
    external_config = ExternalHarnessConfig(checkout_root=tmp_path / "external")
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    manifest = {
        "schema_version": "1",
        "adapter": "financeharness",
        "adapter_version": "v2",
        "repo_url": external_config.repo_url,
        "revision": external_config.revision,
        "tree_sha": external_config.tree_sha,
        "uv_lock_sha256": external_config.uv_lock_sha256,
        "pyproject_sha256": external_config.pyproject_sha256,
        "python_version": "3.12",
        "protocol": FINANCEHARNESS_PROTOCOL,
        "license_status": "UNDECLARED",
    }

    class FakeManager:
        def __init__(self) -> None:
            self.config = external_config
            self.repo_root = tmp_path
            self.source_dir = source_dir
            self.venv_python = Path(sys.executable)

        def verify(self) -> dict[str, object]:
            return manifest

    script = textwrap.dedent(
        f"""
        import json
        import sys
        from benchmark_harness.external_harnesses.ipc import IpcChannel, IPC_PROTOCOL

        channel = IpcChannel(sys.stdin.buffer, sys.stdout.buffer, task_id={task_id!r})
        channel.send(
            "hello",
            {{
                "protocol": IPC_PROTOCOL,
                "adapter": "financeharness",
                "adapter_version": "v2",
                "child_network": "forbidden",
            }},
            request_id="hello",
        )
        channel.read()
        # Bypass the current encoder: it rejects legacy types before transmission.
        legacy = {{
            "schema_version": IPC_PROTOCOL,
            "type": "search.request",
            "request_id": "search-1",
            "task_id": {task_id!r},
            "seq": 1,
            "payload": {{"query": "historical evidence"}},
        }}
        sys.stdout.write(json.dumps(legacy) + "\\n")
        sys.stdout.flush()
        """
    )

    class FakeServiceSession:
        def __init__(self) -> None:
            self.descriptor = SearchSessionDescriptor(
                session_id="a" * 32,
                task_token="task-token",
                run_id="ipc-run",
                task_id=task_id,
                profile="deep_research",
                baseline="financeharness",
                cutoff_date="2024-06-01",
                policy_version="deep_research/v2",
                expires_at="2099-01-01T00:00:00+00:00",
                instance_id="instance",
                limits=ServiceLimits(8, 80, 240, 1860),
                api_contract_sha256="api",
                implementation_manifest_sha256="implementation",
                capability_sha256="capability",
                service_cohort_sha256="cohort",
            )
            self.receipts: tuple[object, ...] = ()
            self.close_result: dict[str, object] | None = None

        def close(self, *, status: str) -> dict[str, object]:
            self.close_result = {
                "status": status,
                "service_manifest": {
                    "protocol": "search_filter_http/v1",
                    "api_contract_sha256": "api",
                    "implementation_manifest_sha256": "implementation",
                    "capability_sha256": "capability",
                    "service_cohort_sha256": "cohort",
                },
                "network_ledger": [],
                "search_audit": [],
            }
            return self.close_result

    class FakeService:
        base_url = "http://127.0.0.1:8765"

        def preflight(self) -> dict[str, str]:
            return {"service_cohort_sha256": "cohort"}

        def open_session(self, **_kwargs: object) -> FakeServiceSession:
            return FakeServiceSession()

    # Keep fixture host pipe reads unbuffered so select() cannot miss a frame
    # read ahead into Python's buffer.
    # This test exercises legacy-frame rejection, not buffered pipe scheduling.
    def unbuffered_process(*args: object, **kwargs: object) -> subprocess.Popen:
        return subprocess.Popen(*args, **kwargs, bufsize=0)

    service = FakeService()
    adapter = FinanceHarnessAdapter(
        FakeManager(),
        search_service=service,  # type: ignore[arg-type]
        worker_command=[sys.executable, "-c", script],
        repo_root=tmp_path,
        process_factory=unbuffered_process,
        deep_config=DeepResearchConfig(task_timeout_seconds=5.0),
    )
    profile = ResolvedModelProfile(
        mode="mock",
        endpoint="mock://local",
        model_name="mock-model",
        api_key=None,
        temperature=0.0,
        max_tokens=128,
        timeout_seconds=1.0,
        retries=0,
        seed=None,
    )
    task = TaskInput(
        task_id=task_id, prompt="Research historical evidence.", cutoff_date="2024-06-01"
    )
    context = RunContext(
        run_id="ipc-run",
        framework="financeharness",
        adapter_version=adapter.adapter_version,
        model=profile,
        trace_content_mode="hash_only",
    ).for_task(task_id)
    with pytest.raises(
        FinanceHarnessBoundaryError, match="unsupported IPC message type:.*search.request"
    ):
        adapter.run_deep(task, context)


@pytest.mark.skipif(os.name == "nt", reason="POSIX pipe readiness")
def test_buffered_ipc_read_ahead_preserves_frames() -> None:
    script = """
import json, sys
from benchmark_harness.external_harnesses.ipc import IPC_PROTOCOL
frames = [{"schema_version": IPC_PROTOCOL, "type": kind, "request_id": str(i),
           "task_id": "race", "seq": i, "payload": {"index": i}}
          for i, kind in enumerate(["audit.event", "model.request"])]
sys.stdout.write("".join(json.dumps(f) + "\\n" for f in frames))
sys.stdout.flush()
sys.stdin.readline()
"""
    # One write followed by an explicit parent-side peek guarantees read-ahead,
    # rather than relying on scheduler timing to coalesce separate writes.
    with subprocess.Popen(
        [sys.executable, "-c", script], stdin=subprocess.PIPE, stdout=subprocess.PIPE
    ) as worker:
        try:
            assert worker.stdout is not None and worker.stdin is not None
            assert worker.stdout.peek(8192).count(b"\n") == 2
            channel = IpcChannel(worker.stdout, worker.stdin, task_id="race")
            assert channel.read().payload == {"index": 0}
            adapter = FinanceHarnessAdapter.__new__(FinanceHarnessAdapter)
            adapter.deep_config = DeepResearchConfig(task_timeout_seconds=0.2)
            frame = adapter._read(channel, worker, time.perf_counter())
            assert (frame.type, frame.seq, frame.payload) == ("model.request", 1, {"index": 1})
            worker.stdin.write(b"response\n")
            worker.stdin.flush()
            assert worker.wait(timeout=2) == 0
        finally:
            if worker.poll() is None:
                worker.kill()
                worker.wait()


def test_silent_worker_deadline_is_bounded() -> None:
    # The outer process deadline also catches an implementation that blocks
    # inside peek before it can inspect the adapter's task deadline.
    script = """
import subprocess, sys, time
from benchmark_harness.config import DeepResearchConfig
from benchmark_harness.external_harnesses.financeharness.adapter import (
    FinanceHarnessAdapter, FinanceHarnessBoundaryError)
from benchmark_harness.external_harnesses.ipc import IpcChannel
worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(2)'],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE)
try:
    adapter = FinanceHarnessAdapter.__new__(FinanceHarnessAdapter)
    adapter.deep_config = DeepResearchConfig(task_timeout_seconds=0.1)
    channel = IpcChannel(worker.stdout, worker.stdin, task_id='silent')
    started = time.perf_counter()
    try:
        adapter._read(channel, worker, started)
    except FinanceHarnessBoundaryError as exc:
        assert str(exc) == 'external_worker_timeout', str(exc)
        assert time.perf_counter() - started < 1.0
        print('bounded timeout')
    else:
        raise AssertionError('silent worker unexpectedly returned a frame')
finally:
    if worker.poll() is None:
        worker.kill()
    worker.wait()
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=3
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "bounded timeout"


@pytest.mark.parametrize("blocking", [False, True])
def test_buffer_probe_restores_fd_state(blocking: bool) -> None:
    import io

    read_fd, write_fd = os.pipe()
    with io.BufferedReader(io.FileIO(read_fd, "rb")) as stream:
        try:
            os.set_blocking(read_fd, blocking)
            assert FinanceHarnessAdapter._stdout_buffered(stream) is False
            assert os.get_blocking(read_fd) is blocking
            os.write(write_fd, b"frame\n")
            assert FinanceHarnessAdapter._stdout_buffered(stream) is True
            assert os.get_blocking(read_fd) is blocking
            assert stream.readline() == b"frame\n"
        finally:
            os.close(write_fd)


def test_raw_stream_and_eof_keep_existing_behavior() -> None:
    from benchmark_harness.external_harnesses.ipc import IpcProtocolError

    assert FinanceHarnessAdapter._stdout_buffered(BytesIO()) is False
    read_fd, write_fd = os.pipe()
    os.close(write_fd)
    with os.fdopen(read_fd, "rb", buffering=0) as stream:
        assert FinanceHarnessAdapter._stdout_buffered(stream) is False
        adapter = FinanceHarnessAdapter.__new__(FinanceHarnessAdapter)
        adapter.deep_config = DeepResearchConfig(task_timeout_seconds=0.2)
        worker = SimpleNamespace(stdout=stream, stderr=None, poll=lambda: None)
        channel = IpcChannel(stream, BytesIO(), task_id="eof")
        with pytest.raises(
            IpcProtocolError, match="^IPC peer closed stdout before task completion$"
        ):
            adapter._read(channel, worker, time.perf_counter())


@pytest.mark.parametrize("error_number", [11, 35])
def test_buffer_probe_would_block_restores_state(error_number: int) -> None:
    read_fd, write_fd = os.pipe()

    class Stream:
        def fileno(self) -> int:
            return read_fd

        def peek(self, _size: int) -> bytes:
            import errno

            raise BlockingIOError(
                errno.EAGAIN if error_number == 11 else errno.EWOULDBLOCK, "empty"
            )

    try:
        assert FinanceHarnessAdapter._stdout_buffered(Stream()) is False
        assert os.get_blocking(read_fd) is True
    finally:
        os.close(read_fd)
        os.close(write_fd)


def test_buffer_probe_unsupported_platform_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    class Stream:
        def peek(self, _size: int) -> bytes:
            raise AssertionError("peek must not be called without set_blocking")

    monkeypatch.delattr(os, "set_blocking")
    assert FinanceHarnessAdapter._stdout_buffered(Stream()) is False
