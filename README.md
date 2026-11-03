# FinForesight

[![Offline release checks](https://github.com/OpenFinMind/FinForesight/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/OpenFinMind/FinForesight/actions/workflows/ci.yml)
[Paper](https://arxiv.org/abs/2611.01621) · [Dataset](https://huggingface.co/datasets/Yilinss27/benchmark-research)

Research code for **FinForesight: A Diagnostic Backtesting Benchmark for Deep Research Systems in Financial Forecasting**
FinForesight separates model-visible tasks, mediated research tools, held-out
labels, and host-side evaluation to study financial forecasting under temporal controls.

| Task | Purpose |
|---|---|
| VAL | Valuation forecasts |
| RANK | Ranking with F, T, and H input variants |
| EVT | Event direction forecasts |
| FUND | Fundamental value forecasts |
| CF | Counterfactual diagnostic |
| CALC | Calculation diagnostic |

Output artifacts retain machine-readable task IDs (A1/A2-F/…); see the
[task ID mapping](docs/tasks.md).

T1, T2, and T3 remain separate temporal populations; CF and CALC are diagnostic
controls and do not enter forecasting aggregates. Pending T3 outcomes remain pending.

## Install and try offline

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are required.

```sh
uv sync --locked --python 3.11
uv run finforesight --help
uv run python scripts/offline_demo.py --output results/offline-demo
```

After installation the example needs no network, API keys, financial data, or
external source checkout. Its fixed synthetic responses exercise the real frozen
parser, evaluator, and atomic artifact writer. They are not model performance results.
Use a new output directory for each invocation.

## Execution paths

| Path | Scope | Inputs |
|---|---|---|
| Offline example | All six tasks, including F/T/H ranking variants | Bundled synthetic fixtures |
| `run` with `one_turn_strict` | Existing native adapter registry | Explicit pinned data and model credentials |
| `run` with `agentic_web_search` | Controlled Volc search | Data, credentials, resident Search+Filter service |
| `run` with `deep_research` | FinanceHarness baseline | Above plus pinned external source and isolated Python 3.12 environment |
| Manifest-driven `experiment` | FinanceHarness, ODR, GPT-Researcher under their gates | Immutable manifest, source/model/data identities and required evidence |

ODR and GPT-Researcher are not legacy `run --baseline` choices. This release does
not supply Claude Code or Codex integrations or certify all manuscript configurations.
The full-release manifest is an input template, not a runnable quickstart.

See [REPRODUCE.md](REPRODUCE.md), [data](docs/data.md),
[configuration](docs/configuration.md), [tasks](docs/tasks.md),
[architecture](docs/architecture.md), and [reproducibility](docs/reproducibility.md).

## License

Project code retains [Apache-2.0](LICENSE). The frozen evaluator retains its
CC-BY-4.0 attribution. See [third-party notices](THIRD_PARTY_NOTICES.md).

## Citation

```bibtex
@misc{liu2026finforesightdiagnosticbacktestingbenchmark,
      title={FinForesight: A Diagnostic Backtesting Benchmark for Deep Research Systems in Financial Forecasting},
      author={Fengyuan Liu and Mukai Li and Jieshun You and Yilin Zheng and Chenjia Min and Yizhou Zhang and Harper Chen and Qi Liu},
      year={2026},
      eprint={2611.01621},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2611.01621},
}
```
