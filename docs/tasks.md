# Tasks and temporal populations

| Task | Machine-readable task ID | Purpose / input variant |
|---|---|---|
| VAL | A1 | Valuation scenarios |
| RANK | A2-F | Fundamental evidence only (`a2_f`) |
| RANK | A2-T | Precomputed technical indicators only (`a2_t`) |
| RANK | A2-H | Both fundamental evidence and precomputed technical indicators (`a2_h`) |
| EVT | B | Event direction |
| FUND | C | Fundamental forecasts |
| CF | D | Counterfactual diagnostic |
| CALC | E | Calculation diagnostic |

Category and schema identifiers remain unchanged in code and output artifacts.

T1/T2/T3 assignments are analysis-side temporal populations determined by pinned
cutoff registries, task dates, guard days and settlement state. Keep them separate;
never send the `time_band` or held-out reference fields to a system. Pending T3
outcomes are not historical resolved labels. The actual dataset/index/cutoff hashes
are part of experiment identity.
