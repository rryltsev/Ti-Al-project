# Ti–Al foundation-model research

Entry points: reports/summary.md, state.json, protocol/benchmark_protocol.md.
The scientific programme is preserved in protocol/CODEX_TiAl_FM_RESEARCH_PLAN.md.

Reproduce the initial read-only audit with the existing project Python:

```powershell
& './testset/.venv/Scripts/python.exe' research/tial_fm/scripts/audit_initial.py --run-id UNIQUE_NEW_RUN_ID
```

Run from the TiAl workspace. A fresh run ID is mandatory; the script refuses to overwrite runs.
Expected Python dependencies: NumPy and ASE. Source paths and their SHA-256 values are recorded.
The unified dataset is a manifest view over original labels; no DFT/teacher labels are mixed or overwritten.
`protocol/experiment_matrix.yaml` records designs and their current gate/submission status.
The concrete N200 DPA4 pilot is frozen in `runs/dpa4-n200-pilot-20261106-02`.
Read `reports/08_n200_pilot.md`, `reports/10_original_harmonics.md` and `state.json` for actual progress.

The N200 data are in `datasets/nested-train-v1-20261106`; development/test remain
the unchanged149/149 of `common992-grouped-v1-20261106-02`.
Every pilot starts from the original pretrained checkpoint. No legacy TiAl FT
checkpoint is a valid independent initialization/baseline for the new test split.

`scripts/cluster_pilot.py` provides authenticated, pinned-host SSH transport;
it never copies credentials. Heavy work runs inside Slurm. A training submission
requires completed B0/H0 gates. Each actual training job additionally verifies
zero-step equivalence, the runtime parameter mask, frozen weights, and exports.
Large datasets, model weights and transfer archives are ignored by Git.
