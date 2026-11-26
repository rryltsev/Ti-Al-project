# GRACE fp32/fp64: GPU

Controlled fp32 cast of the pinned fp64 checkpoint; model tensors are identical after dtype conversion. Complete diagnostics: PASS.

| Structure | fp32, ms | fp64, ms | fp32 speedup |
|---|---:|---:|---:|
| Ti96 | 14.62 | 15.41 | 1.05x |
| TiAl216 | 19.13 | 20.26 | 1.06x |
| TiAl432 | 28.16 | 31.19 | 1.11x |
| TiAl864 | 48.47 | 60.65 | 1.25x |

Precision deltas: {"E_meV_atom": {"RMSE": 0.0001433101365574061, "max_abs": 0.00020274451253499137}, "F_eV_A": {"RMSE": 2.767503806855799e-07, "max_abs": 1.3200603563578994e-06}, "stress_GPa": {"RMSE": 2.3457315438002556e-06, "max_abs": 8.463261893217292e-06}}

PASS means complete finite results, not final model selection. Eight fixed train structures per XC only, no validation/test ranking. CPU main graph JIT off; GPU main graph JIT on. ABBA fresh processes, all-calls cache-bypassed, warmup excluded; model+sync and full ASE timings retained. Not LAMMPS/Kokkos or long-MD validation.

Pending: Full GRACE B0/H0/training qualification
