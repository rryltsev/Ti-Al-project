# GRACE fp32/fp64: CPU

Controlled fp32 cast of the pinned fp64 checkpoint; model tensors are identical after dtype conversion. Complete diagnostics: PASS.

| Structure | fp32, ms | fp64, ms | fp32 speedup |
|---|---:|---:|---:|
| Ti96 | 703.65 | 923.94 | 1.31x |
| TiAl216 | 1379.55 | 1781.68 | 1.29x |
| TiAl432 | 2705.13 | 3461.78 | 1.28x |

Precision deltas: {"E_meV_atom": {"RMSE": 7.626296509960858e-05, "max_abs": 0.0001610977964361382}, "F_eV_A": {"RMSE": 2.814805428617661e-07, "max_abs": 1.3201304371879274e-06}, "stress_GPa": {"RMSE": 1.1032640176679745e-06, "max_abs": 4.279120312129977e-06}}

PASS means complete finite results, not final model selection. Eight fixed train structures per XC only, no validation/test ranking. CPU main graph JIT off; GPU main graph JIT on. ABBA fresh processes, all-calls cache-bypassed, warmup excluded; model+sync and full ASE timings retained. Not LAMMPS/Kokkos or long-MD validation.

Pending: GPU measurements and full GRACE B0/H0/training qualification
