# GRACE PBE N200: three seeds / four masks

PASS

Metrics are validation149 RMSE; energy aligned on train200 only; TiAl sorted band RMSE; Ti3Al archived Cmcm DOS W1. Fixed final epoch50, fp64.

| Mask | E aligned meV/atom | F eV/A | Stress GPa | TiAl bands THz | Ti3Al W1 THz |
|---|---:|---:|---:|---:|---:|
| full | 1.358389 ± 0.064586 | 0.075948 ± 0.000072 | 0.083826 ± 0.019578 | 0.221787 ± 0.003868 | 0.290783 ± 0.006515 |
| restricted | 1.302313 ± 0.045100 | 0.078172 ± 0.000002 | 0.076944 ± 0.000955 | 0.313902 ± 0.000549 | 0.275934 ± 0.001549 |
| restricted_mlp | 1.307759 ± 0.039554 | 0.078221 ± 0.000002 | 0.077894 ± 0.001780 | 0.318047 ± 0.000629 | 0.266616 ± 0.000856 |
| restricted_radial | 1.328433 ± 0.093030 | 0.076609 ± 0.000021 | 0.079248 ± 0.002303 | 0.264684 ± 0.000786 | 0.231697 ± 0.005080 |

Sample SD across training seeds only; no uncertainty across dataset splits. Ti3Al DFT force constants unavailable; no DFT band RMSE claimed.
{}
