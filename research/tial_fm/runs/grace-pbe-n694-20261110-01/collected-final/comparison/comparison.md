# GRACE PBE N694 versus N200

PASS

Mean +/- sample SD across seeds10/20/30. Fixed validation149. Same epoch budget, different update budget. No test access.

| N | Mask | E aligned meV/atom | F eV/A | Stress GPa | TiAl RMSE THz | Ti3Al W1 THz | Ti3Al L1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 200 | full | 1.358389 +/- 0.064586 | 0.075948 +/- 0.000072 | 0.083826 +/- 0.019578 | 0.221787 +/- 0.003868 | 0.290783 +/- 0.006515 | 0.455597 +/- 0.007212 |
| 200 | restricted_radial | 1.328433 +/- 0.093030 | 0.076609 +/- 0.000021 | 0.079248 +/- 0.002303 | 0.264684 +/- 0.000786 | 0.231697 +/- 0.005080 | 0.411723 +/- 0.005088 |
| 694 | full | 1.346458 +/- 0.057131 | 0.072158 +/- 0.000028 | 0.065649 +/- 0.000655 | 0.202214 +/- 0.002085 | 0.412411 +/- 0.015129 | 0.514136 +/- 0.008600 |
| 694 | restricted_radial | 1.327016 +/- 0.100705 | 0.073594 +/- 0.000006 | 0.070370 +/- 0.002741 | 0.252850 +/- 0.011605 | 0.310655 +/- 0.010002 | 0.450258 +/- 0.003890 |

Ti3Al: archived fixed Cmcm geometry; DFT DOS only, force constants unavailable. Seed SD is not uncertainty across dataset splits.
{}
