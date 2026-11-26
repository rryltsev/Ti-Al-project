# GRACE N200: controlled mask comparison

2026-11-09T12:04:00.939886+00:00

All eight models: fixed final epoch50 / step10000, fp64, seed10. Energy alignment is fitted on train200 only. No held-out test access.

| XC | Mask | E raw meV/atom | E aligned meV/atom | F eV/A | Stress GPa | Band RMSE THz | Gamma singlet error THz |
|---|---|---:|---:|---:|---:|---:|---:|
| PBE | Full | 2.5596 | 1.3175 | 0.075899 | 0.069996 | 0.226194 | 0.029296 |
| PBE | Restricted | 1.4024 | 1.2948 | 0.078173 | 0.077110 | 0.313943 | 1.057826 |
| PBE | Restricted + MLP | 2.0766 | 1.3272 | 0.078218 | 0.077380 | 0.318192 | 1.068301 |
| PBE | Restricted + radial | 2.0201 | 1.3550 | 0.076584 | 0.079293 | 0.264114 | 0.541536 |
| r2SCAN | Full | 2.9798 | 1.5353 | 0.086085 | 0.080469 | 0.260594 | 0.104290 |
| r2SCAN | Restricted | 1.6083 | 1.4603 | 0.088937 | 0.086315 | 0.391991 | 1.481377 |
| r2SCAN | Restricted + MLP | 2.2779 | 1.5072 | 0.088948 | 0.086444 | 0.397403 | 1.491102 |
| r2SCAN | Restricted + radial | 2.3450 | 1.4892 | 0.086948 | 0.089502 | 0.326108 | 0.873772 |

One seed per mask: these differences do not quantify seed variability. Matched DFT geometry; central +/-0.01 A displacements, same ASR/symfc, mesh50^3. No inference about DFT convergence or long-time MD stability.

{
  "PBE": {
    "full": {
      "validation_reduction_vs_restricted_percent": {
        "E_raw_RMSE_meV_atom": -82.52088543903977,
        "E_aligned_RMSE_meV_atom": -1.757167772639856,
        "F_RMSE_eV_A": 2.908449933115631,
        "stress_RMSE_GPa": 9.225647408326598
      },
      "phonon_reduction_vs_restricted_percent": 27.95077157845186
    },
    "restricted_mlp": {
      "validation_reduction_vs_restricted_percent": {
        "E_raw_RMSE_meV_atom": -48.078927518901594,
        "E_aligned_RMSE_meV_atom": -2.5048580031006296,
        "F_RMSE_eV_A": -0.058502890585843126,
        "stress_RMSE_GPa": -0.34981288794364573
      },
      "phonon_reduction_vs_restricted_percent": -1.3533656238748026
    },
    "restricted_radial": {
      "validation_reduction_vs_restricted_percent": {
        "E_raw_RMSE_meV_atom": -44.04554618766594,
        "E_aligned_RMSE_meV_atom": -4.655206895329278,
        "F_RMSE_eV_A": 2.032039110780426,
        "stress_RMSE_GPa": -2.8303129801566795
      },
      "phonon_reduction_vs_restricted_percent": 15.87206963372526
    }
  },
  "r2SCAN": {
    "full": {
      "validation_reduction_vs_restricted_percent": {
        "E_raw_RMSE_meV_atom": -85.27351248388528,
        "E_aligned_RMSE_meV_atom": -5.138317907005807,
        "F_RMSE_eV_A": 3.2064444199770725,
        "stress_RMSE_GPa": 6.773381879829243
      },
      "phonon_reduction_vs_restricted_percent": 33.52046144660974
    },
    "restricted_mlp": {
      "validation_reduction_vs_restricted_percent": {
        "E_raw_RMSE_meV_atom": -41.63228475406626,
        "E_aligned_RMSE_meV_atom": -3.213284259703908,
        "F_RMSE_eV_A": -0.013380982261246288,
        "stress_RMSE_GPa": -0.14885707987242203
      },
      "phonon_reduction_vs_restricted_percent": -1.380555637480052
    },
    "restricted_radial": {
      "validation_reduction_vs_restricted_percent": {
        "E_raw_RMSE_meV_atom": -45.80715191324922,
        "E_aligned_RMSE_meV_atom": -1.9812199079833892,
        "F_RMSE_eV_A": 2.2363269870260383,
        "stress_RMSE_GPa": -3.692183762458079
      },
      "phonon_reduction_vs_restricted_percent": 16.807188273138163
    }
  }
}
