"""Difference between independent fixed-c/a atomistic and C46 endpoint estimates."""
import argparse
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent


def summarize(vector, draws):
    return {
        "value": vector.tolist(),
        "conditional_95": np.quantile(draws, [.025, .975], axis=0).T.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paired_mc_json", type=Path)
    parser.add_argument("--dense", type=Path, help="include the independent 25-center discretization check")
    parser.add_argument("--output", type=Path, default=HERE / "direct_comparison.json")
    args = parser.parse_args()
    mc = json.loads(args.paired_mc_json.read_text(encoding="utf-8"))
    assert mc["N"] == 5120 and mc["T_K"] == 1000 and mc["row_order"] == ["quantum", "classical"]
    assert mc["columns"][:2] == ["Al_at_percent_alpha", "Al_at_percent_alpha2"]
    mc_point = np.array(mc["estimates"])[:, :2]
    with np.load(args.paired_mc_json.with_suffix(".npz"), allow_pickle=False) as samples:
        mc_draws = [samples[method][:, :2] for method in mc["row_order"]]
    assert len(mc_draws[0]) == len(mc_draws[1]) == mc["bootstrap_resolved"]
    direct_path = HERE.parent / "solvus_1000/analysis1400_fixed.json"
    all_ensembles = json.loads(direct_path.read_text(encoding="utf-8"))
    direct, = [r for r in all_ensembles.values() if r["N"] == 1400 and r["cap"] == .4 and not r["axial_free"]]
    stats = direct["equal_single_variant"]
    direct_point = 100 * np.array([stats["x_alpha"], stats["x_alpha2"]])
    direct_draws = 100 * np.array(direct["bootstrap"]["joint_mu_xalpha_xalpha2"])[:, 1:3]
    result = dict(phase_order=["alpha", "alpha2"], composition="Al atomic percent",
                  difference_units="percentage points", paired_ce=mc["N"], direct_atoms=direct["N"],
                  T_K=1000, pressure_GPa=0, fixed_ca=4.6046 / 2.9446, direct_W_max=.40,
                  direct=summarize(direct_point, direct_draws),
                  sampling="Cartesian product of bootstrap draws from independent MTP and CE runs; "
                           "CE statistics use the same bootstrap blocks for paired differences. "
                           "Intervals condition on the chosen cell, cage, window grid and sample.",
                  source_mc=args.paired_mc_json.name, source_direct=direct_path.name)
    for label, point, draws in zip(mc["row_order"], mc_point, mc_draws):
        all_diff = (direct_draws[:, None, :] - draws[None, :, :]).reshape(-1, 2)
        result["direct_minus_" + label] = summarize(direct_point - point, all_diff)
    result["classical_minus_quantum"] = summarize(mc_point[1] - mc_point[0], mc_draws[1] - mc_draws[0])
    result["width_classical_minus_quantum"] = summarize(
        np.diff(mc_point[1] - mc_point[0]), np.diff(mc_draws[1] - mc_draws[0], axis=1))
    if args.dense:
        dense_fit = json.loads(args.dense.read_text(encoding="utf-8"))
        assert dense_fit["N"] == mc["N"] and len(dense_fit["window_centers"]) == 25
        point = np.array(dense_fit["primary"]["full_25"]["classical"][:2])
        with np.load(args.dense.with_suffix(".npz"), allow_pickle=False) as series:
            assert series["row_order"].tolist() == ["quantum", "classical"]
            samples = series["full_25"][:, 1, :2]
        diffs = (direct_draws[:, None, :] - samples[None, :, :]).reshape(-1, 2)
        result["direct_minus_classical_25_windows"] = summarize(direct_point - point, diffs)
        result["source_25_windows"] = args.dense.name
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("direct_minus_classical", "classical_minus_quantum")}, indent=2))


if __name__ == "__main__":
    main()
