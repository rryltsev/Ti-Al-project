"""Finite two-orbit interactions and common tangents for the ten native MC hybrids."""
import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from attribute import NAMES, ROOT, VREF, Tangents, write_csv


def read(path):
    with path.open(encoding="utf8") as file:
        return list(csv.DictReader(file))


def main(out, mc_config):
    orbit_rows = read(out/"orbit_attribution.csv")
    states = {row["state"]: row for row in read(out/"states.csv")}
    hybrids = json.loads(mc_config.read_text(encoding="utf8"))["models"]
    model = Tangents()
    cross, selected, differences = [], [], []
    for mode, first, last in (("static", "static_paper", "static_native"), ("HA", "HA_00", "HA_11")):
        rows = [r for r in orbit_rows if (r["mode"], r["branch"]) == (mode, "alpha")]
        assert tuple(r["orbit"] for r in rows) == NAMES
        js = [np.array([float(r[k]) for r in rows]) for k in ("J_paper_eV", "J_native_eV")]
        guess = [[float(states[key]["alpha_at_pct"])/100, float(states[key]["q_alpha2"]),
                  float(states[key]["r_alpha2"])] for key in (first, last)]
        endpoints = [model.solve(j, g) for j, g in zip(js, guess)]
        # Central differences at the path midpoint check the implicit derivatives
        # independently. Perturb by 1e-7 eV to stay inside each smooth branch.
        midpoint = (js[0]+js[1])/2
        middle = model.solve(midpoint, np.mean(guess, axis=0))
        for i, name in enumerate(NAMES):
            step = np.zeros(18)
            step[i] = 1.e-7
            cp = model.solve(midpoint+step, middle["x"], check=False)["c"]
            cm = model.solve(midpoint-step, middle["x"], check=False)["c"]
            numerical = 100*(cp-cm)/(2*step[i])
            analytic = 100*middle["derivatives"][:,i]
            differences.append(dict(mode=mode, orbit=name, step_eV=step[i],
                alpha_analytic_at_pct_eV=analytic[0], alpha_fd_at_pct_eV=numerical[0],
                alpha2_analytic_at_pct_eV=analytic[1], alpha2_fd_at_pct_eV=numerical[1],
                worst_relative_error=float(max(abs(analytic-numerical)/np.maximum(1.,abs(analytic))))))
        for item in (h for h in hybrids if h["model"] == mode):
            idx = 0 if item["base"]=="paper" else 1
            jswap = js[idx].copy()
            which = [NAMES.index(n) for n in item["replace"]["effective"]]
            jswap[which] = js[1-idx][which]
            native_file = mc_config.parent/"hybrids"/(item["name"]+".txt")
            mc = np.loadtxt(native_file)
            if not np.isclose(mc[0], VREF) or np.max(abs(mc[3:]-jswap))>5.e-12:
                raise ValueError(f"Unmatched MC Hamiltonian: {native_file}")
            result = model.solve(jswap, guess[idx])
            shift = 100*(result["c"]-endpoints[idx]["c"])
            selected.append(dict(model=item["name"], base=item["base"],
                alpha_at_pct=100*result["c"][0], alpha2_at_pct=100*result["c"][1],
                delta_alpha_vs_base_at_pct=shift[0], delta_alpha2_vs_base_at_pct=shift[1],
                m_alpha2=result["m"], mc_max_J_difference_eV=np.max(abs(mc[3:]-jswap))))
        write_csv(out/"mc_hybrid_mfa.csv", selected)
        for idx, label in enumerate(("paper", "matched")):
            j0 = js[idx]
            base = endpoints[idx]["c"]
            singles = []
            for i in range(18):
                j1 = j0.copy()
                j1[i] = js[1-idx][i]
                singles.append(model.solve(j1, guess[idx])["c"])
            for i, k in itertools.combinations(range(18), 2):
                j2 = j0.copy()
                j2[[i,k]] = js[1-idx][[i,k]]
                two = model.solve(j2, guess[idx])["c"]
                correction = 100*((two-base)-(singles[i]-base)-(singles[k]-base))
                cross.append(dict(mode=mode, base=label, orbit_1=NAMES[i], orbit_2=NAMES[k],
                    alpha_at_pct=100*two[0], alpha2_at_pct=100*two[1],
                    nonadditive_alpha_at_pct=correction[0], nonadditive_alpha2_at_pct=correction[1]))
            write_csv(out/"two_orbit_swaps.csv", cross)
        write_csv(out/"finite_differences.csv", differences)
    if max(d["worst_relative_error"] for d in differences)>2.e-6 or len(cross)!=612:
        raise ValueError("A derivative check or two-orbit comparison is incomplete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE/"results")
    parser.add_argument("--mc-config", type=Path, default=ROOT/"eci_boundary/mc/hybrids_1000K.json")
    args = parser.parse_args()
    main(args.out, args.mc_config)
