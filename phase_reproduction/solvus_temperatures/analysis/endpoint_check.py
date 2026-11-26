"""Independent K=0 phase-conditional controls for the temperature-specific direct MTP fits."""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "solvus_1000"))
from analysis import KB, tau_int


def phase_mean(samples, high, cutoff, field_shift, n):
    weights, data, inefficiencies, masks = [], [], [], []
    for r in samples:
        selected = (r["Q"] > cutoff) == high
        if selected.any():
            weights.append(field_shift * r["n_Al"][selected])
            data.append(np.column_stack([r[k][selected] for k in ("n_Al", "Q", "volume_A3_atom", "c_a")]))
            g = max(max(tau_int(r[k]) for k in ("n_Al", "Q", "volume_A3_atom", "c_a")), 1)
            inefficiencies.append(np.full(selected.sum(), g))
            masks.append(selected)
    if not weights:
        return None
    logs = np.concatenate(weights)
    w = np.exp(logs - logsumexp(logs))
    bootstrap = np.zeros((512, 2))
    rng = np.random.default_rng(63821)
    blocks_by_chain = []
    for logs_i, obs_i, gs, mask in zip(weights, data, inefficiencies, masks):
        # Keep the original time axis, including possible visits to the other basin.
        num_den = np.zeros((len(mask), 2))
        raw_w = np.exp(logs_i - np.max(logs))
        num_den[mask] = np.column_stack([raw_w * (100 * obs_i[:, 0] / n), raw_w])
        nblocks = max(1, int(len(mask) / (4 * gs[0])))
        blocks_by_chain.append(nblocks)
        totals = np.array([b.sum(axis=0) for b in np.array_split(num_den, nblocks)])
        draws = rng.integers(nblocks, size=(len(bootstrap), nblocks))
        bootstrap += totals[draws].sum(axis=1)
    valid = bootstrap[:, 1] > 0
    estimates = bootstrap[valid, 0] / bootstrap[valid, 1]
    interval = np.quantile(estimates, [.025, .975]).tolist() if min(blocks_by_chain) >= 2 else None
    return dict(means=w @ np.vstack(data), count=len(w), kish_frames=1 / (w @ w),
                approx_kish_after_time_correlation=1 / (np.concatenate(inefficiencies) @ (w * w)),
                conditional_95_Al_at_pct=interval, time_blocks_by_chain=blocks_by_chain,
                bootstrap_valid=int(valid.sum()), bootstrap_draws=len(bootstrap))


def run(fit_path, manifests, output):
    fits = json.loads(fit_path.read_text(encoding="utf-8"))
    chains = defaultdict(list)
    for m in manifests:
        with m.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if float(row["K_eV"]) != 0:
                    continue
                t, n, cap, axial, mu = float(row["T_K"]), int(row["N"]), float(row["cap"]), int(row["axial_free"]), float(row["mu_eV"])
                data = np.genfromtxt(m.parent / (row["run"] + ".csv"), delimiter=",", names=True)
                if not np.allclose(data["T_K"], t, rtol=0, atol=1e-8):
                    raise ValueError(f"Temperature mismatch {row['run']}")
                data = data[data["sweep"] > int(row["burn"])]
                if len(data) != int(row["sweeps"]) // int(row["stride"]):
                    raise ValueError(f"Incomplete series {row['run']}")
                chains[t, n, cap, axial, mu].append((row["start"], row["run"], data))
    results = []
    for (t,n,cap,axial,mu), windows in sorted(chains.items()):
        candidates = [x for x in fits if x["temperature_K"] == t and x["N"] == n
                      and x["cap"] == cap and x["axial_free"] == axial]
        if len(candidates) != 1:
            raise ValueError(f"Expected one matching fit {(t,n,cap,axial)}")
        fit = candidates[0]["coexistence"].get("equal_single_variant")
        if not fit:
            raise ValueError("A resolved transition is required for a phase-conditional estimate")
        target, cut = fit["mu_eV"], fit["order_cut"]
        grouped = {}
        for start in ["all", *sorted({s for s, _, _ in windows})]:
            selection = [r for s,_,r in windows if start == "all" or start == s]
            values = {}
            for phase, high in [("alpha", False), ("alpha2", True)]:
                average = phase_mean(selection, high, cut, (target-mu) / (KB*t), n)
                if average:
                    k, q, v, ca = average.pop("means")
                    average.update(Al_at_pct=100*k/n, mean_Q=q, V_A3_atom=v, c_a=ca)
                values[phase] = average
            grouped[start] = values
        results.append(dict(T_K=t,N=n,W_max=cap,axial_free=axial,source_mu_eV=mu,
                            target_mu_eV=target,Q_cut=cut,MBAR_at_pct=[100*fit["x_alpha"],100*fit["x_alpha2"]],
                            chains=[name for _,name,_ in windows], controls=grouped))
    output.write_text(json.dumps(dict(estimate="Conditional means from K=0 trajectories. No relative phase weight "
                         "or coexistence field is inferred from their number of starts.", rows=results),
                         indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("fit",type=Path)
    p.add_argument("manifests",nargs="+",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    run(a.fit,a.manifests,a.out)
