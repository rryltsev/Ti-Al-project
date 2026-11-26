"""Direct classical VCSGC isotherms and matched quantum importance-sampling diagnostic."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial import QhullError

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "classical_c46_1000"))
import paired_mc
from vcsgc import integrate


def interval(x):
    return np.percentile(x, [2.5, 97.5], axis=0).T.tolist()


def get_contact(means, mu):
    try:
        p = paired_mc.contact(means, mu)
    except (ValueError, QhullError):
        return None
    return None if p is None else p.tolist()


def reconstructed(means, c0, cfg):
    mu = cfg["mu"] - 2 * paired_mc.KT * cfg["kappa"] * (means - c0)
    return mu, get_contact(means, mu)


def summarize(path, nboot, seed, multiplier):
    cfg = json.loads((path / "manifest.json").read_text())
    t, n = cfg["T_K"], cfg["size"]
    assert cfg["source"] == "classical"
    paired_mc.KT = 8.617333262145e-5 * t
    grids = []
    for name in ("quantum", "classical"):
        with np.load(path.parent / (name + ".npz")) as data:
            assert data["temperature_K"].item() == t
            names = data["orbit_names"].astype(str).tolist()
            grids.append(CubicSpline(data["volume_A3_atom"], data["J_static"] + data["J_vib"][:, 0],
                                     extrapolate=False))
    _, traj = paired_mc.read_run(path, grids, names, multiplier)
    c0 = np.array(cfg["windows"])
    blocks = []
    halves = []
    diagnostics = []
    for i, r in enumerate(traj, 1):
        with gzip.open(path / "samples" / f"{i}.csv.gz", "rt") as f:
            assert all(float(row["T_K"]) == t and float(row["kappa"]) == cfg["kappa"]
                       for row in csv.DictReader(f))
        x, m, d = r["x"], r["m"], r["d"]
        w = np.exp(d - np.max(d))
        weights = np.stack((np.ones_like(w), x, w, w*x, m), axis=1)
        blocks.append(np.stack([weights[g].sum(axis=0) for g in r["groups"]]))
        halves.append([g.sum(axis=0) for g in np.array_split(weights, 2)])
        diagnostics.append(r["diag"])
    all_sums = np.stack([b.sum(axis=0) for b in blocks])
    def reduce_weights(data):
        return [reconstructed(data[:, i+1] / data[:, i], c0, cfg) for i in (0, 2)]
    pairs = reduce_weights(all_sums)
    rng = np.random.default_rng(seed)
    bs_contacts = [[], []]
    adjacent = []
    delta = []
    for _ in range(nboot):
        data = np.array([b[rng.integers(len(b), size=len(b))].sum(axis=0) for b in blocks])
        variants = reduce_weights(data)
        adjacent.append(np.diff(variants[0][0]))
        for rec, (_, contact) in zip(bs_contacts, variants):
            if contact is not None:
                rec.append(contact)
        if all(q[1] is not None for q in variants):
            delta.append(np.subtract(variants[0][1], variants[1][1]))
    halves = np.asarray(halves)
    x = all_sums[:, 1] / all_sums[:, 0]
    xq = all_sums[:, 3] / all_sums[:, 2]
    m = all_sums[:, 4] / all_sums[:, 0]
    mu = pairs[0][0]
    _, _, facets = integrate(x, mu, .005)
    for facet in facets:
        at = [facet[s] / 100 for s in ("alpha_at_pct", "alpha2_at_pct")]
        facet["mean_order_left_right"] = np.interp(at, x, m).tolist()
    result = dict(T_K=t, N=n, n_windows=len(x), equilibrate=cfg["eq"], production=cfg["prod"],
                  stride=cfg["stride"], size_shape=cfg["shape"], start=cfg["start"],
                  range_x_Al=x[[0,-1]].tolist(), contact_columns=["left_Al_at_pct","right_Al_at_pct","delta_mu_eV"],
                  class_direct_tangent=pairs[0][1], quantum_reweight_tangent=pairs[1][1],
                  classical_min_adjacent_dmu_eV=float(min(np.diff(mu))), raw_classical_facets=facets,
                  classical_all_adjacent_increasing=int((np.asarray(adjacent).min(axis=1)>0).sum()),
                  bootstrap_draws=nboot, bootstrap_window_block_tau_multiplier=multiplier,
                  lowest_number_of_blocks=min(len(b) for b in blocks),
                  class_bootstrap_resolved=len(bs_contacts[0]), quantum_bootstrap_resolved=len(bs_contacts[1]),
                  classical_95=interval(bs_contacts[0]) if len(bs_contacts[0])>1 else None,
                  quantum_reweighted_95=interval(bs_contacts[1]) if len(bs_contacts[1])>1 else None,
                  class_minus_quantum_95=interval(delta) if len(delta)>1 else None,
                  early_late_classical=[reduce_weights(halves[:, k])[0][1] for k in range(2)],
                  quantum_min_window_Kish_fraction=min(d["target_Kish_fraction"] for d in diagnostics),
                  maximum_energy_reconstruction_error_eV_atom=max(d["energy_check_eV_atom"] for d in diagnostics),
                  mean_order_range=[float(min(m)),float(max(m))])
    result["every_second_window_classical_tangent"] = get_contact(x[::2],mu[::2])
    imax = int(np.argmax(m))
    result["maximum_mean_order"] = dict(x_Al=float(x[imax]),m=float(m[imax]))
    if t == 600:
        xa, mua = np.loadtxt(path.parent / "author_window_means.csv", delimiter=",", skiprows=1).T
        aq = get_contact(xa, mua)
        result["author_quantum"] = aq
        if len(x) == len(xa):
            change = x - xq
            result["historical_shifted_by_classical_minus_quantum"] = get_contact(
                xa+change, mua-2*paired_mc.KT*cfg["kappa"]*change)
    cis = np.asarray(interval(adjacent))
    with (path / "window_stats.csv").open("w", newline="") as f:
        wcsv=csv.writer(f)
        wcsv.writerow(("c0","x_classical","mu_classical_eV","m_order","x_quantum_reweighted",
                       "mu_quantum_reweighted_eV","Kish_fraction","correlation_blocks",
                       "next_dmu_classical_eV_lower95","next_dmu_classical_eV_upper95"))
        for i in range(len(x)):
            ci = cis[i] if i+1 < len(x) else ["", ""]
            wcsv.writerow((c0[i], x[i], mu[i], m[i], xq[i], pairs[1][0][i],
                           diagnostics[i]["target_Kish_fraction"], len(blocks[i]), *ci))
    (path / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ("T_K","N","class_direct_tangent","classical_95",
                        "class_bootstrap_resolved","classical_min_adjacent_dmu_eV",
                        "quantum_min_window_Kish_fraction","quantum_reweight_tangent")}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path, nargs="+")
    p.add_argument("--bootstrap", type=int, default=384)
    p.add_argument("--seed", type=int, default=3642168)
    p.add_argument("--block-multiplier", type=float, default=6)
    a = p.parse_args()
    for folder in a.run:
        summarize(folder,a.bootstrap,a.seed,a.block_multiplier)
