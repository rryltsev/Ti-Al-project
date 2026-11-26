"""Independent heavy-tailed check of all finite-cell partitions in a specified HCP basin."""
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.linalg import null_space
from scipy.special import logsumexp

from hcp_orbits import hcp_geometry
from pilot import Frame, HASH, evaluate, force_constants, relax_fixed_cell, site_planes
from tail_check import KT, proposal


CAP = .40
BLOCKS = 4
COUNT = 1024


def sample(task):
    n, row, binary, model = task
    shape = (2,2,1) if n == 8 else (2,2,2)
    cell, sites = hcp_geometry(shape)
    ids = row["al_indices"] if n == 8 else row["representative_Al_indices"]
    name = row["name"] if n == 8 else row["id"]
    types = np.zeros(n, dtype=int)
    types[ids] = 1
    def calc(frames):
        return evaluate(frames, binary, model, 3)
    if n == 8:
        frame = relax_fixed_cell(Frame(cell, sites, types), calc)
        base = row["static_eV_cell"] + row["harmonic_classical_eV_cell"]
    else:
        frame = calc([Frame(cell, np.array(row["relaxed_positions_A"]), types)])[0]
        base = row["energy_eV_cell"] + row["harmonic_config_eV_cell"]
    basis = null_space(np.tile(np.eye(3),(1,n)))
    val, vec = np.linalg.eigh(basis.T @ force_constants(calc,frame,.0025) @ basis)
    if val.min() <= 0:
        raise ValueError("Unstable " + name)
    transform = (basis @ vec * np.sqrt(KT/val)).T
    planes = site_planes(cell, sites)
    mask = sum(1 << i for i in ids)
    rows, raw = [], []
    for b in range(BLOCKS):
        rng = np.random.default_rng([1135912,n,mask,b])
        z, r2, ratio = proposal(COUNT,len(val),rng)
        pos = frame.positions + (z @ transform).reshape(COUNT,n,3)
        w = np.einsum("sni,nki->snk",pos-sites,planes).max(axis=(1,2))
        inside = np.flatnonzero(w < CAP)
        diff = np.zeros(COUNT)
        for start in range(0,len(inside),256):
            indices = inside[start:start+256]
            energy = [v.energy for v in calc([Frame(cell,pos[i],types) for i in indices])]
            diff[indices] = np.asarray(energy) - (frame.energy+.5*KT*r2[indices])
        # The unrestricted harmonic partition supplies the analytic normalizer
        # in base. Both confined partition estimates use the original draw count.
        lh = np.where(w < CAP,ratio,-np.inf)
        la = lh-diff/KT
        rows.append([logsumexp(lh)-np.log(COUNT), logsumexp(la)-np.log(COUNT)])
        raw.append([diff,ratio,w])
    np.savez_compressed(f"cage_{name}.npz",data=np.asarray(raw),
                        columns=["excess_eV_cell","log_harmonic_over_proposal","wall_fraction"],
                        rng_seed=[1135912,n,mask],cap=CAP)
    return dict(n=n,k=len(ids),name=name,deg=row["degeneracy"],base=base,log_block_ratio=rows)


def estimate(rows, subset):
    states = {}
    for row in rows:
        block = np.array(row["log_block_ratio"])[subset]
        states[row["name"]] = row["base"]-KT*(logsumexp(block,axis=0)-np.log(len(block)))
    def statistic(k):
        selected = [r for r in rows if r["n"] == 16 and r["k"] == k]
        prob = [np.log(r["deg"])-states[r["name"]]/KT for r in selected]
        return -KT*logsumexp(prob,axis=0)
    curvature = statistic(4)-2*statistic(3)+statistic(2)
    result = {"D16_harmonic_meV_cell":1000*curvature[0],
              "D16_MTP_meV_cell":1000*curvature[1],
              "D16_shift_meV_cell":1000*(curvature[1]-curvature[0])}
    for key in ("basal","interlayer"):
        gap = (states[key]-states["d019"])*125
        result.update({key+"_gap_HA_meV_atom": gap[0],key+"_gap_MTP_meV_atom":gap[1],
                       key+"_gap_shift_meV_atom":gap[1]-gap[0]})
    return result


if __name__ == "__main__":
    model = Path("fitted.mtp").resolve()
    if hashlib.sha256(model.read_bytes()).hexdigest() != HASH:
        raise ValueError("Wrong potential")
    n8 = json.loads(Path("pilot.json").read_text())
    n16 = json.loads(Path("n16_static.json").read_text())
    ce = json.loads(Path("c46_N8_1000K.json").read_text())
    if {n8["model_sha256"],n16["model_sha256"]} != {HASH}:
        raise ValueError("Inconsistent static provenance")
    tasks = [(8,dict(r,degeneracy=ce["orbits"][r["name"]]["degeneracy"]),sys.argv[1],str(model))
             for r in n8["results"]] + [(16,r,sys.argv[1],str(model)) for r in n16["results"]]
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(sample,tasks))
    totals = estimate(rows,slice(None))
    omitted = [estimate(rows,np.arange(BLOCKS) != i) for i in range(BLOCKS)]
    se = {k:float(np.sqrt((BLOCKS-1)*np.var([v[k] for v in omitted]))) for k in totals}
    Path("caged_check.json").write_text(json.dumps(dict(cap=CAP,model_sha256=HASH,blocks=BLOCKS,
        draws_per_block=COUNT,proposal="balanced standard Gaussian and scaled t5",
        estimates=totals,jackknife_se=se,rows=rows),indent=2,allow_nan=False))
