"""Independent structural analysis of saved Ti coexistence trajectories.

No MD or source-file changes.  Run with the Python used for the October audit.
Order parameters follow Lechner & Dellago, JCP 129, 114707 (2008), with
the 14 nearest neighbours (the two first shells of ideal bcc).
"""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "oct20" / "deps"))
import json
import hashlib
import re
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import sph_harm_y
from scipy.ndimage import gaussian_filter1d

SOURCE = Path(r"C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project\MD-results\melting")
ROOT = SOURCE / "beta_Ti_two-phase"
THRESHOLDS = np.array([.20, .25, .30, .35])


def read_dump(path):
    frames = []
    with path.open() as f:
        while True:
            header = f.readline()
            if not header:
                break
            assert header.strip() == "ITEM: TIMESTEP", (path, header)
            step = int(f.readline())
            assert f.readline().strip() == "ITEM: NUMBER OF ATOMS"
            n = int(f.readline())
            bounds_header = f.readline()
            bounds = np.array([list(map(float, f.readline().split())) for _ in range(3)])
            # All supplied files declare a triclinic box but have zero tilt.
            if bounds.shape[1] == 3:
                assert np.max(np.abs(bounds[:, 2])) < 1e-12, path
            assert "pp pp pp" in bounds_header
            cols = f.readline().split()[2:]
            raw = np.fromstring(" ".join(f.readline() for _ in range(n)), sep=" ").reshape(n, len(cols))
            ids = raw[:, cols.index("id")].astype(int)
            raw = raw[np.argsort(ids)]
            assert np.array_equal(np.sort(ids), np.arange(1, n + 1))
            xyz = raw[:, [cols.index(c) for c in ("x", "y", "z")]]
            lengths = bounds[:, 1] - bounds[:, 0]
            xyz = (xyz - bounds[:, 0]) % lengths
            frames.append((step, lengths, xyz, raw[:, cols.index("c_2")]))
    return frames


def order(xyz, lengths, k=14):
    _, idx = cKDTree(xyz, boxsize=lengths).query(xyz, k=k+1)
    idx = idx[:, 1:]
    dr = xyz[idx] - xyz[:, None, :]
    dr -= np.rint(dr / lengths) * lengths
    r = np.linalg.norm(dr, axis=2)
    theta = np.arccos(np.clip(dr[:, :, 2] / r, -1, 1))
    phi = np.arctan2(dr[:, :, 1], dr[:, :, 0])
    qlm = np.stack([sph_harm_y(6, m, theta, phi).mean(axis=1) for m in range(-6, 7)], axis=1)
    qbar_lm = (qlm + qlm[idx].sum(axis=1)) / (k + 1)
    q6 = np.sqrt(4*np.pi/13 * np.sum(np.abs(qlm)**2, axis=1))
    qbar6 = np.sqrt(4*np.pi/13 * np.sum(np.abs(qbar_lm)**2, axis=1))
    norm = qlm / np.maximum(np.linalg.norm(qlm, axis=1, keepdims=True), 1e-15)
    coherence = np.real(np.sum(norm[:, None, :].conj() * norm[idx], axis=2))
    bonds = (coherence > .7).sum(axis=1)
    return q6, qbar6, bonds


def ideal_check():
    xyz = np.array([[i+x, j+y, k+z] for i in range(4) for j in range(4)
                    for k in range(4) for x,y,z in [(0,0,0), (.5,.5,.5)]])
    q, qb, bonds = order(xyz, np.array([4.,4.,4.]))
    assert np.allclose(q, .5106882308569509, atol=1e-10), q
    assert np.allclose(qb, q, atol=1e-12)
    assert np.all(bonds == 14)
    return float(q.mean())


def process(path, pilot=False):
    model = path.parents[2].name
    temperature = int(re.search(r"T(\d+)K", str(path)).group(1))
    run = path.name.split("_")[0]
    key = f"{model}_{temperature}_{run}"
    target = HERE / (key + (".pilot.npz" if pilot else ".npz"))
    if target.exists():
        return
    cache = json.loads((path.parents[1]/"cache"/"index.json").read_text())
    params = next(c["params"] for c in cache if c["id"] == run)
    frames = read_dump(path)
    assert len(frames) == 101 and len(frames[0][2]) == 1296, key
    if pilot:
        frames = [frames[i] for i in [0, 10, 50, 100]]
    arrays = {k: [] for k in ("step", "lengths", "xyz", "pe", "q6", "qbar6", "bonds", "profile", "solid_fraction")}
    for step, lengths, xyz, pe in frames:
        q, qb, bonds = order(xyz, lengths)
        bins = np.minimum((xyz[:, 2] / lengths[2] * 36).astype(int), 35)
        counts = np.bincount(bins, minlength=36)
        profile = np.bincount(bins, weights=qb, minlength=36) / np.maximum(counts, 1)
        vals = [step, lengths, xyz, pe, q, qb, bonds, profile, (qb[:, None] > THRESHOLDS).mean(axis=0)]
        for k, v in zip(arrays, vals):
            arrays[k].append(v)
    np.savez_compressed(target, **{k: np.array(v) for k, v in arrays.items()},
                        dt_ps=params["dt"], thresholds=THRESHOLDS, source=str(path),
                        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        model=model, temperature=temperature, run=run, params=json.dumps(params))
    frac = np.array(arrays["solid_fraction"])[:, 1]
    print(key, "f(qbar6>.25):", np.round(frac[[0, len(frac)//2, -1]], 3), flush=True)


if __name__ == "__main__":
    HERE.mkdir(exist_ok=True)
    print("Ideal bcc q6:", ideal_check(), flush=True)
    paths = sorted(ROOT.glob("*/*/dumps/*_Step4_Melt.xyz"))
    print("Production files:", len(paths), flush=True)
    start = time.time()
    for path in paths:
        process(path, pilot="--pilot" in sys.argv)
    print(f"Completed in {time.time()-start:.1f} s", flush=True)
