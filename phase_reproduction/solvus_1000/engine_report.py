"""Validate native Monte Carlo energies against MLIP and time real trial sweeps."""
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

rows = [s.split() for s in Path("engine_energies.txt").read_text().splitlines()]
native = [float(r[2]) for r in rows]
result, = list(Path(".").glob("mlip_cases.cfg*"))
mlip = result.read_text().splitlines()
target = [float(mlip[i + 1]) for i, row in enumerate(mlip) if row.strip() == "Energy"]
assert len(rows) == len(target) == 123
by_kind = {}
for (size, kind, _), val, ref in zip(rows, native, target):
    group = size + "_" + kind
    by_kind[group] = max(by_kind.get(group, 0), abs(val - ref))
if max(by_kind.values()) > 5e-8:
    raise ValueError(by_kind)
res = dict(max_absolute_energy_error_eV_cell=by_kind, primary_MLIP_frames=len(target),
           platform=platform.platform(), model_sha256=hashlib.sha256(Path("fitted.mtp").read_bytes()).hexdigest(),
           native_flags="-std=c++17 -O3 -march=native", performance=[])
for nx, nz in [(4, 4), (6, 6)]:
    prefix = f"mc_{nx}_{nz}"
    sweeps = 8
    tic = time.perf_counter()
    subprocess.run(["./sgmc", "fitted.mtp", str(nx), str(nz), ".35", "3.30", "10", ".4",
                    "627614", "window", "0", str(sweeps), "1", "1", prefix], check=True)
    secs = time.perf_counter()-tic
    res["performance"].append(dict(sites=2*nx*nx*nz, sweeps=sweeps, wall_seconds=secs,
                                   seconds_per_sweep=secs/sweeps,
                                   site_move_attempts_per_second=4*nx*nx*nz*sweeps/secs))
    print(res["performance"][-1], flush=True)
Path("engine_report.json").write_text(json.dumps(res, indent=2)+"\n")
