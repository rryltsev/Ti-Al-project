"""Run one native SGMC process per CPU, retaining traces and restart states for each window."""
import argparse
import csv
import json
import math
import os
import re
import signal
import subprocess
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def load_cases(manifest):
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty window manifest")
    ids = set()
    for row in rows:
        run = row["run"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run) or run in ids:
            raise ValueError("Invalid or duplicate run identifier: " + run)
        ids.add(run)
        nx, ny, nz = map(int, row["shape"].split("x"))
        if nx != ny or nx % 2 or min(nx, nz) < 2 or int(row["N"]) != 2*nx*ny*nz:
            raise ValueError("Inconsistent N or noncommensurate shape: " + run)
        burn, sweeps, stride = (int(row[k]) for k in ("burn", "sweeps", "stride"))
        if burn < 0 or stride <= 0 or (burn+sweeps)//stride <= burn//stride:
            raise ValueError("Expected at least one recorded production sweep: " + run)
        if not all(math.isfinite(float(row[k])) for k in ("mu_eV","K_eV","q0","cap")):
            raise ValueError("Nonfinite field or cage: " + run)
        if not math.isfinite(float(row["T_K"])) or float(row["T_K"]) <= 0:
            raise ValueError("Invalid temperature: " + run)
        if int(row["axial_free"]) not in (0,1) or not 0 <= int(row["seed"]) < 2**64:
            raise ValueError("Invalid axial flag or RNG seed: " + run)
        if row["start"] not in ("ordered","random","window"):
            state = (manifest.parent / row["start"]).resolve(strict=True)
            row["start"] = str(state)
    return rows


def run_case(row, binary, model, output, stop):
    run = row["run"]
    nx, _, nz = row["shape"].split("x")
    args = [str(binary), str(model), nx, nz, row["cap"], row["mu_eV"], row["K_eV"],
            row["q0"], row["seed"], row["start"], row["burn"], row["sweeps"],
            row["stride"], row["axial_free"], run, row["T_K"]]
    started = time.monotonic()
    if stop.is_set():
        return {"status": "interrupted_before_start"}
    with (output/(run+".log")).open("x", encoding="utf-8") as log:
        log.write(json.dumps(args)+"\n")
        log.flush()
        with subprocess.Popen(args, cwd=output, stdout=log, stderr=subprocess.STDOUT) as child:
            while child.poll() is None:
                if stop.wait(.25):
                    child.terminate()
                    break
            code = child.wait()
    report = {"status": "ok" if code == 0 else "failed", "exit_code": code,
              "elapsed_seconds": round(time.monotonic()-started, 2)}
    if code == 0:
        trace = output/(run+".csv")
        state = output/(run+".state")
        burn, sweeps, stride = (int(row[k]) for k in ("burn","sweeps","stride"))
        expected = (burn+sweeps)//stride
        with trace.open(newline="") as stream:
            count = 0
            for frame in csv.DictReader(stream):
                count += 1
                if int(frame["sweep"]) != count*stride or float(frame["T_K"]) != float(row["T_K"]):
                    raise ValueError(run + ": nonuniform or missing sweep")
        if count != expected or not state.stat().st_size:
            raise ValueError(run + ": incomplete trace or checkpoint")
        report["frames"] = count
    return report


def persist(output, status):
    tmp = output/"window_status.tmp"
    tmp.write_text(json.dumps(status, indent=2)+"\n", encoding="utf-8")
    tmp.replace(output/"window_status.json")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--manifest", type=Path, default=Path("windows.csv"))
    cli.add_argument("--executable", type=Path, default=Path("sgmc"))
    cli.add_argument("--model", type=Path, default=Path("fitted.mtp"))
    cli.add_argument("--output", type=Path, default=Path("."))
    cli.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    args = cli.parse_args()
    if args.workers < 1:
        cli.error("--workers must be positive")
    manifest = args.manifest.resolve(strict=True)
    binary = args.executable.resolve(strict=True)
    model = args.model.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = load_cases(manifest)
    files = [r["run"]+suffix for r in cases for suffix in (".log",".csv",".state")]
    for fname in files+["window_status.json","results.tar.gz"]:
        if (output/fname).exists():
            raise FileExistsError("Run destination already occupied: " + fname)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM, getattr(signal,"SIGUSR1",signal.SIGTERM)):
        signal.signal(sig, lambda *_: stop.set())
    status = {"workers": min(args.workers,len(cases)), "windows": {}}
    persist(output,status)
    with ThreadPoolExecutor(max_workers=status["workers"]) as pool:
        work = {pool.submit(run_case,r,binary,model,output,stop):r["run"] for r in cases}
        for future in as_completed(work):
            run = work[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"status":"failed", "error":str(exc)}
            status["windows"][run] = result
            persist(output,status)
            print(run,result,flush=True)
    tmp = output/"results.tar.gz.part"
    with tarfile.open(str(tmp), "w:gz", compresslevel=3) as archive:
        archive.add(str(manifest), arcname="windows.csv")
        for name in ["window_status.json","inputs.sha256"]+files:
            if (output/name).is_file():
                archive.add(str(output/name), arcname=name)
    tmp.replace(output/"results.tar.gz")
    return int(stop.is_set() or any(r["status"] != "ok" for r in status["windows"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
