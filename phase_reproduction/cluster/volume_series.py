"""Run the regenerated ATAT design at the nine Ti-Al volumes."""
import argparse
import concurrent.futures
import json
from pathlib import Path
from types import SimpleNamespace

import umt_job

BASE = Path(__file__).resolve().parent
DATA = BASE.parent / "atomistic"
RATIOS = (1.00, .94, .96, .98, 1.02, 1.04, 1.06, 1.09, 1.12)


def stage_one(ratio, design):
    percent = round(100 * ratio)
    run = f"atat-v{percent}-01"
    record = BASE / "runs" / run
    if (record / "submission.json").is_file():
        return
    if record.exists():
        raise RuntimeError(f"Reconcile initial submission for {run}")
    selection = BASE / "design" / "volumes" / str(percent) / "selection.json"
    selection.parent.mkdir(exist_ok=True, parents=True)
    selection.write_text(json.dumps({"static_only": False, "ratios": [ratio]})+"\n")
    files = [BASE / "design" / "job.sh", BASE / "design" / "run_design.py", selection,
             design / "design.csv", design / "jobs.csv",
             BASE.parent / "inputs" / "fitted.mtp", DATA / "label_cell.py",
             DATA / "hcp_orbits.py", DATA / "mtp_cfg.py",
             *sorted((design / "cells").glob("*.cfg"))]
    cpus = 8 if percent == 100 else 3
    args = SimpleNamespace(files=files, run=run, cpus=cpus, memory=2*cpus, minutes=85)
    umt_job.submit(args, record, f"{umt_job.REMOTE}/{run}")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=["submit", "status", "fetch-labels", "fetch-raw"])
    cli.add_argument("--design", type=Path, help="frozen ATAT design containing cells, design.csv and jobs.csv")
    args = cli.parse_args()
    if args.action == "submit":
        if not args.design:
            cli.error("--design is required to submit an ATAT design")
        stage_one(1.0, args.design)
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            list(pool.map(lambda ratio: stage_one(ratio, args.design), RATIOS[1:]))
        return
    ids = {}
    for ratio in RATIOS:
        prefix = "atat-v%d-" % round(100*ratio)
        attempts = [p for p in sorted((BASE / "runs").glob(prefix+"[0-9][0-9]"))
                    if (p / "submission.json").is_file()]
        if not attempts:
            continue
        record = attempts[-1]
        run = record.name
        if (record / "submission.json").is_file():
            j = json.loads((record / "submission.json").read_text())["job_id"]
            ids[j] = run
    if not ids:
        raise ValueError("No recorded jobs")
    fields = "JobID,State,ExitCode,ElapsedRaw,AllocCPUs,NodeList,Partition"
    report = umt_job.ssh("LC_ALL=C sacct -X -nP -j " + ",".join(ids) + " --format=" + fields).decode()
    (BASE / "design" / "volumes" / "slurm.txt").write_text(fields+"\n"+report)
    print(report, flush=True)
    if args.action == "fetch-raw":
        def download(line):
            j, state, code, *_ = line.split("|")
            if state == "COMPLETED" and code == "0:0":
                run = ids[j]
                record = BASE / "runs" / run
                if not (record / "outputs" / "raw.sha256").exists():
                    umt_job.fetch_raw(record, f"{umt_job.REMOTE}/{run}")
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            list(pool.map(download, report.splitlines()))
    if args.action == "fetch-labels":
        for line in report.splitlines():
            j, state, code, *_ = line.split("|")
            if state != "COMPLETED" or code != "0:0":
                continue
            run, record = ids[j], BASE / "runs" / ids[j]
            if (record / "outputs" / "verified.sha256").exists():
                continue
            names = ["labels.tar.gz", "outcomes.json", "inputs.sha256", f"slurm-{j}.out", "job.id"]
            umt_job.fetch(SimpleNamespace(names=names), record, f"{umt_job.REMOTE}/{run}")


if __name__ == "__main__":
    main()
