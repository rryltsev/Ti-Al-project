"""Retrieve completed isotherms and assemble the verified full MC array."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import umt_run


def completed_ids(path, account):
    raw = umt_run.remote(f'cd {path} && find samples -type f -name "*.csv.gz" -printf "%f\\n"',account=account)
    return {int(s.split(".")[0]) for s in raw.decode().splitlines()}


def combine(run, groups, job):
    target = run / "samples.csv.gz"
    header = None
    logs = []
    n_windows = 0
    with gzip.open(str(target)+".part", "wt", newline="") as out:
        for (model, t), ids in groups.items():
            part = run / "sections" / f"{model}_{t}"
            file = part / "samples.csv.gz"
            receipt = json.loads((part / "retrieved.json").read_text())
            if (int(receipt["windows"]) != len(ids) or receipt["job_id"] != job or
                    hashlib.sha256(file.read_bytes()).hexdigest() != receipt["sha256_combined"]):
                raise ValueError(f"Invalid receipt in {part}")
            with gzip.open(file, "rt", newline="") as src:
                line = src.readline()
                if header is None:
                    header = line
                    out.write(line)
                if line != header:
                    raise ValueError(f"Incompatible sample columns in {part}")
                out.writelines(src)
            section_logs = (part / "window_logs.txt").read_text().splitlines()
            if {int(v.split(":")[0]) for v in section_logs} != set(ids):
                raise ValueError(f"Window log indices differ in {part}")
            n_windows += len(ids)
            logs.extend(section_logs)
    Path(str(target)+".part").replace(target)
    (run / "window_logs.txt").write_text("\n".join(logs)+"\n")
    umt_run.save(run / "retrieved.json", dict(job_id=job, windows=n_windows,
        sha256_combined=hashlib.sha256(target.read_bytes()).hexdigest(), source="verified native-array isotherm archives"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("series", nargs="?", default="atat_s3_5120",
                        choices=["atat_s3_5120", "atat_s3_repeats", "hybrids_1000K"])
    parser.add_argument("--account", choices=["u3129", "u2626"])
    args = parser.parse_args()
    series = args.series
    hybrid = series == "hybrids_1000K"
    if hybrid and not args.account:
        parser.error("select --account for the hybrid series")
    if not hybrid and args.account == "u2626":
        parser.error("historical series are stored under u3129")
    account = args.account or "u3129"
    run = (umt_run.HYBRIDS/"runs"/account if hybrid else umt_run.HERE/"runs"/series)
    submitted = json.loads((run / "submission.json").read_text())
    job = submitted["job_id"]
    groups = {}
    for idx, row in enumerate(csv.reader((run / "windows.tsv").open(), delimiter="\t"), 1):
        groups.setdefault((row[0], int(row[1])), []).append(idx)
    pending = [k for k in groups if not (run / "sections" / f"{k[0]}_{k[1]}" / "retrieved.json").is_file()]
    while pending:
        try:
            done = completed_ids(submitted["remote"], account)
        except subprocess.CalledProcessError as e:
            if e.returncode != 255:
                raise
            time.sleep(30)
            continue
        for k in list(pending):
            if all(i in done for i in groups[k]):
                subprocess.run([sys.executable, str(umt_run.HERE/"umt_run.py"), "fetch", series,
                                "--slice", k[0], str(k[1]), "--account", account], check=True)
                pending.remove(k)
                print("Verified", *k, flush=True)
        if pending:
            print(f"{len(done)} / {sum(map(len, groups.values()))} windows", flush=True)
            time.sleep(30)
    combine(run, groups, job)
    if hybrid:
        subprocess.run([sys.executable, str(umt_run.HYBRIDS/"verify_orbits.py"), str(run/"samples.csv.gz"),
                        "--output", str(run/"orbit_check.json")], check=True)
    print("Verified complete MC array", flush=True)


if __name__ == "__main__":
    main()
