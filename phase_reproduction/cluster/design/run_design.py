"""Execute selected rows of a frozen, independently generated Ti-Al design."""
import concurrent.futures
import csv
import gzip
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import time
import shutil
from pathlib import Path

MLP = "/home/u3129/miniconda3/envs/deepmd_320_a100/bin/mlp"


def reuse_completed(resume_from, dest, row, expected_hash, static):
    if resume_from is None:
        return False
    source = resume_from / "results" / dest.name
    required = ["ideal.cfg", "relaxed.cfg", "label.json"]
    if not static:
        required += ["raw_forces.npz", "mesh.npz", "phonopy.yaml.gz"]
    if not all((source / name).is_file() for name in required):
        return False
    label = json.loads((source / "label.json").read_text())
    if (label["source_cell_sha256"] != expected_hash or
            label["model_sha256"] != hashlib.sha256(Path("fitted.mtp").read_bytes()).hexdigest() or
            label["volume_ratio"] != float(row["ratio"]) or label["fmax_eV_A"] > 1e-5):
        raise ValueError(f"Restart record does not match the frozen design: {source}")
    if (static and label["stable_mesh"] is not None) or (not static and label["stable_mesh"] is None):
        raise ValueError(f"Different label mode in {source}")
    if label["stable_mesh"] and not (source / "vibrations.csv").is_file():
        return False
    if not static:
        # A force-constant gzip could have been interrupted during writing.
        try:
            with gzip.open(source / "phonopy.yaml.gz", "rb") as f:
                hashlib.file_digest(f, "sha256")
        except (OSError, EOFError):
            return False
    shutil.copytree(source, dest)
    return True


def evaluate(row, expected, static, resume_from):
    cell = Path(row["cell"]).name
    if hashlib.sha256(Path(cell).read_bytes()).hexdigest() != expected[row["id"]]:
        raise ValueError("Wrong source CFG for " + row["id"])
    name = row["label_subdir"]
    dest = Path("results") / name
    if reuse_completed(resume_from, dest, row, expected[row["id"]], static):
        return dict(label=name, seconds=0, returncode=0, reused_from=resume_from.name)
    args = [sys.executable, "label_cell.py", "--mlp", MLP, "--model", "fitted.mtp",
            "--cell", cell, "--ratio", row["ratio"], "--output", str(dest)]
    if static:
        args.append("--static-only")
    else:
        args += ["--supercell", row["repeat_a"], row["repeat_b"], row["repeat_c"],
                 "--mesh", row["mesh"]]
    start = time.monotonic()
    with (Path("logs") / (name + ".log")).open("w") as f:
        proc = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT)
    yaml = dest / "phonopy.yaml"
    if proc.returncode == 0 and yaml.is_file():
        compressed = yaml.with_suffix(".yaml.gz")
        with yaml.open("rb") as src, gzip.open(compressed, "xb", compresslevel=2) as out:
            shutil.copyfileobj(src, out)
        with gzip.open(compressed, "rb") as check, yaml.open("rb") as original:
            if hashlib.file_digest(check, "sha256").digest() != hashlib.file_digest(original, "sha256").digest():
                raise ValueError(f"Force-constant archive checksum mismatch in {name}")
        yaml.unlink()
    return dict(label=name, seconds=time.monotonic()-start, returncode=proc.returncode)


def main():
    config = json.loads(Path("selection.json").read_text())
    design = list(csv.DictReader(Path("design.csv").open()))
    expected = {item["id"]: item["cell_sha256"] for item in design}
    ids = set(config.get("ids", expected))
    ratios = set(config["ratios"])
    jobs = [row for row in csv.DictReader(Path("jobs.csv").open())
            if row["id"] in ids and float(row["ratio"]) in ratios]
    if len(jobs) != len(ids)*len(ratios) or len({x["label_subdir"] for x in jobs}) != len(jobs):
        raise ValueError("Incomplete or duplicated selection")
    Path("results").mkdir()
    Path("logs").mkdir()
    previous = config.get("resume_run")
    resume_from = Path.cwd().parent / previous if previous else None
    with concurrent.futures.ThreadPoolExecutor(int(os.environ["SLURM_CPUS_PER_TASK"])) as pool:
        fut = [pool.submit(evaluate, row, expected, config["static_only"], resume_from) for row in jobs]
        outcomes = []
        for f in concurrent.futures.as_completed(fut):
            result = f.result()
            outcomes.append(result)
            print(json.dumps(result), flush=True)
    Path("outcomes.json").write_text(json.dumps(outcomes, indent=2))
    # The numerical fit needs only these small records. Full FCs and forces
    # remain archived separately so fitting does not depend on a large transfer.
    with tarfile.open("labels.tar.gz", "w:gz") as archive:
        for pattern in ("*/label.json", "*/vibrations.csv"):
            for path in sorted(Path("results").glob(pattern)):
                archive.add(path)
    with tarfile.open("results.tar.gz", "w:gz", compresslevel=2) as archive:
        archive.add("results")
        archive.add("logs")
    if any(r["returncode"] for r in outcomes):
        raise RuntimeError("One or more labels failed, see outcomes and logs")


if __name__ == "__main__":
    main()
