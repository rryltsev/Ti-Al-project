"""Stage a Ti-Al CPU job, reconcile its Slurm state and collect its named outputs.

Uses this workspace's pinned UMT SSH configuration, hosted in MLIP-research.
Each attempt gets an immutable directory below the project root. Pass the
input files to 'submit', then use the same unique run name with 'status' or
'fetch'. The payload's entry point must be named job.sh.
"""
import argparse
import hashlib
import importlib.util
import io
import json
import re
import shlex
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REMOTE = "/home6/u3129/tial-pd-20261124-atomistic"
SSH_SOURCE = HERE.parents[2] / "MLIP-research" / "scripts" / "cluster_ssh.py"


def ssh_args():
    if not SSH_SOURCE.is_file():
        return ["ssh", "-o", "BatchMode=yes", "mlip-umt"]
    spec = importlib.util.spec_from_file_location("umt_ssh", SSH_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ssh_args() + ["-o", "BatchMode=yes", "mlip-umt"]


def ssh(command, payload=None):
    return subprocess.run(
        ssh_args() + [command],
        input=payload, stdout=subprocess.PIPE, check=True,
    ).stdout


def digest(data):
    return hashlib.sha256(data).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, data):
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def submit(args, record, remote):
    files = {p.name: p.read_bytes() for p in args.files}
    if len(files) != len(args.files) or "job.sh" not in files:
        raise ValueError("Supply job.sh and uniquely named inputs")
    sums = {name: digest(data) for name, data in files.items()}
    hashes = "".join(f"{sha}  {name}\n" for name, sha in sums.items()).encode()
    files["inputs.sha256"] = hashes
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    job_name = "ta-" + args.run
    partitions = "apollo,nompi,debug" if args.minutes <= 90 else "apollo,nompi"
    sbatch = ["sbatch", "--parsable", f"--partition={partitions}", "--account=imet-2026-2",
              "--qos=normal", "--nodes=1", "--ntasks=1", f"--cpus-per-task={args.cpus}",
              f"--mem={args.memory}G", f"--time={args.minutes}",
              f"--job-name={job_name}", "--output=slurm-%j.out", "job.sh"]
    if getattr(args, "exclude", None):
        sbatch.insert(-1, "--exclude=" + args.exclude)
    if getattr(args, "dependency", None):
        if not re.fullmatch(r"\d+(?::\d+)*", args.dependency):
            raise ValueError("Dependencies must be numeric Slurm IDs separated by colon")
        sbatch[-1:-1] = ["--dependency=afterok:" + args.dependency, "--kill-on-invalid-dep=yes"]
    record.mkdir(parents=True, exist_ok=False)
    save(record / "intent.json", dict(host="umt.imm.uran.ru", user="u3129",
         remote=remote, utc=stamp(), job_name=job_name, sbatch=sbatch, sha256=sums))
    ssh(f'set -eu; test "$(id -un)" = u3129; umask 077; mkdir -p {REMOTE}; '
        f'mkdir {remote}; cd {remote}; tar -xzf -; sha256sum -c inputs.sha256 >/dev/null',
        archive.getvalue())
    out = ssh(f"set -eu; cd {remote}; {shlex.join(sbatch)} > job.id; cat job.id")
    jobid = out.decode().strip().split(";")[0]
    if not jobid.isdigit():
        raise RuntimeError(f"Ambiguous sbatch reply {out!r}. Inspect {remote}/job.id and sacct.")
    save(record / "submission.json", dict(job_id=jobid, utc=stamp()))
    print(jobid)


def status(args, record, remote):
    # job.id is durable when transport breaks immediately after acceptance.
    j = ssh(f"cat {remote}/job.id").decode().strip().split(";")[0]
    if not j.isdigit():
        raise RuntimeError(f"No confirmed job ID. Reconcile sacct for ta-{args.run} before resubmission.")
    data = ssh(f"export LC_ALL=C; squeue -h -j {j} -o '%i|%T|%M|%R'; "
               f"sacct -nP -j {j} --format=JobID,State,ExitCode,Elapsed,AllocCPUs,AllocTRES%100,Partition,NodeList;"
               f" if test -f {remote}/slurm-{j}.out; then tail -20 {remote}/slurm-{j}.out; fi").decode()
    (record / "last_status.txt").write_text(data, encoding="utf-8")
    print(data)


def fetch(args, record, remote):
    outdir = record / "outputs"
    outdir.mkdir(exist_ok=True)
    for name in args.names:
        if not re.fullmatch(r"[\w.-]+", name) or name in (".", ".."):
            raise ValueError("Only filenames directly in the job directory are accepted")
    quoted = shlex.join(args.names)
    raw = ssh(f"set -eu; cd {remote}; tar -czf - -- {quoted}")
    expected = ssh(f"set -eu; cd {remote}; sha256sum -- {quoted}").decode()
    expected = {name: sha for sha, name in (line.split(maxsplit=1) for line in expected.splitlines())}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        for name in args.names:
            member = tar.getmember(name)
            if not member.isfile():
                raise ValueError(name)
            data = tar.extractfile(member).read()
            if digest(data) != expected[name]:
                raise ValueError("Remote/local checksum mismatch for " + name)
            dest = outdir / name
            if dest.exists() and digest(dest.read_bytes()) != expected[name]:
                raise FileExistsError(dest)
            if not dest.exists():
                dest.write_bytes(data)
    manifest = outdir / "verified.sha256"
    previous = {}
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            sha, fname = line.split(maxsplit=1)
            if not fname.endswith(".part") and fname != "raw.sha256":
                previous[fname] = sha
    previous.update(expected)
    manifest.write_text("".join(f"{previous[name]}  {name}\n" for name in sorted(previous)), encoding="utf-8")
    print(outdir)


def fetch_raw(record, remote):
    """Stream already-compressed force data, preserving the digest on both hosts."""
    target = record / "outputs" / "results.tar.gz"
    target.parent.mkdir(exist_ok=True)
    metadata = ssh(f"set -eu; cd {remote}; sha256sum results.tar.gz; stat -c%s results.tar.gz").decode().splitlines()
    expected = metadata[0].split()[0]
    size = int(metadata[1])
    source = target if target.exists() else target.with_suffix(".gz.part")
    if not target.exists():
        offset = source.stat().st_size if source.exists() else 0
        if offset > size:
            raise ValueError(f"Oversized partial download for {remote}")
        if offset:
            with source.open("rb") as f:
                local_prefix = hashlib.file_digest(f, "sha256").hexdigest()
            remote_prefix = ssh(f"cd {remote} && head -c {offset} results.tar.gz | sha256sum").decode().split()[0]
            if local_prefix != remote_prefix:
                raise ValueError(f"Partial archive differs from source in {remote}")
        if offset < size:
            with source.open("ab") as f:
                subprocess.run(ssh_args() + [f"cd {remote} && tail -c +{offset+1} results.tar.gz"],
                               stdout=f, check=True)
    with source.open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    if actual != expected:
        raise RuntimeError(f"Raw force archive checksum mismatch for {remote}")
    if source != target:
        source.replace(target)
    (target.parent / "raw.sha256").write_text(f"{actual}  results.tar.gz\n")
    print(target, flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    s = sub.add_parser("submit")
    s.add_argument("run")
    s.add_argument("--cpus", type=int, default=1)
    s.add_argument("--memory", type=int, default=2)
    s.add_argument("--minutes", type=int, default=10)
    s.add_argument("--exclude", help="Slurm nodes to exclude after a documented node failure")
    s.add_argument("--dependency", help="run only after these Slurm IDs succeed (colon-separated)")
    s.add_argument("files", type=Path, nargs="+")
    sub.add_parser("status").add_argument("run")
    f = sub.add_parser("fetch")
    f.add_argument("run")
    f.add_argument("names", nargs="+")
    args = cli.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,48}", args.run):
        cli.error("Choose a short lower-case run identifier")
    globals()[args.command](args, HERE / "runs" / args.run, REMOTE + "/" + args.run)


if __name__ == "__main__":
    main()
