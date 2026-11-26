"""Use the MLIP-research pinned SSH configuration; never copy credentials."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent.parent / "MLIP-research"
spec = importlib.util.spec_from_file_location("mlip_ssh", SOURCE / "scripts/cluster_ssh.py")
ssh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ssh)
CFG = json.loads((ROOT / "config/protocol.json").read_text())
REMOTE = CFG["remote_root"]
RECORD = ROOT / "runs" / CFG["id"]

def run(command):
    result = subprocess.run(ssh.ssh_args() + ["-o", "BatchMode=yes", "mlip-umt", command],
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return result.stdout

def save(name, data):
    RECORD.mkdir(parents=True, exist_ok=True)
    (RECORD / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

def stage():
    files = [ROOT / "config/protocol.json", *sorted((ROOT / "scripts").glob("*.py")),
             *sorted((ROOT / "scripts").glob("*.sbatch"))]
    manifest = {"run_id": CFG["id"], "created_utc": datetime.now(timezone.utc).isoformat(),
                "remote": REMOTE, "cluster": "umt.imm.uran.ru", "files": {}}
    for p in files:
        manifest["files"][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest["source_project_commit"] = subprocess.check_output(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    save("input-manifest.json", manifest)
    print(run("umask 077; mkdir -p /home6/u3129/tial-phonons/envs; mkdir " + shlex.quote(REMOTE)))
    # Transfer only source/configuration. Each destination is created exclusively.
    for p in files + [RECORD / "input-manifest.json"]:
        dest = REMOTE + "/" + p.name
        command = "set -e; set -o noclobber; cat > " + shlex.quote(dest)
        subprocess.run(ssh.ssh_args() + ["-o", "BatchMode=yes", "mlip-umt", command],
                       input=p.read_bytes(), check=True)
    print(run("cd " + shlex.quote(REMOTE) + "; sha256sum protocol.json workflow.py setup.sbatch production.sbatch"))

def submit(stage_name, dependency=None):
    if stage_name not in ["setup", *CFG["functionals"]]:
        raise ValueError("Unknown stage: " + str(stage_name))
    filename = "setup" if stage_name == "setup" else stage_name
    intent = RECORD / (filename + "-intent.json")
    if intent.exists():
        raise RuntimeError("Submission intent already exists; reconcile queue/accounting before any retry")
    args = ["sbatch", "--parsable", "--partition=apollo", "--account=imet-2026-2", "--qos=normal",
            "--nodes=1", "--job-name=ti-ph-" + stage_name, "--chdir=" + REMOTE,
            "--output=" + REMOTE + "/" + stage_name + "-%j.out"]
    if stage_name == "setup":
        args += ["--ntasks=1", "--cpus-per-task=1", "--mem=4G", "--time=00:20:00", REMOTE + "/setup.sbatch"]
    else:
        args += ["--exclude=tesla-a101"]  # User prohibits all VASP use of the A100 host.
        args += ["--ntasks=16", "--cpus-per-task=1", "--mem=32G", "--time=12:00:00"]
        if dependency:
            args += ["--dependency=afterok:" + dependency, "--kill-on-invalid-dep=yes"]
        args += [REMOTE + "/production-v2.sbatch", stage_name]
    data = {"utc": datetime.now(timezone.utc).isoformat(), "command": args, "state": "submission_intent"}
    save(intent.name, data)
    command = "cd " + shlex.quote(REMOTE) + "; " + shlex.join(args) + " | tee " + shlex.quote(filename + ".jobid")
    job = run("set -o pipefail; " + command).strip()
    if not job.split(";")[0].isdigit():
        raise RuntimeError("Ambiguous sbatch response: " + job)
    save(filename + "-submission.json", dict(data, state="submitted", job_id=job.split(";")[0]))
    print(job)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["stage", "submit", "exec"])
    parser.add_argument("value", nargs="?")
    parser.add_argument("--dependency")
    a = parser.parse_args()
    if a.action == "stage": stage()
    elif a.action == "submit": submit(a.value, a.dependency)
    else: print(run(a.value))
