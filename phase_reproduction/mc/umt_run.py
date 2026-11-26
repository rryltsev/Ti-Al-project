"""Stage, submit, inspect and retrieve the paper's CPU lattice MC windows on UMT."""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
SSH_MODULE = HERE.parents[2] / "MLIP-research/scripts/cluster_ssh.py"
BASE = "/home6/u3129/tial-pd-20261124-ce"
HYBRIDS = HERE.parent/"eci_boundary/mc"


def connection(account):
    if account == "u3129":
        client = ["ssh"]
        if SSH_MODULE.is_file():
            spec = importlib.util.spec_from_file_location("umt_ssh", SSH_MODULE)
            ssh = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ssh)
            client = ssh.ssh_args()
        return client + ["-o","BatchMode=yes","-o","ConnectTimeout=45","mlip-umt"]
    if account == "u2626":
        return ["ssh","-o","BatchMode=yes","-o","ConnectTimeout=45","umt-u2626"]
    raise ValueError("Unknown UMT user")


def remote(command, payload=None, account="u3129"):
    return subprocess.run(connection(account) + [command],
                          input=payload, stdout=subprocess.PIPE, check=True).stdout


def windows(series):
    if series.startswith("atat_") or series == "hybrids_1000K":
        spec = (HYBRIDS/"hybrids" if series == "hybrids_1000K" else HERE/"inputs"/series)/"windows.tsv"
        with spec.open() as stream:
            for row in csv.reader(stream,delimiter="\t"):
                yield (row[0], *(int(x) for x in row[1:5]), float(row[5]),float(row[6]),
                       *(int(x) for x in row[7:10]),row[10],row[11],float(row[12]),*row[13:])
        return
    small = series == "section768"
    full = series == "static_grid"
    sizes = (16, 16, 10) if series == "section5120" else (8, 8, 6)
    temperatures = [t for t in range(600, 1801, 100) if t != 1000] if full else [1000]
    models = ["static_S1"] if full else ["static_S1", "HA_fig2"]
    stride = 2 if small or full else 1
    for t in temperatures:
        for model_index, model in enumerate(models):
            for center in range(2, 37, stride):
                # Single constant tilt per isotherm; the actual sampled c enters Eq. S5.
                mu = 3.09 if model == "static_S1" else 3.2
                seed = t*10000 + model_index*1000 + (100 if sizes[0]==16 else 0) + center
                if series == "dense768":
                    seed += 300
                yield (model, t, *sizes, center/100, mu, seed, 10000, 40000, "random")


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def fetch_with_resume(dest, rec, job, num_windows, indices=None, account="u3129"):
    """Retrieve immutable Slurm outputs in bounded SSH requests and verify bytes."""
    def transfer(command):
        for attempt in range(5):
            p = subprocess.run(connection(account) + [command], stdout=subprocess.PIPE)
            if p.returncode == 0:
                return p.stdout
            if p.returncode != 255 or attempt == 4:
                p.check_returncode()
            time.sleep(min(10 * (attempt + 1), 45))

    if indices is None:
        archive = f"retrieval-{job}.tar"
        ready = f"test $(ls samples/*.csv.gz | wc -l) -eq {num_windows}"
        paths = "samples/*.csv.gz logs/*.out"
    else:
        archive = f"retrieval-{job}-{indices[0]}-{indices[-1]}.tar"
        ready = ":"
        paths = " ".join(f"samples/{i}.csv.gz logs/{job}_{i}.out" for i in indices)
    info = transfer(f"set -eu; cd {dest}; {ready}; "
                  f"if test ! -f {archive}; then part=$(mktemp .retrieve-XXXXXX); "
                  f"tar cf \"$part\" {paths}; mv -- \"$part\" {archive}; fi; "
                  f"sha256sum {archive}; stat -c%s {archive}").decode().splitlines()
    sha, size = info[0].split()[0], int(info[1])
    partial = rec / f"retrieval-{job}-{sha[:12]}.part"
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        raise ValueError(f"Oversized partial MC archive {partial}")
    with partial.open("ab") as f:
        while offset < size:
            length = min(1024 * 1024, size-offset)
            chunk = transfer(f"cd {dest} && tail -c +{offset+1} {archive} | head -c {length}")
            if len(chunk) != length:
                raise RuntimeError(f"Short read of {archive} at {offset}")
            f.write(chunk)
            f.flush()
            offset += length
    raw = partial.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError("MC archive checksum mismatch")
    save(rec / "raw-transfer.json", dict(job_id=job, remote=dest+"/"+archive, size=size, sha256=sha))
    return raw, partial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["stage", "submit", "status", "fetch"])
    parser.add_argument("series", choices=["section768","section5120","static_grid","dense768",
        "atat_static","atat_ha","atat_qha","atat_section","atat_qha_section","atat_static_section","atat_ha_section",
        "atat_s3_5120","atat_s3_repeats","hybrids_1000K"])
    parser.add_argument("--account", choices=["u3129","u2626"], help="UMT login for the new hybrid calculation")
    parser.add_argument("--allocation", help="Slurm charging allocation for the selected account")
    parser.add_argument("--resume-transfer", action="store_true", help="retrieve a verified archive in restartable 1 MiB chunks")
    parser.add_argument("--slice", nargs=2, metavar=("MODEL","T"), help="fetch a completed isotherm from an array")
    args = parser.parse_args()
    hybrid = args.series == "hybrids_1000K"
    if hybrid and not args.account:
        parser.error("select --account u3129 or u2626 for the new hybrid run")
    if not hybrid and args.account == "u2626":
        parser.error("historical series are stored under u3129")
    account = args.account or "u3129"
    dest = BASE + "/" + args.series
    rec = HERE / "runs" / args.series
    if hybrid:
        rec = HYBRIDS/"runs"/account
        if args.action == "stage":
            homedir = remote('test "$(id -un)" = '+account+' && cd && pwd -P',account=account).decode().strip()
            if not re.fullmatch(r"/[A-Za-z0-9_./-]+",homedir):
                raise ValueError("Unexpected UMT home path")
            dest = homedir+"/tial-pd-hybrids1000-20261125"
        else:
            dest = json.loads((rec/"inputs.json").read_text())["directory"]
    rec.mkdir(parents=True, exist_ok=True)
    tasks = list(windows(args.series))
    sub = rec / "submission.json"
    if args.action == "stage":
        files = {"vcs.cpp": (HERE/"vcs.cpp").read_bytes(), "window.sbatch": (HERE/"window.sbatch").read_bytes(),
                 "motifs.txt": (HERE/"static/motifs.txt").read_bytes(),
                 "windows.tsv": "".join("\t".join(map(str,w))+"\n" for w in tasks).encode()}
        if args.series.startswith("atat_") or hybrid:
            for p in (HYBRIDS/"hybrids" if hybrid else HERE/"inputs"/args.series).glob("*"):
                if p.suffix in [".txt",".json"]:
                    files[p.name] = p.read_bytes()
        else:
            ha = json.loads((HERE.parent/"thermo/results/coefficients_HA1000.json").read_text("utf-8"))
            ha_row = " ".join(map(str, [17.288329847774033, *ha["effective_eV"]])) + "\n"
            files["HA_fig2.txt"] = ha_row.encode()
            files["static_S1.txt"] = (HERE/"static/coefficients.txt").read_bytes()
        files = {k:v.replace(b"\r\n",b"\n") for k,v in files.items()}
        hashes = {k:hashlib.sha256(v).hexdigest() for k,v in files.items()}
        if hybrid and (rec/"inputs.json").exists():
            raise SystemExit("Existing staged hybrid record. Verify the remote SHA-256 manifest before restaging.")
        save(rec/"inputs.json", {"resource":f"UMT {account} / CPU", "directory":dest,"sha256":hashes})
        (rec/"windows.tsv").write_bytes(files["windows.tsv"])
        files["sha256.txt"] = "".join(f"{value}  {key}\n" for key,value in hashes.items()).encode()
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for name, data in files.items():
                header = tarfile.TarInfo(name); header.size = len(data)
                tar.addfile(header, io.BytesIO(data))
        first = tasks[0]
        smoke = ["./vcs","motifs.txt",first[11] if len(first)>11 else f"{first[0]}.txt",
                 "8","8","6",str(first[1]),".15","100",str(first[6]),"1","4","2","15","random",
                 first[0],str(first[12]) if len(first)>12 else "0","smoke.csv"]
        if hybrid:
            smoke.append("orbit-sums")
        command = (f"set -eu; test \"$(id -un)\" = {account}; umask 077; mkdir -p {dest.rsplit('/',1)[0]}; mkdir {dest}; "
                   f"cd {dest}; mkdir logs samples; tar xzf -; sha256sum -c sha256.txt; "
                   "g++ -O3 -std=c++11 vcs.cpp -o vcs; "+shlex.join(smoke))
        print(remote(command, buf.getvalue(), account).decode())
    elif args.action == "submit":
        intent = rec/"intent.json"
        if intent.exists() or sub.exists():
            raise SystemExit("Existing submission intent. Reconcile in squeue/sacct and remote jobid.")
        allocation = args.allocation or ("imet-2026-2" if not hybrid else None)
        cmd = ["sbatch", "--parsable", "--partition=nompi,apollo,debug", "--qos=normal",
               "--nodes=1", "--ntasks=1", "--cpus-per-task=1", "--mem=384M",
               "--time=" + ("01:30:00" if args.series == "atat_s3_repeats" else "00:40:00"),
               "--job-name=TiPD-"+args.series, f"--array=1-{len(tasks)}%{96 if hybrid or args.series == 'atat_s3_5120' else 24}",
               "--output=logs/%A_%a.out", "window.sbatch"]
        if allocation:
            cmd.insert(2,"--account="+allocation)
        save(intent, {"utc":datetime.now(timezone.utc).isoformat(), "argv":cmd,"remote":dest})
        response = remote(f'set -eu; test "$(id -un)" = {account}; cd {dest}; test ! -f jobid; sha256sum --quiet -c sha256.txt; ' +
                          shlex.join(cmd) + " > jobid; cat jobid",account=account).decode().strip()
        job = response.split(";")[0]
        if not job.isdigit():
            raise RuntimeError("ambiguous Slurm response " + response)
        save(sub, {"job_id":job, "n_windows":len(tasks), "remote":dest})
        print(job)
    else:
        job = json.loads(sub.read_text())["job_id"]
        if args.action == "status":
            print(remote(f"cd {dest}; squeue -h -j {job} -o '%T' | sort | uniq -c; "
                         "ls samples/*.csv.gz 2>/dev/null | wc -l; "
                         f"sacct -X -n -j {job} -o State | sort | uniq -c",account=account).decode())
        else:
            selected = list(enumerate(tasks, 1))
            if args.slice:
                model, temperature = args.slice
                selected = [(i,w) for i,w in selected if w[0] == model and w[1] == int(temperature)]
                if not selected:
                    raise ValueError("Slice not found in staged array")
                rec = rec / "sections" / f"{model}_{int(temperature)}"
                rec.mkdir(parents=True, exist_ok=True)
            partial = None
            if args.resume_transfer or args.slice:
                ids = [i for i,w in selected] if args.slice else None
                raw, partial = fetch_with_resume(dest, rec, job, len(tasks), ids, account)
            else:
                raw = remote(f"set -eu; cd {dest}; test $(ls samples/*.csv.gz | wc -l) -eq {len(tasks)}; "
                             "tar cf - samples/*.csv.gz logs/*.out",account=account)
            seen = set()
            target = rec/"samples.csv.gz"
            with tarfile.open(fileobj=io.BytesIO(raw)) as tar, gzip.open(str(target)+".part","wt",newline="") as out:
                writer = None
                logs = []
                for index,w in selected:
                    name = f"samples/{index}.csv.gz"
                    log = tar.extractfile(f"logs/{job}_{index}.out").read().decode()
                    if "max_field_error_eV" not in log: raise RuntimeError(f"incomplete job {index}: {log}")
                    logs.append(f"{index}: {log.strip()}")
                    with gzip.open(tar.extractfile(name),"rt") as inp:
                        reader = csv.DictReader(inp)
                        if writer is None:
                            writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
                            writer.writeheader()
                        count = 0
                        for row in reader:
                            count += 1
                            if (row["model"] != w[0] or int(row["T_K"])!=w[1] or float(row["c0"])!=w[5]
                                    or int(row["replica"])!=w[7] or int(row["sweep"])!=w[8]+10*count
                                    or int(row["N"])!=2*w[2]*w[3]*w[4] or not 0<=int(row["n_Al"])<=int(row["N"])):
                                raise RuntimeError(f"sample metadata mismatch {index}")
                            writer.writerow(row)
                        if count!=w[9]//10: raise RuntimeError(f"sample count mismatch {index}")
                    seen.add(name)
            Path(str(target)+".part").replace(target)
            (rec/"window_logs.txt").write_text("\n".join(logs)+"\n",encoding="utf-8")
            save(rec/"retrieved.json", {"job_id":job,"windows":len(seen),"sha256_combined":hashlib.sha256(target.read_bytes()).hexdigest()})
            if partial is not None:
                partial.unlink()
            print(target)


if __name__ == "__main__":
    main()
