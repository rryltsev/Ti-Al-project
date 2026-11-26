from pathlib import Path
import hashlib,json,shlex
import cluster_pilot as c
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/dpa4-n200-pilot-20261106-02';remote=json.loads((RUN/'package.json').read_text())['remote_root']
source=(ROOT/'scripts/trainer_zero_preflight.py').read_text();launcher=(RUN/'package/baseline.sbatch').read_text().split('python baseline_worker.py')[0]+'python trainer_zero_preflight.py\n'
payload={'trainer_zero_preflight.py':source,'trainer-zero.sbatch':launcher};payload['trainer-zero-inputs.json']=json.dumps({k:hashlib.sha256(v.encode()).hexdigest() for k,v in payload.items()},indent=2)+'\n'
code="import json,pathlib,sys; root=pathlib.Path("+repr(remote)+"); files=json.load(sys.stdin); [(root/k).open('x').write(v) for k,v in files.items()]; print('STAGED_TRAINER_ZERO')"
print(c.ssh('/home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python -c '+shlex.quote(code),json.dumps(payload).encode()).decode())
cmd=f'cd {shlex.quote(remote)} && sbatch --parsable --account=imet-2026-2 --partition=apollo --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=24G --time=00:20:00 --job-name=TiAl-FM-zero --output=trainer-zero-%j.log trainer-zero.sbatch'
out=c.ssh(cmd).decode().strip();assert out.split(';')[0].isdigit();(RUN/'trainer-zero-submission.json').write_text(json.dumps(dict(job_id=out.split(';')[0],command=cmd),indent=2)+'\n');print(out)
