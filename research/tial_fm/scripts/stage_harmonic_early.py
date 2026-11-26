from pathlib import Path
import hashlib,json,shlex
import cluster_pilot as c
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/dpa4-n200-pilot-20261106-02';remote=json.loads((RUN/'package.json').read_text())['remote_root']
source=(ROOT/'scripts/harmonic_early.py').read_text();launcher=(RUN/'package/baseline.sbatch').read_text().split('python baseline_worker.py')[0]+'python harmonic_early.py forces\nunset PYTHONPATH\n/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python harmonic_early.py phonons\n'
payload={'harmonic_early.py':source,'harmonic-early.sbatch':launcher};payload['harmonic-early-inputs.json']=json.dumps({k:hashlib.sha256(v.encode()).hexdigest() for k,v in payload.items()},indent=2)+'\n'
code="import json,pathlib,sys; root=pathlib.Path("+repr(remote)+"); files=json.load(sys.stdin); [(root/k).open('x').write(v) for k,v in files.items()]; print('STAGED_INDEPENDENT_H0')"
print(c.ssh('/home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python -c '+shlex.quote(code),json.dumps(payload).encode()).decode())
cmd=f'cd {shlex.quote(remote)} && sbatch --parsable --account=imet-2026-2 --partition=apollo --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=00:10:00 --job-name=TiAl-H0-independent --output=harmonic-early-%j.log harmonic-early.sbatch'
out=c.ssh(cmd).decode().strip();assert out.split(';')[0].isdigit();(RUN/'harmonic-early-submission.json').write_text(json.dumps(dict(job_id=out.split(';')[0],command=cmd,reason='H0 uses only9 reference structures and is independent of the1494 bulk B0 predictions. Evaluate concurrently to remove unnecessary serial waiting.'),indent=2)+'\n');print(out)
