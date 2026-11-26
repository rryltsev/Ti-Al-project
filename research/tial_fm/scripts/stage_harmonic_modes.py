from pathlib import Path
import hashlib,json,shlex
import cluster_pilot as c
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/dpa4-n200-pilot-20261106-02'
remote=json.loads((RUN/'package.json').read_text())['remote_root']
source=(ROOT/'scripts/harmonic_mode_diagnostics.py').read_text()
launcher=f'''#!/bin/bash
set -eo pipefail
umask 077
cd {shlex.quote(remote)}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
test ! -e harmonics/mode_diagnostics.json
/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python harmonic_mode_diagnostics.py harmonics
'''
payload={'harmonic_mode_diagnostics.py':source,'harmonic-modes.sbatch':launcher}
payload['harmonic-modes-inputs.json']=json.dumps({k:hashlib.sha256(v.encode()).hexdigest() for k,v in payload.items()},indent=2)+'\n'
code="import json,pathlib,sys; root=pathlib.Path("+repr(remote)+"); files=json.load(sys.stdin); [(root/k).open('x').write(v) for k,v in files.items()]; print('STAGED_MODE_DIAGNOSTICS')"
print(c.ssh('/home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python -c '+shlex.quote(code),json.dumps(payload).encode()).decode())
jid=json.loads((RUN/'harmonic-submission.json').read_text())['job_id']
cmd=f'cd {shlex.quote(remote)} && sbatch --parsable --dependency=afterok:{jid} --account=imet-2026-2 --partition=apollo --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G --time=00:10:00 --job-name=TiAl-H0-modes --output=harmonic-modes-%j.log harmonic-modes.sbatch'
out=c.ssh(cmd).decode().strip();assert out.split(';')[0].isdigit();(RUN/'harmonic-modes-submission.json').write_text(json.dumps(dict(job_id=out.split(';')[0],command=cmd),indent=2)+'\n');print(out)
