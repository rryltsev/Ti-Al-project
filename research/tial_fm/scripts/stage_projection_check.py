from pathlib import Path
import hashlib,json,shlex
import cluster_pilot as c
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/dpa4-n200-pilot-20261106-02';remote=json.loads((RUN/'package.json').read_text())['remote_root']
source=(ROOT/'scripts/check_phonon_projection.py').read_text();launcher=f'''#!/bin/bash
set -eo pipefail
umask 077
cd {shlex.quote(remote)}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
test ! -e harmonics/projection_diagnostics.json
/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python check_phonon_projection.py
'''
payload={'check_phonon_projection.py':source,'projection-check.sbatch':launcher};payload['projection-check-inputs.json']=json.dumps({k:hashlib.sha256(v.encode()).hexdigest() for k,v in payload.items()},indent=2)+'\n'
code="import json,pathlib,sys; root=pathlib.Path("+repr(remote)+"); files=json.load(sys.stdin); [(root/k).open('x').write(v) for k,v in files.items()]; print('STAGED_PROJECTION_CHECK')"
print(c.ssh('/home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python -c '+shlex.quote(code),json.dumps(payload).encode()).decode())
cmd=f'cd {shlex.quote(remote)} && sbatch --parsable --account=imet-2026-2 --partition=apollo --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=8G --time=00:05:00 --job-name=TiAl-H0-projection --output=projection-%j.log projection-check.sbatch'
out=c.ssh(cmd).decode().strip();assert out.split(';')[0].isdigit();(RUN/'projection-submission.json').write_text(json.dumps(dict(job_id=out.split(';')[0],command=cmd,reason='Newly observed imaginary branches require distinguishing model response from force-constant projection artifacts.'),indent=2)+'\n');print(out)
