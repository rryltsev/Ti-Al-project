"""Attach inexpensive metric evaluation to each successful GPU pilot task."""
from pathlib import Path
import hashlib,json,shlex
import cluster_pilot as c
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/dpa4-n200-pilot-20261106-02';remote=json.loads((RUN/'package.json').read_text())['remote_root']
source=(ROOT/'scripts/analyze_predictions.py').read_text();launcher=f'''#!/bin/bash
set -eo pipefail
umask 077
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
RUN={shlex.quote(remote)}
CASES=(PBE-lora16 PBE-full r2SCAN-lora16 r2SCAN-full)
CASE_NAME=${{CASES[$SLURM_ARRAY_TASK_ID]}}
cd "$RUN/training-plan01/cases/$CASE_NAME"
/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python "$RUN/analyze_predictions.py" --package "$RUN" --predictions evaluation/predictions --output evaluation/metrics
'''
payload={'analyze_predictions.py':source,'pilot-analysis.sbatch':launcher};payload['pilot-analysis-inputs.json']=json.dumps({k:hashlib.sha256(v.encode()).hexdigest() for k,v in payload.items()},indent=2)+'\n'
code="import json,pathlib,sys; root=pathlib.Path("+repr(remote)+"); files=json.load(sys.stdin); [(root/k).open('x').write(v) for k,v in files.items()]; print('STAGED_PILOT_ANALYSIS')"
print(c.ssh('/home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python -c '+shlex.quote(code),json.dumps(payload).encode()).decode())
job=json.loads((RUN/'training-submission.json').read_text())['job_id']
cmd=f'cd {shlex.quote(remote)} && sbatch --parsable --dependency=aftercorr:{job} --account=imet-2026-2 --partition=apollo --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=4G --time=00:05:00 --array=0-3 --job-name=TiAl-pilot-metrics --output=pilot-metrics-%A_%a.log pilot-analysis.sbatch'
out=c.ssh(cmd).decode().strip();assert out.split(';')[0].isdigit();(RUN/'pilot-analysis-submission.json').write_text(json.dumps(dict(job_id=out.split(';')[0],command=cmd,scope='After each corresponding successful GPU task, calculate train/development-only metrics and conservative-group confidence intervals; test remains excluded.'),indent=2)+'\n');print(out)
