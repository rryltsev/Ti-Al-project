"""Add versioned launchers using the verified Phonopy venv; preserve staged originals."""
from pathlib import Path
import hashlib,json,shlex,sys
import cluster_pilot as c
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/dpa4-n200-pilot-20261106-02'
def main():
    cfg=json.loads((RUN/'package.json').read_text());remote=cfg['remote_root'];out=RUN/'launcher-correction02';out.mkdir(exist_ok=False)
    env='/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python'
    train=(RUN/'training-plan01/train.sbatch').read_text().replace('python ../../train_observer.py','python ../../train_observer_v2.py').replace('export PYTHONPATH=/home6/u3129/tial-phonons/envs/phonopy-4.8.2','unset PYTHONPATH').replace('/home6/u3129/mlip-research/envs/kplib_111/bin/python ../../evaluate_pilot_harmonics.py',env+' ../../evaluate_pilot_harmonics.py')
    harmonic=f'''#!/bin/bash
set -eo pipefail
umask 077
cd {shlex.quote(remote)}
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
unset PYTHONPATH
{env} -c 'import json; assert json.load(open("baseline/summary.json"))["status"]=="PASS"'
if test -s harmonics/summary.json; then
  {env} -c 'import json; assert json.load(open("harmonics/summary.json"))["status"]=="PASS"'
  exit 0
fi
test ! -d harmonics
{env} harmonic_worker.py
'''
    files={'training-plan01/train-v2.sbatch':train,'training-plan01/train_observer_v2.py':(ROOT/'scripts/train_observer.py').read_text(),'harmonic-v2.sbatch':harmonic}
    for name,text in files.items():(out/Path(name).name).write_text(text,encoding='utf8',newline='\n')
    payload={name:(out/Path(name).name).read_text() for name in files}
    hashes={name:hashlib.sha256(text.encode()).hexdigest() for name,text in payload.items()}
    record=dict(reason='Verified environment is a venv with bin/python, not a flat PYTHONPATH package directory. Staged original launchers retained. Additional observer verifies input configuration hash.',sha256=hashes)
    payload['launcher-correction02.json']=json.dumps(record,indent=2)+'\n'
    code="import json,pathlib,sys; root=pathlib.Path("+repr(remote)+"); files=json.load(sys.stdin); [(root/k).open('x').write(v) for k,v in files.items()]; print('VERSIONED_LAUNCHERS_STAGED')"
    print(c.ssh('/home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python -c '+shlex.quote(code),json.dumps(payload).encode()).decode())
    (out/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    jid=json.loads((RUN/'baseline-submission.json').read_text())['job_id']
    command=f'cd {shlex.quote(remote)} && test ! -e harmonic-v2-submission.txt && sbatch --parsable --dependency=afterany:{jid} --account=imet-2026-2 --partition=apollo --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=00:15:00 --job-name=TiAl-FM-H0 --output=harmonic-v2-%j.log harmonic-v2.sbatch | tee harmonic-v2-submission.txt'
    result=c.ssh(command).decode().strip();assert result.split(';')[0].isdigit(),result
    (RUN/'harmonic-submission.json').write_text(json.dumps(dict(job_id=result.split(';')[0],command=command),indent=2)+'\n');print(result)
if __name__=='__main__':main()
