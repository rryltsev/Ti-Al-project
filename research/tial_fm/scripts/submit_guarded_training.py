"""Submit four serialized GPU pilots after successful completion of the B0 job chain."""
from pathlib import Path
import hashlib,json,shlex,datetime
import cluster_pilot as c
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/dpa4-n200-pilot-20261106-02';remote=json.loads((RUN/'package.json').read_text())['remote_root']
assert not (RUN/'training-submission.json').exists()
source=(ROOT/'scripts/train_observer.py').read_text();launcher=(RUN/'launcher-correction02/train-v2.sbatch').read_text().replace('train_observer_v2.py','train_observer_v3.py')
payload={'training-plan01/train_observer_v3.py':source,'training-plan01/train-v3.sbatch':launcher}
manifest=dict(reason='Require exact audited backend hashes, full runtime requires_grad mask equality, and four passed real-data zero-step initializations before any optimizer step.',files={k:hashlib.sha256(v.encode()).hexdigest() for k,v in payload.items()})
payload['training-plan01/guards-v3.json']=json.dumps(manifest,indent=2)+'\n'
code='''import json,pathlib,sys
root=pathlib.Path(REMOTE)
for path in ['harmonics/summary.json','runtime-mask-preflight/summary.json','trainer-zero-preflight/summary.json','baseline/export_checks.json']:
    assert json.loads((root/path).read_text())['status']=='PASS',path
assert all(x['pass_checks'] for x in json.loads((root/'baseline/numerical_checks.json').read_text()))
files=json.load(sys.stdin)
for k,v in files.items():(root/k).open('x').write(v)
print('GUARDS_VERIFIED_AND_STAGED')
'''.replace('REMOTE',repr(remote))
print(c.ssh('/home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python -c '+shlex.quote(code),json.dumps(payload).encode()).decode())
outdir=RUN/'training-guards-v3';outdir.mkdir(exist_ok=False)
for k,v in payload.items():(outdir/Path(k).name).write_text(v,encoding='utf8',newline='\n')
dependency=json.loads((RUN/'baseline-continuation-submission.json').read_text())['job_id']
command=f'cd {shlex.quote(remote)}/training-plan01 && test ! -e submission.txt && sbatch --parsable --dependency=afterok:{dependency} --account=imet-2026-2 --partition=v100 --gres=gpu:1 --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=40G --time=04:00:00 --array=0-3%1 --job-name=TiAl-N200-pilot --output=pilot-%A_%a.log train-v3.sbatch | tee submission.txt'
out=c.ssh(command).decode().strip();assert out.split(';')[0].isdigit(),out
record=dict(job_id=out.split(';')[0],remote_root=remote,dependency_afterok=dependency,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command,gate_policy='Numerical B0, H0, backend/runtime masks and all four zero-step initializations passed at submission. Array cannot start until bulk B0 continuation succeeds; actual observer rechecks B0 and H0 summary PASS before constructing trainer.',cases=['PBE-lora16','PBE-full','r2SCAN-lora16','r2SCAN-full'],max_simultaneous_new_GPUs=1,optimizer_steps_executed_at_submission=0)
(RUN/'training-submission.json').write_text(json.dumps(record,indent=2)+'\n');print(out)
