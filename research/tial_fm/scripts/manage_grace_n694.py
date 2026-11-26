"""Idempotent Slurm stages for PBE-only six-case N694 experiment."""
import argparse,json,sys
import manage_expansion as m
import manage_grace_pbe_seeds as previous
from prepare_grace_n694 import ROOT,RUN,REMOTE,PREVIOUS_REMOTE,load,dump

def submit(key,args):
    if not (RUN/(key+'-intent.json')).exists():dump(RUN/(key+'-intent.json'),dict(remote=REMOTE,command=args))
    m.RUN=RUN;return m.submit(REMOTE,key,args)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['preflight','prepare','retry-smoke','queue-production','launch','status','collect']);ap.add_argument('--collection-name',default='collected-final');a=ap.parse_args()
    common=['sbatch','--parsable','--account=imet-2026-2','--nodes=1','--ntasks=1','--cpus-per-task=4']
    if a.action=='preflight':
        code='''from pathlib import Path
import ast,hashlib,json,subprocess,sys
p=Path(REMOTE);sys.path.insert(0,str(p))
from grace_storage_guard import quota_snapshot
def sha(q):return hashlib.sha256(q.read_bytes()).hexdigest()
manifest=json.loads((p/'FILES_SHA256.json').read_text())
for n,h in manifest.items():assert sha(p/n)==h,n
plan=json.loads((p/'plan.json').read_text());assert len(plan['cases'])==6
for n,h in plan['source_hashes'].items():assert sha(Path(plan['source_checkpoint'])/n)==h
assert sorted(f.name for f in (p/'data').iterdir())==['PBE']
assert sha(p/'data/PBE/validation.extxyz')==sha(Path(PREVIOUS)/'data/PBE/validation.extxyz')
for parent in plan['parents']:
 w=Path(parent['root'])/'cases'/parent['case']['name'];s=json.loads((w/'training-summary.json').read_text());assert s['status']=='PASS'
 for n,h in s['hashes'].items():assert sha(w/'final-checkpoint'/n)==h
 assert (Path(parent['ti3al_root'])/'analysis'/parent['ti3al_name']/'summary.json').exists()
for f in p.rglob('*.py'):ast.parse(f.read_text())
for f in p.glob('*.sbatch'):subprocess.run(['bash','-n',str(f)],check=True)
assert subprocess.check_output(['id','-un'],text=True).strip()=='u3129'
print(json.dumps(dict(status='PASS',files=len(manifest),quota=quota_snapshot(),cases=6,PBE_only=True,parent_checkpoints=6)))
'''.replace('REMOTE',repr(REMOTE)).replace('PREVIOUS',repr(PREVIOUS_REMOTE))
        d=json.loads(m.remote_python(code));dump(RUN/'remote-preflight.json',d);print(d)
    elif a.action=='prepare':
        assert load(RUN/'remote-preflight.json')['status']=='PASS'
        b=submit('baseline',common+['--partition=v100','--gres=gpu:a100:1','--mem=24G','--time=00:20:00','--job-name=TiAl-GRA694-baseline','--output=baseline-%j.log','baseline.sbatch'])
        h=submit('baseline-harmonics',common+['--partition=apollo','--mem=8G','--time=00:10:00','--dependency=afterany:'+b,'--job-name=TiAl-GRA694-H0','--output=H0-%j.log','baseline-harmonics.sbatch'])
        s=submit('smoke',common+['--partition=v100','--gres=gpu:a100:1','--mem=24G','--time=00:10:00','--array=0-5%1','--dependency=afterany:'+h,'--job-name=TiAl-GRA694-smoke','--output=smoke-%A_%a.log','smoke.sbatch'])
        submit('optimizer-gate',common+['--partition=apollo','--mem=4G','--time=00:05:00','--dependency=afterany:'+s,'--job-name=TiAl-GRA694-gate','--output=gate-%j.log','gate.sbatch'])
    elif a.action=='retry-smoke':
        code='''from pathlib import Path
import json,sys,subprocess
p=Path(REMOTE).resolve();sys.path.insert(0,str(p))
from grace_storage_guard import quota_snapshot
quota_snapshot(int(1.6*2**30))
assert json.loads((p/'baseline/summary.json').read_text())['status']=='PASS'
assert json.loads((p/'baseline/harmonics/summary.json').read_text())['status']=='PASS'
dest=p/'recovery/smoke-quota-20991836'
if not dest.exists():
 source=p/'smoke';assert source.resolve().is_relative_to(p)
 assert not list(source.rglob('attempt.json')) and not list(source.rglob('optimizer-check.json'))
 assert '20991836' not in subprocess.check_output(['squeue','-h','-u','u3129','-o','%i'],text=True)
 dest.parent.mkdir(exist_ok=True);source.rename(dest)
print('RETRY_READY')
'''.replace('REMOTE',repr(REMOTE))
        print(m.remote_python(code).decode())
        s=submit('smoke-retry1',common+['--partition=v100','--gres=gpu:a100:1','--mem=24G','--time=00:10:00','--array=0-5%1','--job-name=TiAl-GRA694-smoke','--output=smoke-%A_%a.log','smoke.sbatch'])
        submit('optimizer-gate-retry1',common+['--partition=apollo','--mem=4G','--time=00:05:00','--dependency=afterany:'+s,'--job-name=TiAl-GRA694-gate','--output=gate-%j.log','gate.sbatch'])
    elif a.action in ['launch','queue-production']:
        code='''from pathlib import Path
import json,sys
p=Path(REMOTE);sys.path.insert(0,str(p))
from grace_storage_guard import quota_snapshot
pending=ALLOW_PENDING
gate=json.loads((p/'optimizer-gate.json').read_text()) if (p/'optimizer-gate.json').exists() else dict(status='PENDING')
assert gate['status']=='PASS' or (pending and gate['status']=='PENDING')
for name in ['baseline/summary.json','baseline/harmonics/summary.json','aligned/PBE/alignment.json']:assert json.loads((p/name).read_text())['status']=='PASS'
cases=json.loads((p/'plan.json').read_text())['cases']
checked=[c for c in cases if (p/'smoke'/c['name']/'optimizer-check.json').exists()]
assert len(checked)==6 if not pending else all(any(c['method']==m for c in checked) for m in ['full','restricted_radial'])
for c in checked:assert json.loads((p/'smoke'/c['name']/'optimizer-check.json').read_text())['status']=='PASS'
inventories=[json.loads((p/'smoke'/c['name']/'parameter-inventory.json').read_text()) for c in checked]
signatures=[sorted((v['name'],v['shape'],v['initial_sha256']) for v in x['parameters']) for x in inventories];assert all(s==signatures[0] for s in signatures)
print(json.dumps(dict(gate=gate,quota=quota_snapshot(int(4.1*2**30)),initial_weights_identical_for_completed_smoke=True,completed_smoke=[c['name'] for c in checked])))
'''.replace('REMOTE',repr(REMOTE)).replace('ALLOW_PENDING',repr(a.action=='queue-production'))
        d=json.loads(m.remote_python(code));dump(RUN/'launch-gates.json',d)
        dependency=[]
        if a.action=='queue-production':dependency=['--dependency=afterany:'+load(RUN/'submissions/optimizer-gate-retry1.json')['job_id']]
        j=submit('train',common+['--partition=v100','--gres=gpu:a100:1','--mem=32G','--time=01:30:00','--array=0-5%1','--job-name=TiAl-GRA694-PBE','--output=train-%A_%a.log']+dependency+['train.sbatch'])
        k=submit('postprocess',common+['--partition=apollo','--mem=16G','--time=00:30:00','--dependency=afterany:'+j,'--job-name=TiAl-GRA694-compare','--output=post-%j.log','post.sbatch'])
        dump(RUN/'submission-summary.json',dict(train=j,postprocess=k,cases=6,seeds=[10,20,30],XC='PBE',N=694,epochs=50,steps=34700,GPU_concurrency=1,optimizer_gate_at_submission=d['gate']['status'],production_gate='All six optimizer smoke PASS required by train.sbatch before training',postprocess_dependency='afterany; explicit failure status'))
    else:
        previous.RUN=RUN;previous.REMOTE=REMOTE
        sys.argv=[sys.argv[0],a.action,'--collection-name',a.collection_name];previous.main()
if __name__=='__main__':main()
