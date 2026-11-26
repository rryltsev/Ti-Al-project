"""Stage aligned PBE weights, submit gated replicates, inspect and collect."""
import argparse,io,json,tarfile
from pathlib import Path
import manage_expansion as m
from prepare_grace_pbe_seeds import ROOT,RUN,REMOTE,PARENT_REMOTE,load,dump,sha
def submit(key,command):
    if not (RUN/(key+'-intent.json')).exists():dump(RUN/(key+'-intent.json'),dict(remote=REMOTE,command=command))
    m.RUN=RUN;return m.submit(REMOTE,key,command)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['preflight','smoke','launch','status','collect']);ap.add_argument('--collection-name',default='collected-final');a=ap.parse_args()
    common=['sbatch','--parsable','--account=imet-2026-2','--nodes=1','--ntasks=1','--cpus-per-task=4']
    if a.action=='preflight':
        code='''from pathlib import Path
import ast,hashlib,json,shutil,subprocess,sys
p=Path(REMOTE);parent=Path(PARENT);sys.path.insert(0,str(p))
from grace_storage_guard import quota_snapshot
def sha(q):return hashlib.sha256(q.read_bytes()).hexdigest()
for n,h in json.loads((p/'FILES_SHA256.json').read_text()).items():assert sha(p/n)==h,n
files={}
for n,h in json.loads((p/'aligned/PBE/alignment.json').read_text())['checkpoint_hashes'].items():
 source=parent/'aligned/PBE'/n;dest=p/'aligned/PBE'/n;assert sha(source)==h
 if not dest.exists():shutil.copyfile(source,dest)
 assert sha(dest)==h;files[str(dest.relative_to(p))]=h
(p/'CHECKPOINT_FILES_SHA256.json').write_text(json.dumps(files,indent=2))
for part in ['train','validation']:assert sha(p/'data/PBE'/(part+'.extxyz'))==sha(parent/'data/PBE'/(part+'.extxyz'))
assert sorted(f.name for f in (p/'data').iterdir())==['PBE']
for f in p.rglob('*.py'):ast.parse(f.read_text())
for f in p.glob('*.sbatch'):subprocess.run(['bash','-n',str(f)],check=True)
print(json.dumps(dict(status='PASS',checkpoint_files=files,quota=quota_snapshot(),cases=8,PBE_only=True)))
'''.replace('REMOTE',repr(REMOTE)).replace('PARENT',repr(PARENT_REMOTE))
        d=json.loads(m.remote_python(code));dump(RUN/'remote-preflight.json',d);print(d)
    elif a.action=='smoke':
        assert load(RUN/'remote-preflight.json')['status']=='PASS'
        j=submit('smoke',common+['--partition=v100','--gres=gpu:a100:1','--mem=24G','--time=00:10:00','--array=0-7%1','--job-name=TiAl-GRA-PBE-smoke','--output=smoke-%A_%a.log','smoke.sbatch'])
        submit('optimizer-gate',common+['--partition=apollo','--mem=4G','--time=00:05:00','--dependency=afterany:'+j,'--job-name=TiAl-GRA-PBE-gate','--output=gate-%j.log','gate.sbatch'])
    elif a.action=='launch':
        code='''from pathlib import Path
import json,sys
p=Path(REMOTE);sys.path.insert(0,str(p))
from grace_storage_guard import quota_snapshot
gate=json.loads((p/'optimizer-gate.json').read_text());assert gate['status']=='PASS'
q=quota_snapshot(int(5.7*2**30))
print(json.dumps(dict(gate=gate,quota=q)))
'''.replace('REMOTE',repr(REMOTE))
        d=json.loads(m.remote_python(code));dump(RUN/'launch-gates.json',d)
        j=submit('train',common+['--partition=v100','--gres=gpu:a100:1','--mem=24G','--time=00:45:00','--array=0-7%1','--job-name=TiAl-GRA-PBE-seeds','--output=train-%A_%a.log','train.sbatch'])
        k=submit('postprocess',common+['--partition=apollo','--mem=16G','--time=00:30:00','--dependency=afterany:'+j,'--job-name=TiAl-GRA-PBE-compare','--output=post-%j.log','post.sbatch'])
        dump(RUN/'submission-summary.json',dict(train=j,postprocess=k,cases=8,seeds=[20,30],XC='PBE',GPU_concurrency=1,postprocess_dependency='afterany; missing/failed models explicitly reported'))
    elif a.action=='status':
        code='''from pathlib import Path
import json,subprocess
p=Path(REMOTE);jobs=[json.loads(f.read_text())['job_id'] for f in (p/'submissions').glob('*.json')]
d=dict(accounting=subprocess.run(['sacct','-j',','.join(jobs),'--format=JobID,State,Elapsed,ExitCode','-P'],capture_output=True,text=True).stdout,queue=subprocess.run(['squeue','-j',','.join(jobs),'-o','%i %T %R'],capture_output=True,text=True).stdout,files={},logs={})
files=[p/'optimizer-gate.json',p/'comparison/summary.json',p/'postprocessing-status.json',p/'baseline/summary.json',p/'baseline/harmonics/summary.json',p/'aligned/PBE/alignment.json']
for c in json.loads((p/'plan.json').read_text())['cases']:
 for base in ['smoke','cases']:
  w=p/base/c['name'];files += [w/f for f in ['optimizer-check.json','training-summary.json','evaluation-summary.json','progress.json']]
for f in files:
 if f.exists():d['files'][str(f.relative_to(p))]=json.loads(f.read_text())
for f in list(p.glob('*.log'))+list(p.glob('smoke/*/train.log'))+list(p.glob('cases/*/train.log'))+list(p.glob('cases/*/evaluate.log')):
 text=f.read_text(errors='replace')[-2500:]
 if 'Traceback' in text or 'Error' in text:d['logs'][str(f.relative_to(p))]=text
print(json.dumps(d))
'''.replace('REMOTE',repr(REMOTE))
        d=json.loads(m.remote_python(code));dump(RUN/'latest-status.json',d)
        print('\n'.join(x for x in d['accounting'].splitlines() if '.batch|' not in x and '.extern|' not in x));print(d['queue'])
        print({k:(v.get('status'),v.get('steps')) for k,v in d['files'].items()});print(json.dumps(d['logs'],indent=2))
    else:
        assert a.collection_name.replace('-','').isalnum()
        code='''from pathlib import Path
import sys,json,tarfile,io,hashlib
p=Path(REMOTE);files=[]
for folder in ['smoke','cases','submissions','comparison','baseline','aligned','ti3al/forces','ti3al/analysis','ti3al/analysis-asr']:
 files += [f for f in (p/folder).rglob('*') if f.is_file() and f.suffix in ['.json','.yaml','.log','.npy','.npz','.md'] and 'final_model' not in f.parts]
files += list(p.glob('*.log'))+[p/'optimizer-gate.json']
files=[f for f in files if f.exists()];h={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for f in files:t.add(f,arcname=str(f.relative_to(p)))
 b=json.dumps(h).encode();item=tarfile.TarInfo('COLLECTION_SHA256.json');item.size=len(b);t.addfile(item,io.BytesIO(b))
'''.replace('REMOTE',repr(REMOTE))
        blob=m.remote_python(code);out=RUN/a.collection_name;out.mkdir(exist_ok=False)
        with tarfile.open(fileobj=io.BytesIO(blob)) as t:
            for f in t:
                p=(out/f.name).resolve();assert p.is_relative_to(out.resolve()) and f.isfile();p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(t.extractfile(f).read())
        h=load(out/'COLLECTION_SHA256.json')
        for n,v in h.items():assert sha(out/n)==v,n
        print('COLLECTED',len(h),len(blob))
if __name__=='__main__':main()
