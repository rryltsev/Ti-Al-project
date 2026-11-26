import argparse,hashlib,json,shlex,io,tarfile
from pathlib import Path
import manage_expansion as m
from prepare_grace_pilot import RUN,REMOTE,ENV
def main():
    global RUN,REMOTE
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['baseline','status','smoke','train','collect']);ap.add_argument('--deferred',action='store_true');ap.add_argument('--collection-name',default='collected');ap.add_argument('--run');ap.add_argument('--brief',action='store_true');a=ap.parse_args()
    if a.run:
        assert Path(a.run).name==a.run
        RUN=RUN.parent/a.run;REMOTE=json.loads((RUN/'package.json').read_text())['remote_root']
    m.RUN=RUN
    common=['sbatch','--parsable','--account=imet-2026-2','--nodes=1','--ntasks=1']
    gpu=['--partition=v100','--gres=gpu:a100:1','--cpus-per-task=4','--mem=40G']
    cpu=['--partition=apollo','--cpus-per-task=4','--mem=16G']
    if a.action=='baseline':
        j=m.submit(REMOTE,'baseline',common+gpu+['--time=00:30:00','--job-name=TiAl-GRACE-B0','--output=baseline-%j.log','baseline.sbatch'])
        h=m.submit(REMOTE,'H0',common+cpu+['--time=00:15:00','--dependency=afterok:'+j,'--job-name=TiAl-GRACE-H0','--output=H0-%j.log','harmonics.sbatch']);print('BASELINE',j,'H0',h);return
    if a.action in ['smoke','train']:
        code="import json;from pathlib import Path;p=Path("+repr(REMOTE)+");assert json.loads((p/'baseline/summary.json').read_text())['status']=='PASS';assert json.loads((p/'baseline/harmonics/summary.json').read_text())['status']=='PASS'"
        if a.action=='train':code+=";assert all(json.loads((p/'smoke'/c['name']/'optimizer-check.json').read_text())['status']=='PASS' for c in json.loads((p/'plan.json').read_text())['cases'])"
        dependency=[]
        if a.deferred:
            parent='H0' if a.action=='smoke' else 'optimizer-gate'
            job=json.loads((RUN/'submissions'/(parent+'.json')).read_text())['job_id'];dependency=['--dependency=afterok:'+job]
        else:m.remote_python(code)
        j=m.submit(REMOTE,a.action,common+gpu+dependency+['--time='+('01:00:00' if a.action=='smoke' else '04:00:00'),'--array=0-3%1','--job-name=TiAl-GRACE-'+a.action,'--output='+a.action+'-%A_%a.log',a.action+'.sbatch'])
        if a.action=='smoke':m.submit(REMOTE,'optimizer-gate',common+cpu+['--time=00:10:00','--dependency=afterok:'+j,'--job-name=TiAl-GRACE-opt-gate','--output=optimizer-gate-%j.log','optimizer-gate.sbatch'])
        if a.action=='train':m.submit(REMOTE,'trained-harmonics',common+cpu+['--time=00:15:00','--array=0-3%2','--dependency=afterok:'+j,'--job-name=TiAl-GRACE-pilot-H','--output=trained-H-%A_%a.log','trained-harmonics.sbatch'])
        return
    if a.action=='status':
        code='''from pathlib import Path
import json,subprocess,datetime
p=Path(REMOTE);jobs=[json.loads(f.read_text())['job_id'] for f in (p/'submissions').glob('*.json')]
files={}
for pattern in ['baseline/summary.json','baseline/numerical-QC.json','baseline/harmonics/summary.json','optimizer-gate.json','storage-probe.json','comparison/summary.json','aligned/*/alignment.json','smoke/*/*.json','cases/*/optimizer-check.json','cases/*/progress.json','cases/*/training-summary.json','cases/*/evaluation-summary.json','cases/*/harmonics/summary.json']:
 for f in p.glob(pattern):files[f.relative_to(p).as_posix()]=json.loads(f.read_text())
logs={f.relative_to(p).as_posix():f.read_text(errors='replace').splitlines()[-12:] for f in list(p.glob('*.log'))+list(p.glob('smoke/*/train.log'))+list(p.glob('cases/*/train.log'))}
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,logs=logs,accounting=subprocess.check_output(['sacct','-X','-j',','.join(jobs),'--format=JobID,State,Elapsed,ExitCode','-P'],text=True),queue=subprocess.check_output(['squeue','-j',','.join(jobs),'-o','%.20i %.10T %.30R'],text=True))))
'''.replace('REMOTE',repr(REMOTE))
        d=json.loads(m.remote_python(code));(RUN/'latest-status.json').write_text(json.dumps(d,indent=2)+'\n');print(d['accounting']);print(d['queue'])
        if a.brief:
            for k,v in d['files'].items():
                if any(s in k for s in ['optimizer-gate.json','storage-probe.json','optimizer-check.json','progress.json','training-summary.json','evaluation-summary.json','harmonics/summary.json','comparison/summary.json']):print(k,v.get('status'),v.get('steps',''))
            for k,v in d['logs'].items():
                if any('Error' in s or 'Traceback' in s for s in v):print(k,'\n'+'\n'.join(v))
            return
        for k,v in d['files'].items():print(k,v.get('status'),v.get('steps',''))
        for k,v in d['logs'].items():print(k,'\n'+'\n'.join(v[-5:]))
        return
    code='''from pathlib import Path
import hashlib,io,tarfile,sys
p=Path(REMOTE);t=tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz');hashes={}
def add_file(f):
 name=f.relative_to(p).as_posix();data=f.read_bytes();hashes[name]=hashlib.sha256(data).hexdigest()
 item=tarfile.TarInfo(name);item.size=len(data);t.addfile(item,io.BytesIO(data))
for directory in ['baseline','aligned','smoke','submissions','recovery','cases','comparison']:
 for f in (p/directory).rglob('*'):
  if f.is_file() and f.suffix in ['.json','.log','.yaml','.npz','.npy','.py','.md'] and 'original-export' not in f.parts:add_file(f)
for name in ['optimizer-gate.json','storage-probe.json','STORAGE_POLICY.json','CHECKPOINT_FILES_SHA256.json']:
 if (p/name).exists():add_file(p/name)
data=__import__('json').dumps(hashes,indent=2).encode();item=tarfile.TarInfo('COLLECTION_SHA256.json');item.size=len(data);t.addfile(item,io.BytesIO(data))
t.close()
'''.replace('REMOTE',repr(REMOTE))
    assert Path(a.collection_name).name==a.collection_name and a.collection_name not in ['.','..']
    data=m.c.ssh(m.PY+' -c '+shlex.quote(code));out=RUN/a.collection_name;out.mkdir(exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as t:
        for f in t.getmembers():
            p=(out/f.name).resolve();assert p.is_relative_to(out.resolve()) and f.isfile();b=t.extractfile(f).read();p.parent.mkdir(parents=True,exist_ok=True)
            if p.exists():assert p.read_bytes()==b,f.name
            else:p.write_bytes(b)
    hashes=json.loads((out/'COLLECTION_SHA256.json').read_text())
    for name,h in hashes.items():assert hashlib.sha256((out/name).read_bytes()).hexdigest()==h,name
    print('COLLECTED',len(data),'bytes;',len(hashes),'files verified')
if __name__=='__main__':main()
