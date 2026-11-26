"""Idempotent Ti3Al job submission, status and hash-verified collection."""
from pathlib import Path
import argparse,io,json,tarfile
import manage_expansion as m
from prepare_ti3al_pbe import RUN,REMOTE,ROOT,dump,sha
def submit(key,args):
    intent=RUN/(key+'-intent.json')
    if not intent.exists():dump(intent,dict(remote=REMOTE,command=args))
    m.RUN=RUN
    return m.submit(REMOTE,key,args)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','launch','status','collect','collect-asr']);ap.add_argument('--prepare-key',default='prepare');a=ap.parse_args()
    common=['sbatch','--parsable','--account=imet-2026-2','--nodes=1','--ntasks=1','--cpus-per-task=4']
    if a.action=='prepare':
        code='''from pathlib import Path
import hashlib,json,subprocess
p=Path(REMOTE)
def sha(q):return hashlib.sha256(q.read_bytes()).hexdigest()
for n,h in json.loads((p/'FILES_SHA256.json').read_text()).items():assert sha(p/n)==h,n
for c in json.loads((p/'models.json').read_text()):
 if c['backend']=='DPA4':assert sha(Path(c['model']))==c['sha256'],c['name']
 else:
  w=Path(c['model'])
  for n,h in c['hashes'].items():assert sha(w/'final-checkpoint'/n)==h,c['name']
  assert sha(w/'seed/10/model.yaml')==c['yaml_sha256']
for f in p.glob('*.sbatch'):subprocess.run(['bash','-n',str(f)],check=True)
print(json.dumps({'status':'PASS','models':16}))
'''.replace('REMOTE',repr(REMOTE))
        check=json.loads(m.remote_python(code));dump(RUN/'remote-preflight.json',check)
        submit(a.prepare_key,common+['--partition=apollo','--mem=8G','--time=00:05:00','--job-name=Ti3Al-PBE-prepare','--output=prepare-%j.log','cpu.sbatch','prepare'])
    elif a.action=='launch':
        check=json.loads(m.remote_python('import json;from pathlib import Path;p=Path('+repr(REMOTE)+');s=json.loads((p/"prepared/summary.json").read_text());assert s["status"]=="PASS";print(json.dumps(s))'))
        dump(RUN/'prepared-summary.json',check)
        j=submit('forces',common+['--partition=v100','--gres=gpu:a100:1','--mem=24G','--time=00:15:00','--array=0-15%1','--job-name=Ti3Al-PBE-forces','--output=forces-%A_%a.log','forces.sbatch'])
        k=submit('analysis',common+['--partition=apollo','--mem=16G','--time=00:30:00','--dependency=afterany:'+j,'--job-name=Ti3Al-PBE-compare','--output=analysis-%j.log','cpu.sbatch','analyze'])
        state=json.loads((ROOT/'state.json').read_text(encoding='utf8'));state['ti3al_phonons'].update(status='submitted',forces_job=j,analysis_job=k);dump(ROOT/'state.json',state)
    elif a.action=='status':
        code='''from pathlib import Path
import json,subprocess
p=Path(REMOTE);jobs=[json.loads(f.read_text())['job_id'] for f in (p/'submissions').glob('*.json')]
d=dict(accounting=subprocess.run(['sacct','-j',','.join(jobs),'--format=JobID,State,Elapsed,ExitCode','-P'],capture_output=True,text=True).stdout,queue=subprocess.run(['squeue','-j',','.join(jobs),'-o','%i %T %R'],capture_output=True,text=True).stdout,summaries={},logs={})
for f in [p/'prepared/summary.json',p/'analysis/summary.json',p/'analysis-asr/summary.json']:
 if f.exists():d['summaries'][str(f.relative_to(p))]=json.loads(f.read_text())
d['analyzed_cases']={f.parent.name:json.loads(f.read_text())['metrics'] for f in (p/'analysis').glob('*/summary.json')}
for f in p.glob('*.log'):d['logs'][f.name]=f.read_text(errors='replace')[-1800:]
print(json.dumps(d))
'''.replace('REMOTE',repr(REMOTE))
        d=json.loads(m.remote_python(code));dump(RUN/'latest-status.json',d)
        print('\n'.join(line for line in d['accounting'].splitlines() if '.batch|' not in line and '.extern|' not in line));print(d['queue']);print({k:v.get('status') for k,v in d['summaries'].items()});print('Analyzed cases:',len(d['analyzed_cases']))
        for k,v in d['logs'].items():
            if k=='prepare-20991461.log' and d['summaries'].get('prepared/summary.json',{}).get('status')=='PASS':continue
            if 'Traceback' in v or 'Error' in v:print(k,v)
    else:
        code='''from pathlib import Path
import io,tarfile,json,sys,hashlib
p=Path(REMOTE);files=[f for folder in FOLDERS for f in (p/folder).rglob('*') if f.is_file()]+list(p.glob('*.log'))
h={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
t=tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz')
for f in files:t.add(f,arcname=str(f.relative_to(p)))
b=json.dumps(h,indent=2).encode();i=tarfile.TarInfo('COLLECTION_SHA256.json');i.size=len(b);t.addfile(i,io.BytesIO(b));t.close()
'''.replace('REMOTE',repr(REMOTE)).replace('FOLDERS',repr(['analysis-asr','submissions'] if a.action=='collect-asr' else ['prepared','forces','analysis','submissions']))
        blob=m.remote_python(code);out=RUN/('collected-asr' if a.action=='collect-asr' else 'collected');out.mkdir(exist_ok=False)
        with tarfile.open(fileobj=io.BytesIO(blob)) as t:
            for f in t.getmembers():
                p=(out/f.name).resolve();assert p.is_relative_to(out.resolve()) and f.isfile()
                p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(t.extractfile(f).read())
        hashes=json.loads((out/'COLLECTION_SHA256.json').read_text())
        for n,h in hashes.items():assert sha(out/n)==h,n
        print('COLLECTED',len(hashes),'files',len(blob),'bytes')
if __name__=='__main__':main()
