"""Stage verification, idempotent benchmarks, status and immutable collection."""
import argparse,hashlib,io,json,shlex,tarfile
from pathlib import Path
import manage_expansion as m
from prepare_grace_precision import RUN,REMOTE,ENV

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['preflight','submit','status','collect']);ap.add_argument('--device',choices=['cpu','gpu'],default='cpu');a=ap.parse_args();m.RUN=RUN
    if a.action=='preflight':
        code='''from pathlib import Path
import ast,hashlib,json,subprocess
root=Path(REMOTE)
manifest=json.loads((root/'FILES_SHA256.json').read_text())
for name,h in manifest.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==h,name
for p in root.glob('*.py'):ast.parse(p.read_text())
for p in root.glob('*.sbatch'):subprocess.run(['bash','-n',str(p)],check=True)
plan=json.loads((root/'plan.json').read_text());fp64=plan['checkpoints']['fp64']
for name,h in fp64['file_hashes'].items():assert hashlib.sha256((Path(fp64['path'])/name).read_bytes()).hexdigest()==h
assert not any(p.name=='test' for p in (root/'inputs').rglob('*'))
print(json.dumps(dict(status='PASS',package_files=len(manifest),external_fp64_files=3,no_test_data=True)))
'''.replace('REMOTE',repr(REMOTE))
        data=json.loads(m.remote_python(code));(RUN/'remote-preflight.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data));return
    if a.action=='submit':
        assert json.loads((RUN/'remote-preflight.json').read_text())['status']=='PASS'
        common=['sbatch','--parsable','--account=imet-2026-2','--nodes=1','--ntasks=1']
        cpu=m.submit(REMOTE,'CPU-benchmark',common+['--partition=apollo','--cpus-per-task=4','--mem=40G','--time=00:45:00','--job-name=TiAl-GRACE-fp-CPU','--output=CPU-benchmark-%j.log','cpu.sbatch'])
        ca=m.submit(REMOTE,'CPU-analysis',common+['--partition=apollo','--cpus-per-task=1','--mem=12G','--time=00:15:00','--dependency=afterany:'+cpu,'--job-name=TiAl-GRACE-fp-CPU-QC','--output=CPU-analysis-%j.log','analysis.sbatch','cpu'])
        gpu=m.submit(REMOTE,'GPU-benchmark',common+['--partition=v100','--gres=gpu:a100:1','--cpus-per-task=4','--mem=40G','--time=01:10:00','--dependency=afterany:20989407,afterok:'+cpu,'--job-name=TiAl-GRACE-fp-A100','--output=GPU-benchmark-%j.log','gpu.sbatch'])
        ga=m.submit(REMOTE,'GPU-analysis',common+['--partition=apollo','--cpus-per-task=1','--mem=12G','--time=00:15:00','--dependency=afterany:'+gpu,'--job-name=TiAl-GRACE-fp-A100-QC','--output=GPU-analysis-%j.log','analysis.sbatch','gpu'])
        data=dict(CPU_benchmark=cpu,CPU_analysis=ca,GPU_benchmark=gpu,GPU_analysis=ga,GPU_after_N694_array='20989407',new_training=False)
        (RUN/'submission-summary.json').write_text(json.dumps(data,indent=2)+'\n');return
    if a.action=='status':
        code='''from pathlib import Path
import json,subprocess,datetime
root=Path(REMOTE)
jobs=[json.loads(p.read_text())['job_id'] for p in (root/'submissions').glob('*.json')]
files={}
for p in list((root/'results').glob('*/*/summary.json'))+list((root/'results').glob('*/*/failure.json'))+list((root/'results').glob('*/*/progress.json'))+list((root/'comparison').glob('*/summary.json'))+[root/'weight-equivalence.json']:
    if p.exists():files[p.relative_to(root).as_posix()]=json.loads(p.read_text())
tails={p.relative_to(root).as_posix():p.read_text(errors='replace').splitlines()[-8:] for p in list(root.glob('*-benchmark-*.log'))+list(root.glob('*-analysis-*.log'))+list((root/'results').glob('*/*.log'))}
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,tails=tails,accounting=subprocess.check_output(['sacct','-X','-j',','.join(jobs),'--format=JobID,State,Elapsed,ExitCode','-P'],text=True),queue=subprocess.check_output(['squeue','-j',','.join(jobs),'-o','%.18i %.10T %.10M %.35R'],text=True))))
'''.replace('REMOTE',repr(REMOTE))
        data=json.loads(m.remote_python(code));(RUN/'latest-status.json').write_text(json.dumps(data,indent=2)+'\n')
        print(data['accounting']);print(data['queue'])
        for name,d in data['files'].items():
            print(name,d.get('status',d.get('phase')),d.get('error',''))
            if 'timings' in d:print({x['name']:round(1000*x['median_seconds'],2) for x in d['timings']})
        for name,tail in data['tails'].items():print(name,'\n'+'\n'.join(tail[-4:]))
        return
    code='''from pathlib import Path
import json,tarfile,sys,subprocess
root=Path(REMOTE);device=DEVICE
key=device.upper()+'-analysis';job=json.loads((root/'submissions'/(key+'.json')).read_text())['job_id']
state=subprocess.check_output(['sacct','-X','-j',job,'-nP','--format=State'],text=True).strip()
assert state.startswith(('COMPLETED','FAILED','CANCELLED','TIMEOUT')),state
paths=[root/'weight-equivalence.json',root/'controlled-cast.json']+list((root/'results'/device).rglob('*'))+list((root/'comparison'/device).rglob('*'))
paths += list(root.glob(device.upper()+'-*.log'))
t=tarfile.open(fileobj=sys.stdout.buffer,mode='w|')
for p in paths:
    if p.is_file():t.add(p,arcname=p.relative_to(root).as_posix())
t.close()
'''.replace('REMOTE',repr(REMOTE)).replace('DEVICE',repr(a.device))
    data=m.c.ssh(m.PY+' -c '+shlex.quote(code));folder=RUN/'collected';folder.mkdir(exist_ok=True);records={}
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        for member in tar.getmembers():
            dest=(folder/member.name).resolve();assert member.isfile() and dest.is_relative_to(folder.resolve())
            payload=tar.extractfile(member).read();dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert dest.read_bytes()==payload,member.name
            else:dest.write_bytes(payload)
            records[member.name]=hashlib.sha256(payload).hexdigest()
    (RUN/(a.device+'-collection.json')).write_text(json.dumps(dict(bytes=len(data),files=records),indent=2)+'\n')
    print('COLLECTED',a.device,len(records),len(data))

if __name__=='__main__':main()
