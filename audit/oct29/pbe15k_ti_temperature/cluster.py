"""Manage one inherited Slurm job; deliberately provides no sbatch command."""
from pathlib import Path
import sys,importlib.util,json,hashlib,shlex,subprocess,time,tarfile,io
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[2]
spec=importlib.util.spec_from_file_location('ssh_helpers',WORK.parent/'MLIP-research/scripts/cluster_ssh.py')
ssh=importlib.util.module_from_spec(spec);spec.loader.exec_module(ssh)
CFG=json.loads((ROOT/'protocol.json').read_text());REMOTE=CFG['remote']
OLD=json.loads((ROOT/'submission.json').read_text())['previous_remote']
PYTHON='/home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python'
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(name,obj):
    (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def run(command,content=None,timeout=180,binary=False):
    r=subprocess.run(ssh.ssh_args()+['-o','BatchMode=yes','mlip-umt',command],input=content,capture_output=True,timeout=timeout)
    if r.returncode:raise RuntimeError(r.stdout.decode(errors='replace')+r.stderr.decode(errors='replace'))
    return r.stdout if binary else r.stdout.decode('utf-8',errors='replace')
def py(code,**kw):return run('python3 -c '+shlex.quote(code),**kw)
def jobid():
    jid=str(json.loads((ROOT/'submission.json').read_text())['job_id']);assert jid.isdigit();return jid

def stage():
    assert not (ROOT/'staging-intent.json').exists()
    names=['protocol.json','baseline_5k.json','known_15k.json','workflow.py','md_core.py','analysis_common.py','analyze_point.py','test_analysis.py','README_RU.txt']+[s['initial_file'] for s in CFG['states']]
    files={n:(ROOT/n).read_bytes() for n in names};hashes={n:sha(b) for n,b in files.items()}
    save('staging-intent.json',dict(remote=REMOTE,files=hashes,unix=time.time()))
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as t:
        for n,b in files.items():
            info=tarfile.TarInfo(n);info.size=len(b);info.mode=0o600;t.addfile(info,io.BytesIO(b))
    print(run('umask 077; mkdir '+shlex.quote(REMOTE)+' && tar -xzf - -C '+shlex.quote(REMOTE),content=buf.getvalue()))
    code='import pathlib,hashlib,shutil,json; p=pathlib.Path('+repr(REMOTE)+'); h='+repr(hashes)+'; assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==v for n,v in h.items()); models='+repr([CFG['checkpoint'],CFG['frozen_model']])+'; assert all(hashlib.sha256(pathlib.Path(m["path"]).read_bytes()).hexdigest()==m["sha256"] for m in models); shutil.copytree('+repr(OLD+'/analysis_deps')+',str(p/"analysis_deps")); (p/"staging-manifest.json").write_text(json.dumps(h)); print("Inputs/models verified; private analysis dependencies copied")'
    print(py(code));save('staging-complete.json',dict(remote=REMOTE,files=hashes,unix=time.time()))

def test():
    result=run('cd '+shlex.quote(REMOTE)+' && CUDA_VISIBLE_DEVICES= DEVICE=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR='+shlex.quote(REMOTE+'/cache/matplotlib')+' '+PYTHON+' -B test_analysis.py')
    print(result);(ROOT/'remote-tests.txt').write_text(result,encoding='utf-8');assert 'PASS' in result

def handoff():
    assert (ROOT/'staging-complete.json').exists() and 'PASS' in (ROOT/'remote-tests.txt').read_text()
    assert not (ROOT/'handoff-intent.json').exists(),'Inspect receipts/queue before any retry'
    jid=jobid();assert jid=='20962153'
    dispatcher='''# Pure-Ti temperature campaign handoff, authorized 2026-10-29.
from pathlib import Path
import os,sys,runpy
root=Path(REMOTE)
assert root.is_dir() and (root/'protocol.json').exists()
os.chdir(root);sys.path.insert(0,str(root))
for key,folder in [('TORCHINDUCTOR_CACHE_DIR','torchinductor'),('TORCH_EXTENSIONS_DIR','torch_extensions'),('TRITON_CACHE_DIR','triton'),('MPLCONFIGDIR','matplotlib')]:
    path=root/'cache'/folder;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
runpy.run_path(str(root/'workflow.py'),run_name='__main__')
'''.replace('REMOTE',repr(REMOTE))
    code='''import pathlib,json,hashlib,subprocess,time,os
old=pathlib.Path(OLD);new=pathlib.Path(REMOTE);jid=JID
assert not (new/'handoff.json').exists()
job=subprocess.check_output(['scontrol','show','job',jid,'-o'],universal_newlines=True)
assert 'JobId='+jid+' ' in job and 'WorkDir='+str(old) in job
assert 'JobState=RUNNING' in job or 'JobState=PENDING' in job
manifest=json.loads((new/'staging-manifest.json').read_text())
assert all(hashlib.sha256((new/n).read_bytes()).hexdigest()==v for n,v in manifest.items())
original=(old/'workflow.py').read_bytes()
assert hashlib.sha256(original).hexdigest()==EXPECTED
backup=old/'workflow.py.before-ti-temperature-20261029'
with backup.open('xb') as f:f.write(original)
counts={s:len(list((old/'states'/s/'replicas').glob('replica_*/summary.json'))) for s in ['xAl_0.20','xAl_0.45','xAl_0.80']}
progress=json.loads((old/'progress.json').read_text())
receipt=dict(job_id=jid,previous_remote=str(old),remote=str(new),old_completed=counts,old_progress=progress,previous_job=job,unix=time.time(),status='dispatcher_installed_requeue_requested',original_workflow_sha256=EXPECTED)
tmp=old/'workflow.py.temperature-handoff.tmp';tmp.write_text(DISPATCHER);tmp.replace(old/'workflow.py')
(new/'handoff.json').write_text(json.dumps(receipt,indent=2))
r=subprocess.run(['scontrol','requeue',jid],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True)
receipt.update(requeue_returncode=r.returncode,requeue_output=r.stdout,status='requeued' if r.returncode==0 else 'requeue_failed')
(new/'handoff.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
assert r.returncode==0,r.stdout
'''
    for k,v in [('DISPATCHER',dispatcher),('EXPECTED',json.loads((ROOT/'preparation_manifest.json').read_text())['source_md_sha256']),('REMOTE',REMOTE),('OLD',OLD),('JID',jid)]:code=code.replace(k,repr(v))
    save('handoff-intent.json',dict(job_id=jid,remote=REMOTE,unix=time.time(),dispatcher_sha256=sha(dispatcher.encode())))
    result=py(code);print(result);save('handoff.json',json.loads(result))

def collect(download=True):
    jid=jobid();account=run('squeue -j '+jid+' -o "%.18i %.24j %.12T %.10M %.10l %R"; sacct -j '+jid+' --format=JobID,State,ExitCode,Elapsed,Timelimit,AllocTRES,NodeList -Pn')
    (ROOT/'scheduler-latest.txt').write_text(account);print(account)
    code='''import pathlib,json
p=pathlib.Path(REMOTE);old=pathlib.Path(OLD)
d={'metadata':{},'logs':{},'completed':{}}
for n in ['preflight.json','progress.json','final.json','failure.json','analysis/latest.json','analysis/failure.json','handoff.json','stop-after-batch-20261106.json','control.json']:
 f=p/n
 if f.exists():d['metadata'][n]=json.loads(f.read_text())
for s in ['Ti_2100K','Ti_1873K','Ti_2000K']:d['completed'][s]=len(list((p/'states'/s/'replicas').glob('replica_*/summary.json')))
for f in [old/'workflow.log',old/('allocation-'+JID+'.log')]+sorted((p/'analysis').glob('*/*.log'),key=lambda f:f.stat().st_mtime)[-2:]:
 if f.exists():
  with f.open('rb') as fp:fp.seek(max(0,f.stat().st_size-3500));d['logs'][str(f)]=fp.read().decode(errors='replace')
plan=d['metadata'].get('stop-after-batch-20261106.json',{});pid=plan.get('guard_pid');d['stop_guard_alive']=bool(pid and pathlib.Path('/proc/'+str(pid)).exists())
print(json.dumps(d))
'''
    for k,v in [('REMOTE',REMOTE),('OLD',OLD),('JID',jid)]:code=code.replace(k,repr(v))
    data=json.loads(py(code));save('status.json',data)
    print(json.dumps({'completed':data['completed'],'metadata':{k:v for k,v in data['metadata'].items() if k!='analysis/latest.json'}},ensure_ascii=True,indent=2))
    if download:
        code='''import pathlib,tarfile,sys
p=pathlib.Path(REMOTE)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for n in ['analysis','preflight.json','progress.json','final.json','failure.json','handoff.json','stop-after-batch-20261106.json','control.json']:
  f=p/n
  if f.exists():t.add(f,arcname=n)
'''.replace('REMOTE',repr(REMOTE))
        payload=py(code,binary=True);dest=ROOT/'collected';dest.mkdir(exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as t:
            for m in t.getmembers():
                target=(dest/m.name).resolve();assert target.is_relative_to(dest.resolve()) and not m.issym() and not m.islnk()
                if m.isfile():target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(t.extractfile(m).read())
        save('collection.json',dict(unix=time.time(),bytes=len(payload),job_id=jid))

if __name__=='__main__':
    action=sys.argv[1]
    if action=='stage':stage()
    elif action=='test':test()
    elif action=='handoff':handoff()
    elif action in ['status','collect']:collect(action=='collect')
    elif action=='exec':print(run(Path(sys.argv[2]).read_text(encoding='utf-8')))
    else:raise ValueError(action)
