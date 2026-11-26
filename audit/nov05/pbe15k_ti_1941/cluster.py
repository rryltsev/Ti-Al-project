"""Dedicated1941K job: one guarded initial sbatch, then sameID requeue only."""
from pathlib import Path
import sys,importlib.util,json,hashlib,shlex,subprocess,time,tarfile,io,contextlib
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[2]
spec=importlib.util.spec_from_file_location('ssh_helpers',WORK.parent/'MLIP-research/scripts/cluster_ssh.py');ssh=importlib.util.module_from_spec(spec);spec.loader.exec_module(ssh)
CFG=json.loads((ROOT/'protocol.json').read_text());REMOTE=CFG['remote'];PYTHON='/home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python'
DEPS='/home6/u3129/tial-testsets/pbe15k-ti-temperature-20261029-01/analysis_deps'
sha=lambda b:hashlib.sha256(b).hexdigest()
def save(name,obj):(ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def run(command,content=None,timeout=180,binary=False):
 r=subprocess.run(ssh.ssh_args()+['-o','BatchMode=yes','mlip-umt',command],input=content,capture_output=True,timeout=timeout)
 if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace')[-1500:]+('' if binary else r.stdout.decode(errors='replace')[-1500:]))
 return r.stdout if binary else r.stdout.decode('utf-8',errors='replace')
def py(code,**kw):return run('python3 -c '+shlex.quote(code),**kw)
def jobid():
 j=str(json.loads((ROOT/'submission.json').read_text())['job_id']);assert j.isdigit() and j!='20962153';return j
def stage():
 assert not (ROOT/'staging-intent.json').exists(),'Inspect stage receipts before retry'
 names=['protocol.json','baseline_5k.json','known_15k.json','md_core.py','workflow.py','analysis_common.py','analyze_point.py','test_analysis.py','README_RU.txt','job.sbatch','initial_Ti_1941K.data'];files={n:(ROOT/n).read_bytes() for n in names};hashes={n:sha(b) for n,b in files.items()}
 save('staging-intent.json',dict(unix=time.time(),remote=REMOTE,files=hashes))
 buf=io.BytesIO()
 with tarfile.open(fileobj=buf,mode='w:gz') as t:
  for n,b in files.items():
   info=tarfile.TarInfo(n);info.size=len(b);info.mode=0o600;t.addfile(info,io.BytesIO(b))
 run('umask 077; mkdir '+shlex.quote(REMOTE)+' && tar -xzf - -C '+shlex.quote(REMOTE),content=buf.getvalue())
 code="import pathlib,hashlib,json,shutil; p=pathlib.Path("+repr(REMOTE)+"); h="+repr(hashes)+"; assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==v for n,v in h.items()); cfg=json.loads((p/'protocol.json').read_text()); assert all(hashlib.sha256(pathlib.Path(cfg[k]['path']).read_bytes()).hexdigest()==cfg[k]['sha256'] for k in ['checkpoint','frozen_model']); shutil.copytree("+repr(DEPS)+",str(p/'analysis_deps')); (p/'staging-manifest.json').write_text(json.dumps(h,indent=2)); print('STAGE_PASS')"
 print(py(code));print(run('bash -n '+shlex.quote(REMOTE+'/job.sbatch')));save('staging-complete.json',dict(unix=time.time(),remote=REMOTE,files=hashes))
def test():
 cmd='cd '+shlex.quote(REMOTE)+' && CUDA_VISIBLE_DEVICES= DEVICE=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR='+shlex.quote(REMOTE+'/cache/matplotlib')+' '+PYTHON+' -B test_analysis.py'
 result=run(cmd);assert '"status": "PASS"' in result,result
 (ROOT/'cpu-tests.txt').write_text(result,encoding='utf-8')
 print(py("import pathlib; p=pathlib.Path("+repr(REMOTE)+"); (p/'cpu-tests.txt').write_text("+repr(result)+"); print('CPU_TESTS_RECORDED')"));print(result)
def submit():
 assert (ROOT/'staging-complete.json').exists() and '"status": "PASS"' in (ROOT/'cpu-tests.txt').read_text()
 assert not (ROOT/'submit-intent.json').exists(),'Do not repeat sbatch; inspect queue/remote submission receipt'
 save('submit-intent.json',dict(unix=time.time(),remote=REMOTE,run_id=CFG['run_id'],authorization='User requested separate1941K oneA100 job2026-11-05',status='attempting_once'))
 code=r"""import pathlib,json,hashlib,subprocess,time,fcntl
p=pathlib.Path(REMOTE);lock=(p/'submission.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (p/'submission.json').exists():
 print((p/'submission.json').read_text());raise SystemExit(0)
assert not (p/'submit-intent.json').exists(),'Remote submission already attempted; reconcile before further action'
cfg=json.loads((p/'protocol.json').read_text());a=cfg['allocation'];h=json.loads((p/'staging-manifest.json').read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();assert all(sha(p/n)==v for n,v in h.items());assert '"status": "PASS"' in (p/'cpu-tests.txt').read_text()
for k in ['checkpoint','frozen_model']:assert sha(pathlib.Path(cfg[k]['path']))==cfg[k]['sha256']
jobname='Ti1941-visc-PBE15k';queue=subprocess.check_output(['squeue','-h','-u','u3129','-o','%i|%j|%k'],universal_newlines=True);assert jobname not in queue and cfg['run_id'] not in queue,'Matching queued job exists'
args=['sbatch','--parsable','--partition='+a['partition'],'--nodelist='+a['node'],'--account='+a['account'],'--qos='+a['qos'],'--nodes=1','--ntasks=1','--cpus-per-task='+str(a['cpus']),'--gres=gpu:'+a['gpu'],'--mem='+a['memory'],'--time='+a['walltime'],'--requeue','--job-name='+jobname,'--comment='+cfg['run_id'],'--chdir='+str(p),'--output='+str(p/'allocation-%j.log'),'--open-mode=append',str(p/'job.sbatch')]
intent=dict(unix=time.time(),remote=str(p),run_id=cfg['run_id'],args=args,status='attempting_once');(p/'submit-intent.json').write_text(json.dumps(intent,indent=2))
r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True);receipt=dict(intent,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
if r.returncode==0:
 jid=r.stdout.strip().split(';')[0];assert jid.isdigit() and jid!='20962153';receipt.update(job_id=jid,status='submitted',new_sbatch_created=True)
else:receipt['status']='submission_failed'
(p/'submission.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt));assert r.returncode==0
""".replace('REMOTE',repr(REMOTE))
 result=json.loads(py(code));save('submission.json',result);print(json.dumps(result,indent=2))
def collect():
 jid=jobid();account=run('squeue -j '+jid+' -o "%.18i %.24j %.12T %.10M %.10l %R"; sacct -j '+jid+' --format=JobID,State,ExitCode,Elapsed,Timelimit,AllocTRES,NodeList -Pn');(ROOT/'scheduler-latest.txt').write_text(account,encoding='utf-8')
 code=r"""import pathlib,json
p=pathlib.Path(REMOTE);d={'metadata':{},'logs':{},'completed':{}}
for n in ['preflight.json','progress.json','final.json','failure.json','analysis/latest.json','analysis/failure.json','submission.json','stop-after-batch-20261106.json','control.json']:
 f=p/n
 if f.exists():d['metadata'][n]=json.loads(f.read_text())
d['completed']['Ti_1941K']=len(list((p/'states/Ti_1941K/replicas').glob('replica_*/summary.json')))
for f in [p/'workflow.log',p/('allocation-'+JID+'.log')]+sorted((p/'analysis').glob('*/*.log'),key=lambda f:f.stat().st_mtime)[-1:]:
 if f.exists():
  with f.open('rb') as fp:fp.seek(max(0,f.stat().st_size-3500));d['logs'][str(f)]=fp.read().decode(errors='replace')
plan=d['metadata'].get('stop-after-batch-20261106.json',{});pid=plan.get('guard_pid');d['stop_guard_alive']=bool(pid and pathlib.Path('/proc/'+str(pid)).exists())
print(json.dumps(d))
""".replace('REMOTE',repr(REMOTE)).replace('JID',repr(jid))
 data=json.loads(py(code));save('status.json',data)
 code=r"""import pathlib,tarfile,sys
p=pathlib.Path(REMOTE)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for n in ['analysis','preflight.json','progress.json','final.json','failure.json','stop-after-batch-20261106.json','control.json']:
  f=p/n
  if f.exists():t.add(f,arcname=n)
""".replace('REMOTE',repr(REMOTE))
 payload=py(code,binary=True);dest=ROOT/'collected';dest.mkdir(exist_ok=True)
 with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as t:
  for m in t.getmembers():
   target=(dest/m.name).resolve();assert target.is_relative_to(dest.resolve()) and not m.issym() and not m.islnk()
   if m.isfile():target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(t.extractfile(m).read())
 save('collection.json',dict(unix=time.time(),bytes=len(payload),job_id=jid));print(account);print(json.dumps(dict(completed=data['completed'],metadata=list(data['metadata']),progress=data['metadata'].get('progress.json'),preflight=data['metadata'].get('preflight.json',{}).get('status')),indent=2))
if __name__=='__main__':
 action=sys.argv[1]
 if action=='stage':stage()
 elif action=='test':test()
 elif action=='submit':submit()
 elif action=='collect':collect()
 else:raise ValueError(action)
