"""Four explicitly authorized r2SCAN jobs; atomic one-time submissions and scoped collection."""
from pathlib import Path
import json,sys,subprocess,importlib.util,hashlib,tarfile,io,shlex,time
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[2]
spec=importlib.util.spec_from_file_location('ssh_helpers',WORK.parent/'MLIP-research/scripts/cluster_ssh.py');ssh=importlib.util.module_from_spec(spec);spec.loader.exec_module(ssh)
REG=json.loads((ROOT/'registry.json').read_text());REMOTE=REG['remote'];PYTHON='/home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,data):
 path=Path(path);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(path)
def run(command,content=None,timeout=180,binary=False):
 result=subprocess.run(ssh.ssh_args()+['-o','BatchMode=yes','mlip-umt',command],input=content,capture_output=True,timeout=timeout)
 if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace')[-3000:]+('' if binary else result.stdout.decode(errors='replace')[-5000:]))
 return result.stdout if binary else result.stdout.decode(errors='replace')
def py(code,**kw):return run('python3 -c '+shlex.quote(code),**kw)
def stage():
 assert REG.get('execution_mode','four_jobs_v1')=='four_jobs_v1','Four-job mode superseded; use single_job_transition'
 assert not (ROOT/'staging-intent.json').exists(),'Reconcile staging receipt before retry'
 names=['protocol.json','initial.data','md_core.py','workflow.py','analysis_common.py','analyze_point.py','pooled_analysis.py','test_analysis.py','prequeue_check.py','job.sbatch','README_RU.txt','monitor.json']
 files={}
 for sid,entry in REG['runs'].items():
  p=ROOT/sid
  for name in names:files[sid+'/'+name]=(p/name).read_bytes()
  for f in sorted((p/'legacy').iterdir()):files[sid+'/legacy/'+f.name]=f.read_bytes()
 hashes={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
 save(ROOT/'staging-intent.json',dict(unix=time.time(),remote=REMOTE,files=hashes))
 buf=io.BytesIO()
 with tarfile.open(fileobj=buf,mode='w:gz') as t:
  for name,data in files.items():
   info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o600;t.addfile(info,io.BytesIO(data))
 run('umask 077; mkdir '+shlex.quote(REMOTE)+' && tar -xzf - -C '+shlex.quote(REMOTE),content=buf.getvalue(),timeout=600)
 code="""import pathlib,hashlib,json,shutil,subprocess
p=pathlib.Path(REMOTE);h=HASHES;sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert all(sha(p/n)==v for n,v in h.items())
shutil.copytree('/home6/u3129/tial-testsets/pbe15k-ti-temperature-20261029-01/analysis_deps',str(p/'analysis_deps'))
for sid in SIDS:
 q=p/sid;(q/'analysis_deps').symlink_to('../analysis_deps',target_is_directory=True)
 cfg=json.loads((q/'protocol.json').read_text())
 for key in ['checkpoint','frozen_model']:assert sha(pathlib.Path(cfg[key]['path']))==cfg[key]['sha256']
 manifest={n[len(sid)+1:]:v for n,v in h.items() if n.startswith(sid+'/') and not n.endswith('/monitor.json')}
 (q/'staging-manifest.json').write_text(json.dumps(manifest,indent=2))
 subprocess.check_call(['bash','-n',str(q/'job.sbatch')])
print('STAGE_PASS four temperatures and immutable legacy snapshots')
""".replace('REMOTE',repr(REMOTE)).replace('HASHES',repr(hashes)).replace('SIDS',repr(list(REG['runs'])))
 print(py(code,timeout=600));save(ROOT/'staging-complete.json',dict(unix=time.time(),remote=REMOTE,files=hashes))
def validate():
 assert (ROOT/'staging-complete.json').exists()
 for sid,entry in REG['runs'].items():
  q=entry['remote'];command='cd '+shlex.quote(q)+' && CUDA_VISIBLE_DEVICES= DEVICE=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR='+shlex.quote(q+'/cache/matplotlib')+' '+PYTHON+' -B prequeue_check.py'
  print('Validating',sid,flush=True)
  result=run(command,timeout=1200);(ROOT/sid/'cpu-validation.log').write_text(result,encoding='utf-8')
  check=json.loads(py('import pathlib;print((pathlib.Path('+repr(q)+')/"prequeue-validation.json").read_text())'))
  save(ROOT/sid/'prequeue-validation.json',check);assert check['status']=='PASS'
  print(sid,'PASS; max pressure error bar',check['max_error_bar'],'legacy eta',check['baseline']['eta_mPa_s'],flush=True)
def submit():
 assert REG.get('execution_mode','four_jobs_v1')=='four_jobs_v1','No four-job sbatch: superseded by one inherited ID'
 assert all(json.loads((ROOT/sid/'prequeue-validation.json').read_text())['status']=='PASS' for sid in REG['runs'])
 for sid,entry in REG['runs'].items():
  if entry.get('job_id'):
   print('Already recorded',sid,entry['job_id']);continue
  intent=ROOT/sid/'submit-intent.json';assert not intent.exists(),'Submission attempted; reconcile remote receipt and queue before retry'
  save(intent,dict(unix=time.time(),remote=entry['remote'],authorized_by='Human2026-11-06 requested all four independent r2SCAN temperatures',status='attempting_once'))
  code="""import pathlib,json,hashlib,subprocess,time
p=pathlib.Path(REMOTE)
if (p/'submission.json').exists():
 print((p/'submission.json').read_text());raise SystemExit(0)
(p/'submission.guard').mkdir()
assert not (p/'submit-intent.json').exists(),'Prior attempt needs reconciliation'
cfg=json.loads((p/'protocol.json').read_text());a=cfg['allocation'];sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
h=json.loads((p/'staging-manifest.json').read_text());assert all(sha(p/n)==v for n,v in h.items())
check=json.loads((p/'prequeue-validation.json').read_text());assert check['status']=='PASS' and check['protocol_sha256']==sha(p/'protocol.json')
assert check.get('final_cpu_tests','').startswith('PASS') and check['validated_staging_manifest_sha256']==sha(p/'staging-manifest.json')
for key in ['checkpoint','frozen_model']:assert sha(pathlib.Path(cfg[key]['path']))==cfg[key]['sha256']
T=cfg['states'][0]['temperature_K'];name='Ti'+str(T)+'-visc-r2SCAN'
queue=subprocess.check_output(['squeue','-h','-u','u3129','-o','%i|%j|%k'],universal_newlines=True)
assert name not in queue and cfg['run_id'] not in queue,'Matching job already queued'
args=['sbatch','--parsable','--partition='+a['partition'],'--account='+a['account'],'--qos='+a['qos'],'--nodes=1','--ntasks=1','--cpus-per-task='+str(a['cpus']),'--gres=gpu:'+a['gpu'],'--mem='+a['memory'],'--time='+a['walltime'],'--requeue','--job-name='+name,'--comment='+cfg['run_id'],'--chdir='+str(p),'--output='+str(p/'allocation-%j.log'),'--open-mode=append',str(p/'job.sbatch')]
intent=dict(unix=time.time(),remote=str(p),run_id=cfg['run_id'],args=args,status='attempting_once');(p/'submit-intent.json').write_text(json.dumps(intent,indent=2))
r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
receipt=dict(intent,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,status='submission_failed')
if r.returncode==0:
 jid=r.stdout.strip().split(';')[0];assert jid.isdigit() and jid not in ['20962153','20987437'];receipt.update(job_id=jid,status='submitted',new_sbatch_created=True)
(p/'submission.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt));assert r.returncode==0
""".replace('REMOTE',repr(entry['remote']))
  receipt=json.loads(py(code));assert receipt['status']=='submitted';save(ROOT/sid/'submission.json',receipt)
  entry['job_id']=receipt['job_id'];save(ROOT/'registry.json',REG);print(sid,receipt['job_id'],flush=True)
def _collect_four_jobs_v1():
 for sid,entry in REG['runs'].items():
  receipt=ROOT/sid/'submission.json'
  if not receipt.exists():continue
  jid=str(json.loads(receipt.read_text())['job_id']);assert jid.isdigit() and jid not in ['20962153','20987437']
  queue=run('squeue -j '+jid+' -o "%.18i %.24j %.12T %.10M %.10l %R"; sacct -j '+jid+' --format=JobID,State,ExitCode,Elapsed,Timelimit,AllocTRES,NodeList -Pn')
  (ROOT/sid/'scheduler-latest.txt').write_text(queue,encoding='utf-8')
  code="""import pathlib,json,subprocess,hashlib
p=pathlib.Path(REMOTE);data={'metadata':{},'completed':{},'logs':{}}
for name in ['submission.json','preflight.json','progress.json','final.json','failure.json','analysis/latest.json','analysis/failure.json']:
 f=p/name
 if f.exists():data['metadata'][name]=json.loads(f.read_text())
data['completed'][SID]=len(list((p/'states'/SID/'replicas').glob('replica_*/summary.json')))
for f in [p/'workflow.log',p/('allocation-'+JID+'.log')]:
 if f.exists():
  with f.open('rb') as fp:fp.seek(max(0,f.stat().st_size-3500));data['logs'][f.name]=fp.read().decode(errors='replace')
cfg=json.loads((p/'protocol.json').read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
staged=json.loads((p/'staging-manifest.json').read_text());assert all(sha(p/name)==expected for name,expected in staged.items())
assert all(sha(pathlib.Path(cfg[key]['path']))==cfg[key]['sha256'] for key in ['checkpoint','frozen_model'])
assert sha(p/cfg['states'][0]['initial_file'])==cfg['states'][0]['initial_sha256']
assert cfg['atoms']==512 and cfg['states'][0]['counts']=={'Al':0,'Ti':512} and cfg['dt_ps']==.001 and cfg['production_steps']==40000
assert sha(p/cfg['legacy']['manifest'])==cfg['legacy']['sha256']
pre=data['metadata'].get('preflight.json')
if pre:
 assert pre['status']=='passed' and 'A100' in pre['device']['name'] and pre['checkpoint_sha256']==cfg['checkpoint']['sha256'] and pre['frozen_sha256']==cfg['frozen_model']['sha256'] and pre['protocol_sha256']==sha(p/'protocol.json')
latest=data['metadata'].get('analysis/latest.json',{}).get('points',{}).get(SID)
checked=[]
if latest:
 assert latest['protocol_sha256']==sha(p/'protocol.json') and latest['checkpoint_sha256']==cfg['checkpoint']['sha256']
 n=latest['n_replicas'];assert n>=5 and n%5==0
 for i in range(n-4,n+1):
  m=json.loads((p/'states'/SID/'replicas'/('replica_%04d'%i)/'summary.json').read_text())
  assert m['n_samples']==40001 and m['temperature_K']==cfg['states'][0]['temperature_K'] and m['protocol_sha256']==sha(p/'protocol.json') and m['checkpoint_sha256']==cfg['checkpoint']['sha256']
  assert sha(p/m['stress_file'])==m['stress_sha256'];checked.append(i)
data['runtime_checks']={'status':'PASS','models_and_initial_hashes':True,'protocol_sha256':sha(p/'protocol.json'),'latest_new5_raw_sha_checked':checked,'normalization':'state-specific V/(kB*T),bar->Pa,five shear channels;baseline reproduced in prequeue validation'}
sc=subprocess.run(['scontrol','show','job','-o',JID],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
data['scontrol']=sc.stdout.strip();data['scontrol_returncode']=sc.returncode
if sc.returncode==0:assert 'gres/gpu:a100=1' in sc.stdout or 'gres:gpu:a100:1' in sc.stdout
print(json.dumps(data))
""".replace('REMOTE',repr(entry['remote'])).replace('SID',repr(sid)).replace('JID',repr(jid))
  data=json.loads(py(code));save(ROOT/sid/'status.json',data)
  code="""import pathlib,tarfile,sys
p=pathlib.Path(REMOTE)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for name in ['analysis','preflight.json','progress.json','final.json','failure.json']:
  f=p/name
  if f.exists():t.add(f,arcname=name)
""".replace('REMOTE',repr(entry['remote']))
  payload=py(code,binary=True);dest=ROOT/sid/'collected';dest.mkdir(exist_ok=True)
  with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as t:
   for m in t.getmembers():
    target=(dest/m.name).resolve();assert target.is_relative_to(dest.resolve()) and not m.issym() and not m.islnk()
    if m.isfile():target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(t.extractfile(m).read())
  save(ROOT/sid/'collection.json',dict(unix=time.time(),job_id=jid,bytes=len(payload)))
  print(queue);print(sid,json.dumps(dict(completed=data['completed'],progress=data['metadata'].get('progress.json'),preflight=data['metadata'].get('preflight.json',{}).get('status'))),flush=True)
def collect():
 active=REG.get('active_campaign')
 if active:
  spec=importlib.util.spec_from_file_location('single_job_active_cluster',Path(active['local'])/'cluster.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module.collect()
 return _collect_four_jobs_v1()
if __name__=='__main__':
 action=sys.argv[1]
 if action=='stage':stage()
 elif action=='validate':validate()
 elif action=='submit':submit()
 elif action=='collect':collect()
 else:raise ValueError(action)
