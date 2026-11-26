"""Deploy/monitor the single inherited r2SCAN refinement job; no sbatch exists here."""
from pathlib import Path
import importlib.util,json,hashlib,sys,io,tarfile,time,shlex,base64
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('legacy_cluster_helpers',ROOT.parent/'cluster.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
run=base.run;py=base.py;PYTHON=base.PYTHON
REG=json.loads((ROOT/'registry.json').read_text());REMOTE=REG['remote'];JID=REG['job_id']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,data):
 path=Path(path);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(path)
def stage():
 assert not (ROOT/'staging-intent.json').exists(),'Reconcile staging receipts before any retry'
 names=['protocol.json','plan.json','registry.json','inventory.json','data_confirmation.json','workflow.py','run_block.py','test_serial.py','README_RU.txt']
 files={name:(ROOT/name).read_bytes() for name in names}
 for sid in REG['runs']:
  for name in ['protocol.json','initial.data','verified_legacy_baseline.json','md_core.py','analysis_common.py','pooled_analysis.py','analyze_point.py','test_analysis.py','README_RU.txt']:
   files[sid+'/'+name]=(ROOT/sid/name).read_bytes()
  for f in (ROOT/sid/'legacy').iterdir():files[sid+'/legacy/'+f.name]=f.read_bytes()
 hashes={n:hashlib.sha256(b).hexdigest() for n,b in files.items()};save(ROOT/'staging-intent.json',dict(unix=time.time(),remote=REMOTE,files=hashes))
 buf=io.BytesIO()
 with tarfile.open(fileobj=buf,mode='w:gz') as t:
  for n,b in files.items():
   m=tarfile.TarInfo(n);m.size=len(b);m.mode=0o600;t.addfile(m,io.BytesIO(b))
 run('umask 077; mkdir '+shlex.quote(REMOTE)+' && tar -xzf - -C '+shlex.quote(REMOTE),content=buf.getvalue(),timeout=600)
 code="""import pathlib,hashlib,json,shutil
p=pathlib.Path(REMOTE);h=HASHES;sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();assert all(sha(p/n)==v for n,v in h.items())
shutil.copytree('/home6/u3129/tial-testsets/r2scan-ti-temperature-20261106-01/analysis_deps',str(p/'analysis_deps'))
for sid in SIDS:(p/sid/'analysis_deps').symlink_to('../analysis_deps',target_is_directory=True)
(p/'staging-manifest.json').write_text(json.dumps(h,indent=2));print('SINGLE_JOB_STAGE_PASS')
""".replace('REMOTE',repr(REMOTE)).replace('HASHES',repr(hashes)).replace('SIDS',repr(list(REG['runs'])))
 print(py(code,timeout=600));save(ROOT/'staging-complete.json',dict(unix=time.time(),files=hashes,remote=REMOTE))
def validate():
 assert (ROOT/'staging-complete.json').exists()
 # Model identity audit is reused only when all raw/model hashes are exactly unchanged.
 code="""import pathlib,hashlib,json
p=pathlib.Path(REMOTE);cfg=json.loads((p/'protocol.json').read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert all(sha(p/n)==h for n,h in json.loads((p/'staging-manifest.json').read_text()).items())
ack=json.loads((p/'data_confirmation.json').read_text());assert ack['status']=='confirmed' and ack['inventory_sha256']==sha(p/'inventory.json')==cfg['inventory_sha256']
for key in ['checkpoint','frozen_model']:assert sha(pathlib.Path(cfg[key]['path']))==cfg[key]['sha256']
seen=set();rows=[]
for sid,e in cfg['states'].items():
 q=p/sid;c=json.loads((q/'protocol.json').read_text());m=json.loads((q/'legacy/manifest.json').read_text());old=json.loads((q/'verified_legacy_baseline.json').read_text())
 assert sha(q/'protocol.json')==e['protocol_sha256'] and sha(q/'verified_legacy_baseline.json')==c['previous_model_audit']['sha256']
 assert old['status']=='PASS' and old['checkpoint_sha256']==c['checkpoint']['sha256']==cfg['checkpoint']['sha256']
 assert sha(q/'legacy/manifest.json')==c['legacy']['sha256']==old['legacy_manifest_sha256']
 assert sha(q/'legacy/stress.npz')==m['stress_sha256'] and sha(q/'initial.data')==c['states'][0]['initial_sha256']
 assert c['production_steps']==40000 and c['dt_ps']==.001 and c['states'][0]['counts']=={'Al':0,'Ti':512} and 'min_replicas' not in c['convergence']
 for item in m['raw_manifest']:
  assert sha(q/item['mol_file'])==item['mol_sha256'] and item['source_pressure_sha256'] not in seen;seen.add(item['source_pressure_sha256'])
 rows.append({'id':sid,'legacy_replicas':len(m['raw_manifest']),'baseline':old['baseline']['eta_mPa_s'],'max_pressure_error_bar':old['max_error_bar']})
assert len(seen)==75
(p/'input-validation.json').write_text(json.dumps({'status':'PASS','same_model_and_legacy_raw':True,'distinct_legacy_replicas':75,'rows':rows},indent=2));print(json.dumps(rows))
""".replace('REMOTE',repr(REMOTE))
 print(py(code))
 for folder,script in [(REMOTE,'test_serial.py')]+[(e['remote'],'test_analysis.py') for e in REG['runs'].values()]:
  cmd='cd '+shlex.quote(folder)+' && CUDA_VISIBLE_DEVICES= DEVICE=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR='+shlex.quote(folder+'/cache/matplotlib')+' '+PYTHON+' -B '+script
  result=run(cmd,timeout=600);assert '"status": "PASS"' in result,result
  print(folder.split('/')[-1],result.strip(),flush=True)
 result=json.loads(py("import pathlib,json;print((pathlib.Path("+repr(REMOTE)+")/'input-validation.json').read_text())"));result.update(final_tests='PASS',unix=time.time());save(ROOT/'validation.json',result)
 py("import pathlib; (pathlib.Path("+repr(REMOTE)+")/'validation.json').write_text("+repr(json.dumps(result,indent=2))+")")
def install():
 assert json.loads((ROOT/'validation.json').read_text())['status']=='PASS'
 assert not (ROOT/'transition-intent.json').exists(),'Reconcile transition receipt, queue and dispatcher before retry'
 code_text="import os,sys\nos.execv(sys.executable,[sys.executable,'-B','-u',"+repr(REMOTE+'/workflow.py')+"])\n"
 old_workflow=ROOT.parent/'Ti_1873K/workflow.py';expected=sha(old_workflow)
 save(ROOT/'transition-intent.json',dict(unix=time.time(),job_id=JID,remote=REMOTE,old_workflow_sha256=expected,new_dispatcher_sha256=hashlib.sha256(code_text.encode()).hexdigest(),superseded=REG['superseded_job_ids']))
 code=r'''import pathlib,json,hashlib,subprocess,time,base64
p=pathlib.Path(REMOTE);old=pathlib.Path(OLD);sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();reg=json.loads((p/'registry.json').read_text());actions=[]
assert not (p/'transition-receipt.json').exists()
for jid in [reg['job_id']]+reg['superseded_job_ids']:
 assert jid not in ['20962153','20987437']
 job=subprocess.check_output(['scontrol','show','job','-o',jid],universal_newlines=True)
 assert 'UserId=u3129(' in job and 'JobState=PENDING' in job and 'Priority=0 ' in job
assert json.loads((p/'validation.json').read_text())['status']=='PASS'
assert all(sha(p/n)==h for n,h in json.loads((p/'staging-manifest.json').read_text()).items())
ack=json.loads((p/'data_confirmation.json').read_text());assert ack['status']=='confirmed'
assert sha(old/'workflow.py')==EXPECTED
assert not list((old/'states/Ti_1873K/replicas').glob('replica_*/summary.json'))
(old/'workflow.before-single-job-v2.py').write_bytes((old/'workflow.py').read_bytes())
dispatcher=base64.b64decode(DISPATCHER);temp=old/'workflow.py.transition.tmp';temp.write_bytes(dispatcher);temp.replace(old/'workflow.py')
subprocess.check_call(['scontrol','update','JobId='+reg['job_id'],'JobName=Ti-visc-r2SCAN-refine','Comment='+reg['run_id']])
for jid in reg['superseded_job_ids']:
 before=subprocess.check_output(['scontrol','show','job','-o',jid],universal_newlines=True)
 assert 'JobState=PENDING' in before and 'Priority=0 ' in before
 r=subprocess.run(['scancel',jid],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True);assert r.returncode==0,r.stderr
 actions.append({'job_id':jid,'action':'cancel_superseded_unstarted_temperature_job','stdout':r.stdout,'stderr':r.stderr})
receipt={'unix':time.time(),'status':'single_job_prepared_held','job_id':reg['job_id'],'dispatcher_remote':str(old/'workflow.py'),'dispatcher_sha256':sha(old/'workflow.py'),'original_workflow_sha256':EXPECTED,'remote':str(p),'cancelled_jobs':actions,'new_sbatch_created':False,'data_confirmation':'confirmed'}
(p/'transition-receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
'''.replace('REMOTE',repr(REMOTE)).replace('OLD',repr(REG['dispatcher_remote'])).replace('EXPECTED',repr(expected)).replace('DISPATCHER',repr(base64.b64encode(code_text.encode()).decode()))
 result=json.loads(py(code));save(ROOT/'transition-receipt.json',result)
 # Store the one active allocation receipt; original per-temperature receipts remain archived.
 receipt=json.loads((ROOT.parent/'Ti_1873K/submission.json').read_text());receipt.update(mode='single_job_sequential_refinement_v2',remote=REMOTE,dispatcher_remote=REG['dispatcher_remote'],status='prepared_held',new_sbatch_created=False,transition=result)
 save(ROOT/'submission.json',receipt)
 print(json.dumps(result),flush=True)
def release():
 receipt=json.loads((ROOT/'submission.json').read_text());assert receipt['status']=='prepared_held'
 code=r'''import pathlib,json,hashlib,subprocess,time
p=pathlib.Path(REMOTE);reg=json.loads((p/'registry.json').read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();r=json.loads((p/'transition-receipt.json').read_text())
assert json.loads((p/'data_confirmation.json').read_text())['status']=='confirmed' and json.loads((p/'validation.json').read_text())['status']=='PASS'
assert all(sha(p/n)==h for n,h in json.loads((p/'staging-manifest.json').read_text()).items()) and sha(pathlib.Path(r['dispatcher_remote']))==r['dispatcher_sha256']
job=subprocess.check_output(['scontrol','show','job','-o',reg['job_id']],universal_newlines=True);assert 'JobState=PENDING' in job and 'Priority=0 ' in job
subprocess.check_call(['scontrol','release',reg['job_id']]);after=subprocess.check_output(['scontrol','show','job','-o',reg['job_id']],universal_newlines=True)
r.update(status='single_job_released',released_unix=time.time(),scheduler_after_release=after);(p/'transition-receipt.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''.replace('REMOTE',repr(REMOTE))
 result=json.loads(py(code));save(ROOT/'transition-receipt.json',result);receipt.update(status='single_job_released',transition=result);save(ROOT/'submission.json',receipt)
 py("import pathlib; (pathlib.Path("+repr(REMOTE)+")/'submission.json').write_text("+repr(json.dumps(receipt,indent=2))+")")
 print(json.dumps(dict(job_id=JID,status=result['status'],new_sbatch_created=False)),flush=True)
def collect():
 jid=str(json.loads((ROOT/'submission.json').read_text())['job_id']);assert jid==JID and jid not in ['20962153','20987437']
 scheduler=run('squeue -j '+jid+' -o "%.18i %.24j %.12T %.10M %.10l %R"; sacct -j '+jid+' --format=JobID,State,ExitCode,Elapsed,Timelimit,AllocTRES,NodeList -Pn');(ROOT/'scheduler-latest.txt').write_text(scheduler,encoding='utf-8')
 code=r'''import pathlib,json,hashlib,subprocess,time
p=pathlib.Path(REMOTE);reg=json.loads((p/'registry.json').read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();out={'metadata':{},'states':{},'checked_unix':time.time()}
assert all(sha(p/n)==h for n,h in json.loads((p/'staging-manifest.json').read_text()).items())
for key in ['checkpoint','frozen_model']:assert sha(pathlib.Path(reg[key]['path']))==reg[key]['sha256']
for name in ['progress.json','final.json','failure.json','submission.json','transition-receipt.json','plan.json','gpu-export.json','gpu-model-binding.json','gpu-export-repair.json']:
 if (p/name).exists():out['metadata'][name]=json.loads((p/name).read_text())
for sid,e in reg['runs'].items():
 q=p/sid;c=json.loads((q/'protocol.json').read_text());d={'completed':len(list((q/'states'/sid/'replicas').glob('replica_*/summary.json'))),'metadata':{}}
 assert sha(q/'protocol.json')==e['protocol_sha256']
 for name in ['preflight.json','progress.json','failure.json','analysis/failure.json','analysis/latest_'+sid+'.json']:
  if (q/name).exists():d['metadata'][name]=json.loads((q/name).read_text())
 pre=d['metadata'].get('preflight.json')
 if pre:assert pre['status']=='passed' and 'A100' in pre['device']['name'] and pre['device']['job_id']==reg['job_id'] and pre['protocol_sha256']==e['protocol_sha256'] and pre['checkpoint_sha256']==reg['checkpoint']['sha256'] and pre['frozen_sha256']==reg['frozen_model']['sha256']
 latest=d['metadata'].get('analysis/latest_'+sid+'.json')
 if latest:
  assert latest['protocol_sha256']==e['protocol_sha256'];n=latest['n_replicas'];assert n>=5 and n%5==0
  for i in range(n-4,n+1):
   m=json.loads((q/'states'/sid/'replicas'/('replica_%04d'%i)/'summary.json').read_text());assert m['n_samples']==40001 and m['temperature_K']==e['temperature_K'] and m['protocol_sha256']==e['protocol_sha256'] and m['checkpoint_sha256']==reg['checkpoint']['sha256'] and sha(q/m['stress_file'])==m['stress_sha256']
 d['protocol_text']=(q/'protocol.json').read_text();d['runtime_checks']='PASS';out['states'][sid]=d
s=subprocess.run(['scontrol','show','job','-o',reg['job_id']],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True);out['scontrol']=s.stdout;out['scontrol_returncode']=s.returncode
if s.returncode==0:assert 'gres/gpu:a100=1' in s.stdout or 'gres:gpu:a100:1' in s.stdout
out['active_registry_text']=(p/'registry.json').read_text();out['active_protocol_text']=(p/'protocol.json').read_text();out['runtime_checks']='PASS';print(json.dumps(out))
'''.replace('REMOTE',repr(REMOTE))
 status=json.loads(py(code));save(ROOT/'status.json',status)
 # GPU export pins the device-specific artifact before MD. Preserve exact remote
 # protocol bytes so local provenance hashes match the replica summaries.
 (ROOT/'registry.json').write_bytes(status['active_registry_text'].encode())
 (ROOT/'protocol.json').write_bytes(status['active_protocol_text'].encode())
 for sid,d in status['states'].items():(ROOT/sid/'protocol.json').write_bytes(d['protocol_text'].encode())
 for name in ['gpu-export.json','gpu-model-binding.json','gpu-export-repair.json','transition-receipt.json']:
  if name in status['metadata']:save(ROOT/name,status['metadata'][name])
 code=r'''import pathlib,sys,tarfile,json
p=pathlib.Path(REMOTE);reg=json.loads((p/'registry.json').read_text())
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for name in ['progress.json','final.json','failure.json']:
  if (p/name).exists():t.add(p/name,arcname='campaign/'+name)
 for sid in reg['runs']:
  for name in ['analysis','preflight.json','progress.json','failure.json']:
   if (p/sid/name).exists():t.add(p/sid/name,arcname=sid+'/'+name)
'''.replace('REMOTE',repr(REMOTE))
 payload=py(code,binary=True)
 with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as t:
  for m in t.getmembers():
   parts=Path(m.name).parts;assert parts and parts[0] in ['campaign']+list(REG['runs']) and not m.issym() and not m.islnk()
   dest=ROOT/'collected' if parts[0]=='campaign' else ROOT/parts[0]/'collected';target=dest.joinpath(*parts[1:]).resolve();assert target.is_relative_to(dest.resolve())
   if m.isfile():target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(t.extractfile(m).read())
 # A recovered technical failure remains archived, never an active failure.
 for sid in ['campaign']+list(REG['runs']):
  folder=ROOT/'collected' if sid=='campaign' else ROOT/sid/'collected'
  metadata=status['metadata'] if sid=='campaign' else status['states'][sid]['metadata']
  for name in ['failure.json','analysis/failure.json']:
   stale=folder/name
   if stale.exists() and name not in metadata:
    archived=ROOT/'technical_failures/local_collected'/str(int(time.time()*1000))/sid/name;archived.parent.mkdir(parents=True,exist_ok=True);stale.replace(archived)
 save(ROOT/'collection.json',dict(unix=time.time(),bytes=len(payload),job_id=jid))
 print(scheduler);print(json.dumps(dict(runtime_checks=status['runtime_checks'],progress=status['metadata'].get('progress.json'),final=status['metadata'].get('final.json'),states={sid:dict(completed=x['completed'],preflight=x['metadata'].get('preflight.json',{}).get('status')) for sid,x in status['states'].items()})),flush=True)
if __name__=='__main__':
 action=sys.argv[1]
 if action=='stage':stage()
 elif action=='validate':validate()
 elif action=='install':install()
 elif action=='release':release()
 elif action=='collect':collect()
 else:raise ValueError('No sbatch/restart action; unsupported '+action)
