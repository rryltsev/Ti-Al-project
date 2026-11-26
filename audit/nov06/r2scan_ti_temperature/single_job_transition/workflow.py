"""One physical A100, one inherited job, round-robin immutable five-replica blocks."""
from pathlib import Path
import os,sys,json,time,subprocess,hashlib,fcntl,traceback
ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'protocol.json').read_text());START=float(os.environ.get('SERIAL_ALLOCATION_START_UNIX',time.time()))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,data):
 path=Path(path);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
def completed(sid):
 cfg=json.loads((ROOT/sid/'protocol.json').read_text());out=[]
 for p in sorted((ROOT/sid/'states'/sid/'replicas').glob('replica_*/summary.json')):
  r=json.loads(p.read_text());assert r['replica']==len(out)+1 and r['protocol_sha256']==sha(ROOT/sid/'protocol.json') and r['checkpoint_sha256']==CFG['checkpoint']['sha256']
  assert r['temperature_K']==cfg['states'][0]['temperature_K'] and sha(ROOT/sid/r['stress_file'])==r['stress_sha256'];out.append(r)
 return len(out)
def choose(plan,counts):
 # Finish an interrupted block before rotating; already completed/analyzed blocks are skipped.
 for sid in plan['order']:
  n=counts[sid]
  if n%5:return sid,min((n//5+1)*5,plan['targets_new_replicas'][sid])
 pending=[]
 for sid in plan['order']:
  n=counts[sid];target=plan['targets_new_replicas'][sid]
  if n and not (ROOT/sid/f'analysis/{sid}/batch_{n:04d}/summary.json').exists():return sid,n
  if n<target:pending.append(sid)
 if not pending:return None
 sid=min(pending,key=lambda s:(counts[s]//5,plan['order'].index(s)))
 return sid,min(counts[sid]+5,plan['targets_new_replicas'][sid])
def main():
 assert os.environ['SLURM_JOB_ID']==CFG['job_id']
 lock=(ROOT/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 agreement=json.loads((ROOT/'data_confirmation.json').read_text());assert agreement['status']=='confirmed' and agreement['inventory_sha256']==sha(ROOT/'inventory.json')==CFG['inventory_sha256']
 for name,h in json.loads((ROOT/'staging-manifest.json').read_text()).items():assert sha(ROOT/name)==h,name
 for key in ['checkpoint','frozen_model']:assert sha(CFG[key]['path'])==CFG[key]['sha256']
 plan=json.loads((ROOT/'plan.json').read_text());assert plan['batch_size']==5 and not plan['automatic_extension']
 assert all(v>=0 and v%5==0 for v in plan['targets_new_replicas'].values())
 while True:
  counts={sid:completed(sid) for sid in plan['order']};next_block=choose(plan,counts)
  if next_block is None:
   final=dict(status='refinement_cycle_complete_review_required',completed=counts,legacy={sid:e['legacy_replicas'] for sid,e in CFG['states'].items()},plan_version=plan['version'],job_id=CFG['job_id'],unix=time.time(),automatic_extension=False)
   save(ROOT/'final.json',final);save(ROOT/'progress.json',dict(phase='cycle_completed',**final));return 0
  sid,target=next_block
  if time.time()-START>20*3600-2400:
   save(ROOT/'progress.json',dict(phase='requeue_requested',completed=counts,reason='allocation time reserve',job_id=CFG['job_id'],unix=time.time()));return 75
  save(ROOT/'progress.json',dict(phase='block_running',composition=sid,target_new_replicas=target,completed=counts,job_id=CFG['job_id'],unix=time.time()))
  env=dict(os.environ,SERIAL_ALLOCATION_START_UNIX=str(START))
  with (ROOT/f'{sid}-block-{target:04d}.log').open('a') as log:
   r=subprocess.run([sys.executable,'-B','-u',str(ROOT/'run_block.py'),sid,str(target)],cwd=ROOT/sid,env=env,stdout=log,stderr=subprocess.STDOUT)
  if r.returncode==75:return 75
  if r.returncode:raise RuntimeError('Block failed '+sid+';see private logs; completed replicas retained')
  summary=json.loads((ROOT/sid/f'analysis/{sid}/batch_{target:04d}/summary.json').read_text());assert summary['n_replicas']==target
  save(ROOT/'progress.json',dict(phase='block_analyzed',composition=sid,new_replicas=target,total_replica_clusters=summary['total_replica_clusters'],job_id=CFG['job_id'],unix=time.time()))
if __name__=='__main__':
 try:sys.exit(main())
 except Exception:
  save(ROOT/'failure.json',dict(unix=time.time(),job_id=os.environ.get('SLURM_JOB_ID'),traceback=traceback.format_exc()));raise
