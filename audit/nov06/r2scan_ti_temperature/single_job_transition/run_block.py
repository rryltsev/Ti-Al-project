"""Single-temperature child of the serial allocation; never submits jobs."""
from pathlib import Path
import os,sys,time,subprocess,json,traceback
PARENT=Path(__file__).resolve().parent
sid=sys.argv[1];target=int(sys.argv[2]);ROOT=PARENT/sid
sys.path.insert(0,str(ROOT));import md_core as md
assert md.ROOT==ROOT and os.environ['SLURM_JOB_ID']=='20988268' and target>=5 and target%5==0
START=float(os.environ['SERIAL_ALLOCATION_START_UNIX'])
def main():
 s=md.CFG['states'][0];assert s['id']==sid
 pre=md.preflight();md.prepare(s)
 for i in range(len(md.completed(s))+1,target+1):
  margin=max(2400,2*max(pre['seconds_per_step'])*46000)
  if time.time()-START>20*3600-margin:
   md.status('requeue_requested',reason='completed-replica allocation boundary',composition=sid);return 75
  md.replica(s,i)
 assert len(md.completed(s))==target
 out=ROOT/f'analysis/{sid}/batch_{target:04d}/summary.json'
 if out.exists():return 0
 reserve=max(2400,45*(md.CFG['legacy']['n_replicas']+target))
 if time.time()-START>20*3600-reserve:
  md.status('requeue_requested',reason='CPU analysis time reserve',composition=sid,n_replicas=target);return 75
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',DEVICE='cpu',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
 out.parent.mkdir(parents=True,exist_ok=True);md.status('analysis',composition=sid,n_replicas=target)
 with (out.parent/'analysis.log').open('a') as log:
  subprocess.run([sys.executable,'-B','-u',str(ROOT/'analyze_point.py'),sid,str(target)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 assert out.exists();return 0
if __name__=='__main__':
 try:sys.exit(main())
 except Exception:
  md.save(ROOT/'failure.json',dict(traceback=traceback.format_exc(),unix=time.time(),job_id=os.environ.get('SLURM_JOB_ID')));raise
