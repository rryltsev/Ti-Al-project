"""One allocation, sequential temperature batches, resumable at every replica."""
import fcntl,json,os,subprocess,sys,time,traceback
import md_core as md
ROOT=md.ROOT;CFG=md.CFG

def analyze(s,n):
    target=ROOT/f"analysis/{s['id']}/batch_{n:04d}/summary.json"
    if target.exists():return
    # Cluster jackknife grows with replica count. Requeue before starting an
    # analysis that may outlive this allocation; immutable MD replicas remain.
    reserve=max(2400,45*(CFG['legacy']['n_replicas']+n))
    if time.monotonic()-md.START>20*3600-reserve:
        md.status('requeue_requested',reason='CPU analysis time reserve',composition=s['id'],n_replicas=n)
        raise SystemExit(75)
    target.parent.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','DEVICE':'cpu','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    md.status('analysis',temperature_K=s['temperature_K'],n_replicas=n)
    with target.parent.with_suffix('.log').open('a') as log:
        subprocess.run([sys.executable,'-u',str(ROOT/'analyze_point.py'),s['id'],str(n)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)

def latest(s):
    p=ROOT/f"analysis/latest_{s['id']}.json"
    return json.loads(p.read_text()) if p.exists() else {}

def choose():
    active=[s for s in CFG['states'] if not (latest(s).get('stop_recommended') or latest(s).get('review_required'))]
    if not active:return None
    for s in active:
        if len(md.completed(s))<CFG['scheduling']['initial_replicas_each']:return s
    return min(active,key=lambda s:len(md.completed(s))//CFG['batch_size'])

def control_stop():
    p=ROOT/'control.json'
    return p.exists() and json.loads(p.read_text()).get('stop_after_replica',False)

def main():
    lock=(ROOT/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'final.json').exists():return 0
    pre=md.preflight()
    # Analyze any complete five-replica group left by a walltime interruption.
    for s in CFG['states']:
        for n in range(5,len(md.completed(s))+1,5):analyze(s,n)
    while True:
        if control_stop():
            md.save(ROOT/'final.json',dict(status='requested_pause',completed={s['id']:len(md.completed(s)) for s in CFG['states']},unix=time.time(),job_id=os.environ['SLURM_JOB_ID']))
            return 0
        s=choose()
        if s is None:
            reports={s['id']:latest(s) for s in CFG['states']}
            state='protocol_review_required' if any(r.get('review_required') for r in reports.values()) else 'sampling_convergence_rule_met'
            md.save(ROOT/'final.json',dict(status=state,completed={s['id']:len(md.completed(s)) for s in CFG['states']},unix=time.time(),job_id=os.environ['SLURM_JOB_ID']))
            md.status('completed',reason=state);return 0
        md.prepare(s)
        n=(len(md.completed(s))//5+1)*5
        for i in range(len(md.completed(s))+1,n+1):
            if control_stop():break
            margin=max(2400,2*max(pre['seconds_per_step'])*46000)
            if time.monotonic()-md.START>20*3600-margin:
                md.status('requeue_requested',completed={s['id']:len(md.completed(s)) for s in CFG['states']});return 75
            md.replica(s,i)
        if len(md.completed(s))==n:analyze(s,n)

if __name__=='__main__':
    try:sys.exit(main())
    except Exception:
        md.save(ROOT/'failure.json',dict(traceback=traceback.format_exc(),unix=time.time(),job_id=os.environ.get('SLURM_JOB_ID')))
        raise
