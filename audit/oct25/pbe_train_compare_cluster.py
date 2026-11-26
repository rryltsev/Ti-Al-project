"""Stage/submit once or collect one small energy-inference array; no training/DFT."""
from pathlib import Path
import sys, json, hashlib, shlex, time, base64
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]; MON=ROOT/'testset/monitor'
OUT=HERE/'pbe_train_comparison'; OUT.mkdir(exist_ok=True)
REMOTE='/home6/u3129/tial-testsets/pbe-train-diagnostic-20261025-01'
sys.path.insert(0,str(ROOT/'testset/dft'))
import cluster_test50 as c
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
save=lambda n,v:(OUT/n).write_text(json.dumps(v,indent=2),encoding='utf-8')
action=sys.argv[1]
if action=='submit':
    assert not (OUT/'staging-intent.json').exists(), 'Reconcile existing intent before any retry'
    cal=MON/'alignment/train-offsets-v1'; audit=read(cal/'input-audit.json')
    assert sha(cal/'training-labels.json')==audit['labels_sha256']
    labels=read(cal/'training-labels.json');ids=[r['geometry_id'] for r in labels if r['functional']=='PBE']
    assert len(ids)==850 and len(set(ids))==850
    model=read(MON/'model_registry.json')['models']['PBE']
    cfg={'model_path':model['path'],'model_sha256':model['sha256'],
         'geometries_path':read(cal/'submission.json')['remote']+'/training-geometries.npz',
         'geometries_sha256':audit['geometries_sha256'],'geometry_ids':ids,
         'purpose':'Unchanged article PBE inference on all 850 original training frames; no fitting or energy shifts.'}
    job=(cal/'training_job.sbatch').read_text().replace('python predict_training.py','python pbe_train_compare_remote.py')
    files={'inference-input.json':json.dumps(cfg,indent=2).encode(),
           'pbe_train_compare_remote.py':(HERE/'pbe_train_compare_remote.py').read_bytes(),
           'job.sbatch':job.encode()}
    hashes={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
    save('staging-intent.json',{'remote':REMOTE,'hashes':hashes,'unix':time.time()})
    c.cluster.run('umask 077; mkdir '+shlex.quote(REMOTE))
    for n,b in files.items():
        c.upload(b,REMOTE+'/'+n);(OUT/n).write_bytes(b)
    code='import pathlib,hashlib,json; p=pathlib.Path('+repr(REMOTE)+'); h='+repr(hashes)+'; assert all(hashlib.sha256((p/k).read_bytes()).hexdigest()==v for k,v in h.items()); print("verified")'
    print(c.cluster.run('python3 -c '+shlex.quote(code)))
    args=['sbatch','--parsable','--partition=apollo','--account=imet-2026-2','--qos=normal',
          '--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=16G','--time=00:20:00',
          '--array=0-7%4','--no-requeue','--job-name=pbe-train-diag','--chdir='+REMOTE,
          '--output='+REMOTE+'/inference-%A_%a.out',REMOTE+'/job.sbatch']
    save('submission-intent.json',{'command':args,'unix':time.time()})
    jid=c.cluster.run(shlex.join(args)).strip().split(';')[0]
    assert jid.isdigit(),jid
    save('submission.json',{'job_id':jid,'remote':REMOTE,'unix':time.time()}); print(jid)
else:
    jid=read(OUT/'submission.json')['job_id']
    accounting=c.cluster.run('sacct -j '+jid+' --format=JobID,State,ExitCode,Elapsed,AllocCPUS,CPUTimeRAW,NodeList -Pn')
    (OUT/'accounting.txt').write_text(accounting);print(accounting)
    if action=='retry':
        assert not (OUT/'retry-submission-intent.json').exists()
        records={r.split('|')[0]:r.split('|') for r in accounting.splitlines()}
        failed=[i for i in range(8) if records.get(jid+'_'+str(i),['',''])[1]=='FAILED']
        assert failed==[3,4,5,6,7],failed
        assert all(records[jid+'_'+str(i)][2]=='4:0' and records[jid+'_'+str(i)][-1]=='apollo11' for i in failed)
        assert records[jid+'_1'][1:3]==['COMPLETED','0:0'] and records[jid+'_1'][-1]=='apollo31'
        code='import pathlib; p=pathlib.Path('+repr(REMOTE)+'); assert all(not (p/("result-"+str(i)+".json")).exists() for i in '+repr(failed)+'); print("failed tasks have no result files")'
        print(c.cluster.run('python3 -c '+shlex.quote(code)))
        args=['sbatch','--parsable','--partition=apollo','--account=imet-2026-2','--qos=normal',
          '--nodes=1','--nodelist=apollo31','--ntasks=1','--cpus-per-task=4','--mem=16G','--time=00:20:00',
          '--array=3-7%4','--no-requeue','--job-name=pbe-train-retry','--chdir='+REMOTE,
          '--output='+REMOTE+'/inference-%A_%a.out',REMOTE+'/job.sbatch']
        save('retry-submission-intent.json',{'command':args,'unix':time.time(),'reason':'CPU ISA incompatibility of frozen AOT export on apollo11; unchanged inputs on validated apollo31','failed_tasks':failed})
        retryid=c.cluster.run(shlex.join(args)).strip().split(';')[0];assert retryid.isdigit()
        save('retry-submission.json',{'job_id':retryid,'remote':REMOTE,'tasks':failed,'unix':time.time()});print(retryid)
    if (OUT/'retry-submission.json').exists():
        retry=read(OUT/'retry-submission.json')
        retryacct=c.cluster.run('sacct -j '+retry['job_id']+' --format=JobID,State,ExitCode,Elapsed,AllocCPUS,CPUTimeRAW,NodeList -Pn')
        (OUT/'retry-accounting.txt').write_text(retryacct);print(retryacct)
    if action=='logs':
        code='import pathlib,json; p=pathlib.Path('+repr(REMOTE)+'); print(json.dumps({f.name:f.read_text(errors="replace")[-7000:] for f in p.glob("inference-*.out")}))'
        logs=json.loads(c.cluster.run('python3 -c '+shlex.quote(code)))
        save('logs.json',logs)
        print(json.dumps(logs,indent=2))
    if action=='collect':
        records={r.split('|')[0]:r.split('|') for r in accounting.splitlines()}
        expected={i:jid+'_'+str(i) for i in range(8)}
        if (OUT/'retry-submission.json').exists():
            records.update({r.split('|')[0]:r.split('|') for r in retryacct.splitlines()})
            expected.update({i:retry['job_id']+'_'+str(i) for i in retry['tasks']})
        assert all(records.get(expected[i],['',''])[1:3]==['COMPLETED','0:0'] for i in range(8))
        names=[f'result-{i}.{ext}' for i in range(8) for ext in ['json','sha256']]
        code='import pathlib,base64,json,hashlib; p=pathlib.Path('+repr(REMOTE)+'); names='+repr(names)+'; print(json.dumps({n:{"sha256":hashlib.sha256((p/n).read_bytes()).hexdigest(),"base64":base64.b64encode((p/n).read_bytes()).decode()} for n in names}))'
        files=json.loads(c.cluster.run('python3 -c '+shlex.quote(code)))
        for n,d in files.items():
            b=base64.b64decode(d['base64']);assert hashlib.sha256(b).hexdigest()==d['sha256']
            p=OUT/n
            if p.exists(): assert p.read_bytes()==b
            else: p.write_bytes(b)
        save('collection.json',{'job_id':jid,'files':{n:d['sha256'] for n,d in files.items()},'unix':time.time()})
        print('Collected and checksummed all 8 chunks.')
