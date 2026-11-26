"""Stage and collect the authorized all-functional CPU checkpoint audit."""
from pathlib import Path
import sys,json,hashlib,shlex,time,base64,tarfile,io
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'all_xc_audit'
sys.path.insert(0,str(ROOT/'testset/dft'));import cluster_test50 as c
read=lambda p:json.loads(Path(p).read_text())
save=lambda n,v:(OUT/n).write_text(json.dumps(v,indent=2))
sha=lambda b:hashlib.sha256(b).hexdigest()
cfg=read(OUT/'protocol.json');REMOTE=cfg['remote'];tasks=read(OUT/'tasks.json');action=sys.argv[1]
if action=='stage':
    assert not (OUT/'staging-intent.json').exists(),'Reconcile previous staging first'
    files={n:(OUT/n).read_bytes() for n in ['protocol.json','tasks.json','POSCAR-DFT']}
    for xc in cfg['datasets']:
        for n in [cfg['datasets'][xc]['geometry_file'],cfg['liquids'][xc]['geometry_file']]:files[n]=(OUT/n).read_bytes()
    for key,m in cfg['models'].items():
        if m['functional']!='PBE':
            b=Path(m['local_path']).read_bytes();assert sha(b)==m['sha256'];files['models/'+key+'.pt']=b
    files['all_xc_worker.py']=(HERE/'all_xc_worker.py').read_bytes()
    files['job.sbatch']=(HERE/'pbe_phonons_dft_geometry/job.sbatch').read_text().replace('python dft_geometry_phonons_remote.py','python all_xc_worker.py').encode()
    hashes={n:sha(b) for n,b in files.items()};save('staging-intent.json',{'remote':REMOTE,'files':hashes,'unix':time.time()})
    c.cluster.run('umask 077; mkdir '+shlex.quote(REMOTE)+'; mkdir '+shlex.quote(REMOTE+'/models'))
    for n,b in files.items():c.upload(b,REMOTE+'/'+n);print('Uploaded',n,len(b),flush=True)
    code='import pathlib,hashlib; p=pathlib.Path('+repr(REMOTE)+'); h='+repr(hashes)+'; assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==v for n,v in h.items()); print("verified")'
    print(c.cluster.run('python3 -c '+shlex.quote(code)),flush=True);save('staging-complete.json',{'files':hashes,'unix':time.time()})
    args=['sbatch','--parsable','--partition=apollo','--account=imet-2026-2','--qos=normal','--nodes=1','--ntasks=1',
          '--cpus-per-task=4','--mem=12G','--time=00:30:00','--array=0-'+str(len(tasks)-1)+'%8','--no-requeue',
          '--job-name=all-xc-audit','--chdir='+REMOTE,'--output='+REMOTE+'/run-%A_%a.out',REMOTE+'/job.sbatch']
    save('submission-intent.json',{'command':args,'unix':time.time()})
    jid=c.cluster.run(shlex.join(args)).strip().split(';')[0];assert jid.isdigit(),jid
    save('submission.json',{'job_id':jid,'remote':REMOTE,'unix':time.time()});print('Submitted',jid,flush=True)
elif action=='scale16':
    jid=read(OUT/'submission.json')['job_id']
    args=['scontrol','update','JobId='+jid,'ArrayTaskThrottle=16']
    save('scale16-intent.json',{'command':args,'reason':'Complete the authorized audit sooner with unchanged task count; at most 64 CPU cores, no GPUs.','unix':time.time()})
    print(c.cluster.run(shlex.join(args)))
    state=c.cluster.run('scontrol show job '+jid+' -o')
    (OUT/'scale16-verification.txt').write_text(state)
    print('Audit array concurrency updated to 16 tasks x 4 CPU cores')
elif action=='verify-scale':
    jid=read(OUT/'submission.json')['job_id']
    state=c.cluster.run('scontrol show job '+jid+' -o')
    (OUT/'scale16-verification.txt').write_text(state)
    print('\n'.join(' '.join(w for w in line.split() if w.startswith(('JobId=','ArrayTaskId=','ArrayTaskThrottle=','JobState='))) for line in state.splitlines()))
else:
    jid=read(OUT/'submission.json')['job_id']
    acct=c.cluster.run('sacct -j '+jid+' --format=JobID,State,ExitCode,Elapsed,AllocCPUS,CPUTimeRAW,NodeList -Pn')
    (OUT/'accounting.txt').write_text(acct)
    records={r.split('|')[0]:r.split('|') for r in acct.splitlines()}
    task_states=[records.get(jid+'_'+str(t['id']),['','PENDING_OR_NOT_MATERIALIZED'])[1] for t in tasks]
    states={s:task_states.count(s) for s in sorted(set(task_states))}
    print(json.dumps({'job':jid,'states':states}),flush=True)
    code='import pathlib,json; p=pathlib.Path('+repr(REMOTE)+'); print(json.dumps({f.name:f.read_text(errors="replace")[-4000:] for f in p.glob("run-*.out")}))'
    logs=json.loads(c.cluster.run('python3 -c '+shlex.quote(code)));save('logs.json',logs)
    failed=[t for t in tasks if records.get(jid+'_'+str(t['id']),['',''])[1] in ['FAILED','TIMEOUT','OUT_OF_MEMORY']]
    for t in failed:print('FAILED',t,logs.get('run-'+jid+'_'+str(t['id'])+'.out','')[-2500:],flush=True)
    if action=='collect':
        ids=[t['id'] for t in tasks if records.get(jid+'_'+str(t['id']),['',''])[1:3]==['COMPLETED','0:0'] and not (OUT/'results'/str(t['id'])/'summary.json').exists()]
        # Bounded batches reduce SSH overhead while keeping each transfer modest.
        for start in range(0,len(ids),8):
            batch=ids[start:start+8]
            code=('import pathlib,json,base64,hashlib,tarfile,io; p=pathlib.Path('+repr(REMOTE+'/results')+'); files=[];\n'
                +'for i in '+repr(batch)+':\n files.extend(f for f in (p/str(i)).rglob("*") if f.is_file() and "forces" not in f.parts)\n'
                +'b=io.BytesIO()\nwith tarfile.open(fileobj=b,mode="w:gz") as tf:\n'
                +' for f in files: tf.add(f,arcname=str(f.relative_to(p)),recursive=False)\n'
                +'data=b.getvalue(); print(json.dumps({"sha256":hashlib.sha256(data).hexdigest(),"data":base64.b64encode(data).decode(),"files":{str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}))')
            payload=json.loads(c.cluster.run('python3 -c '+shlex.quote(code)));b=base64.b64decode(payload['data']);assert sha(b)==payload['sha256']
            with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz') as tf:
                for member in tf.getmembers():
                    assert member.isfile();dest=OUT/'results'/member.name;assert OUT.resolve() in dest.resolve().parents
                    data=tf.extractfile(member).read();assert sha(data)==payload['files'][member.name]
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    if dest.exists():assert dest.read_bytes()==data
                    else:dest.write_bytes(data)
            save('collection-'+str(batch[0])+'.json',{'ids':batch,'files':payload['files'],'unix':time.time()});print('Collected',batch,flush=True)
