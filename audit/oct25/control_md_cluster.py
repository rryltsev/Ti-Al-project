"""Stage, preflight, submit once, and inspect the eight authorized control MD runs."""
from pathlib import Path
import sys,json,shlex,time,hashlib,base64,tarfile,io
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'pbe_control_md'
sys.path.insert(0,str(ROOT/'testset/dft'));import cluster_test50 as c
read=lambda p:json.loads(Path(p).read_text());cfg=read(OUT/'protocol.json');REMOTE=cfg['remote']
save=lambda n,v:(OUT/n).write_text(json.dumps(v,indent=2))
sha=lambda b:hashlib.sha256(b).hexdigest()
action=sys.argv[1]
if action=='retry15k':
    assert not (OUT/'retry15k-submission-intent.json').exists(),'Reconcile retry before resubmitting'
    jid=read(OUT/'production-submission.json')['job_id']
    state=c.cluster.run('scontrol show job '+jid+' -o')
    assert 'JobName=pbe-md-production' in state and 'WorkDir='+REMOTE in state
    script=(HERE/'control_md_worker.py').read_text().replace("out=ROOT/'results'/str(task['id'])", "out=ROOT/'retry15k'/'results'/str(task['id'])")
    job=(HERE/'pbe_phonons_dft_geometry/job.sbatch').read_text().replace('python dft_geometry_phonons_remote.py','python control_md_worker_long.py')
    job=job.replace('export DP_TF32_INFER=0','export DP_INTRA_OP_PARALLELISM_THREADS=4 DP_INTER_OP_PARALLELISM_THREADS=1\nexport DP_TF32_INFER=0')
    c.upload(script.encode(),REMOTE+'/control_md_worker_long.py');c.upload(job.encode(),REMOTE+'/job-long.sbatch')
    save('retry15k-files.json',{'control_md_worker_long.py':sha(script.encode()),'job-long.sbatch':sha(job.encode())})
    cancel=['scancel']+[jid+'_'+str(i) for i in range(4)]
    save('retry15k-cancel-intent.json',{'command':cancel,'reason':'20h running job limits cannot be extended by user; measured node speed requires a longer reservation. Preserve early outputs and restart same MD protocol.','unix':time.time()})
    c.cluster.run(shlex.join(cancel))
    args=['sbatch','--parsable','--partition=apollo','--account=imet-2026-2','--qos=normal','--nodes=1','--ntasks=1',
      '--cpus-per-task=4','--mem=8G','--no-requeue','--job-name=pbe-md-15k-long','--chdir='+REMOTE,
      '--output='+REMOTE+'/retry15k-%A_%a.out','--time=1-15:00:00','--array=0-3%4',REMOTE+'/job-long.sbatch']
    save('retry15k-submission-intent.json',{'command':args,'unix':time.time()})
    retry=c.cluster.run(shlex.join(args)).strip().split(';')[0];assert retry.isdigit(),retry
    save('retry15k-submission.json',{'job_id':retry,'command':args,'unix':time.time(),'supersedes_tasks':[0,1,2,3],'result_subdir':'retry15k/results'})
    print('Submitted 15k restart',retry)
elif action=='extend':
    jid=read(OUT/'production-submission.json')['job_id']
    args=['scontrol','update','JobId='+jid,'TimeLimit=1-12:00:00']
    save('time-limit-update-intent.json',{'command':args,'reason':'Assigned CPU nodes slower than preflight node; preserve the same 60000-step protocol.','unix':time.time()})
    print(c.cluster.run(shlex.join(args)))
    state=c.cluster.run('scontrol show job '+jid+' -o')
    (OUT/'time-limit-verification.txt').write_text(state)
    print('Requested 36h limit for existing array; verify scheduler state.')
elif action=='resources':
    print(c.cluster.run("sinfo -p apollo -N -h -o '%N %t %C'; squeue -u u3129 -h -o '%i %j %T %M %C %R'"))
elif action=='stage':
    assert not (OUT/'staging-intent.json').exists(),'Reconcile previous staging; no blind retry'
    files={n:(OUT/n).read_bytes() for n in ['protocol.json','tasks.json','baseline_5k.json']+[s['initial_file'] for s in cfg['states']]}
    files['control_md_worker.py']=(HERE/'control_md_worker.py').read_bytes()
    job=(HERE/'pbe_phonons_dft_geometry/job.sbatch').read_text().replace('python dft_geometry_phonons_remote.py','python control_md_worker.py "$@"')
    job=job.replace('$SLURM_ARRAY_TASK_ID','${SLURM_ARRAY_TASK_ID:-preflight}')
    job=job.replace('export DP_TF32_INFER=0','export DP_INTRA_OP_PARALLELISM_THREADS=4 DP_INTER_OP_PARALLELISM_THREADS=1\nexport DP_TF32_INFER=0')
    files['job.sbatch']=job.encode();hashes={n:sha(b) for n,b in files.items()}
    save('staging-intent.json',{'remote':REMOTE,'files':hashes,'unix':time.time()})
    c.cluster.run('umask 077; mkdir '+shlex.quote(REMOTE))
    for n,b in files.items():c.upload(b,REMOTE+'/'+n)
    code='import pathlib,hashlib; p=pathlib.Path('+repr(REMOTE)+'); h='+repr(hashes)+'; assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==v for n,v in h.items()); print("verified")'
    print(c.cluster.run('python3 -c '+shlex.quote(code)))
    save('staging-complete.json',{'files':hashes,'unix':time.time()})
elif action in ['preflight','submit']:
    name='preflight' if action=='preflight' else 'production'
    assert not (OUT/(name+'-submission-intent.json')).exists(),'Reconcile existing submission first'
    if action=='submit':
        pre=read(OUT/'preflight.json');assert pre['status']=='passed' and pre['protocol_sha256']==sha((OUT/'protocol.json').read_bytes())
        assert max(x['projected_60ps_hours'] for x in pre['benchmarks'])<16,'Reassess resources before production'
    args=['sbatch','--parsable','--partition=apollo','--account=imet-2026-2','--qos=normal','--nodes=1',
      '--ntasks=1','--cpus-per-task=4','--mem=8G','--no-requeue','--job-name=pbe-md-'+name,
      '--chdir='+REMOTE,'--output='+REMOTE+'/'+name+'-%A_%a.out']
    if action=='preflight':args+=['--time=00:15:00',REMOTE+'/job.sbatch','preflight']
    else:args+=['--time=20:00:00','--array=0-7%8',REMOTE+'/job.sbatch']
    save(name+'-submission-intent.json',{'command':args,'unix':time.time()})
    jid=c.cluster.run(shlex.join(args)).strip().split(';')[0];assert jid.isdigit(),jid
    save(name+'-submission.json',{'job_id':jid,'command':args,'unix':time.time()});print('Submitted',name,jid)
elif action in ['status','collect']:
    jobs={name:read(OUT/(name+'-submission.json'))['job_id'] for name in ['preflight','production','retry15k'] if (OUT/(name+'-submission.json')).exists()}
    acct=c.cluster.run('sacct -j '+','.join(jobs.values())+' --format=JobID,State,ExitCode,Elapsed,AllocCPUS,CPUTimeRAW,NodeList -Pn')
    (OUT/'accounting.txt').write_text(acct);print(acct,flush=True)
    code=('import pathlib,json; p=pathlib.Path('+repr(REMOTE)+'); print(json.dumps({"logs":{f.name:f.read_text(errors="replace")[-4000:] for f in p.glob("*.out")},'
      +'"preflight":json.loads((p/"preflight.json").read_text()) if (p/"preflight.json").exists() else None,'
      +'"progress":{str(f.relative_to(p)):f.read_text().splitlines()[-1] for folder in [p/"results",p/"preflight",p/"retry15k/results"] for f in folder.glob("*/*-thermo.jsonl") if f.stat().st_size}}))')
    data=json.loads(c.cluster.run('python3 -c '+shlex.quote(code)));save('status.json',data)
    if data['preflight']:save('preflight.json',data['preflight'])
    print(json.dumps({'preflight':data['preflight'],'progress':data['progress'],'logs_tail':{k:v.splitlines()[-3:] for k,v in data['logs'].items()}},indent=2))
    if action=='collect':
        records={r.split('|')[0]:r.split('|') for r in acct.splitlines()}
        job_for=lambda i:jobs.get('retry15k',jobs['production']) if i<4 else jobs['production']
        ids=[i for i in range(8) if records.get(job_for(i)+'_'+str(i),['',''])[1:3]==['COMPLETED','0:0'] and not (OUT/'results'/str(i)/'summary.json').exists()]
        for i in ids:
            jid=job_for(i);subdir='/retry15k/results' if i<4 and 'retry15k' in jobs else '/results'
            code='import pathlib,tarfile,io,base64,json,hashlib; p=pathlib.Path('+repr(REMOTE+subdir)+'); b=io.BytesIO(); tf=tarfile.open(fileobj=b,mode="w:gz"); tf.add(p/'+repr(str(i))+',arcname='+repr(str(i))+'); tf.close(); data=b.getvalue(); print(json.dumps({"sha256":hashlib.sha256(data).hexdigest(),"data":base64.b64encode(data).decode()}))'
            payload=json.loads(c.cluster.run('python3 -c '+shlex.quote(code)));b=base64.b64decode(payload['data']);assert sha(b)==payload['sha256']
            with tarfile.open(fileobj=io.BytesIO(b),mode='r:gz') as tf:
                for member in tf.getmembers():
                    if member.isdir():continue
                    assert member.isfile();dest=OUT/'results'/member.name;assert OUT.resolve() in dest.resolve().parents
                    dest.parent.mkdir(parents=True,exist_ok=True);data=tf.extractfile(member).read()
                    if dest.exists():assert dest.read_bytes()==data
                    else:dest.write_bytes(data)
            save('collection-'+str(i)+'.json',{'job_id':jid,'task_id':i,'archive_sha256':payload['sha256'],'unix':time.time()})
            print('Collected',i)
