from pathlib import Path
import sys,json,hashlib,shlex,time,base64
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'pbe_checkpoint_phonons'
REMOTE='/home6/u3129/tial-phonons/pbe-checkpoints-20261025-01'
sys.path.insert(0,str(ROOT/'testset/dft'));import cluster_test50 as c
sha=lambda b:hashlib.sha256(b).hexdigest();save=lambda n,d:(OUT/n).write_text(json.dumps(d,indent=2))
action=sys.argv[1]
if action=='submit':
    assert not (OUT/'size9-submission-intent.json').exists()
    script=(HERE/'phonon_checkpoint_size9.py').read_bytes()
    job=(OUT/'job.sbatch').read_text().replace('phonon_checkpoint_remote.py','phonon_checkpoint_size9.py').encode()
    files={'phonon_checkpoint_size9.py':script,'size9.sbatch':job}
    for n,b in files.items():c.upload(b,REMOTE+'/'+n);(OUT/n).write_bytes(b)
    hashes={n:sha(b) for n,b in files.items()}
    code='import pathlib,hashlib; p=pathlib.Path('+repr(REMOTE)+'); h='+repr(hashes)+'; assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==v for n,v in h.items()); print("verified")'
    print(c.cluster.run('python3 -c '+shlex.quote(code)))
    old=json.loads((OUT/'submission.json').read_text())['job_id']
    args=['sbatch','--parsable','--partition=apollo','--account=imet-2026-2','--qos=normal','--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=16G','--time=00:30:00','--array=0-2%3','--no-requeue','--dependency=afterok:'+old,'--kill-on-invalid-dep=yes','--job-name=pbe-ti-size9','--chdir='+REMOTE,'--output='+REMOTE+'/size9-%A_%a.out',REMOTE+'/size9.sbatch']
    save('size9-submission-intent.json',dict(command=args,files=hashes,unix=time.time()))
    jid=c.cluster.run(shlex.join(args)).strip().split(';')[0];assert jid.isdigit()
    save('size9-submission.json',dict(job_id=jid,unix=time.time()));print(jid)
else:
    jid=json.loads((OUT/'size9-submission.json').read_text())['job_id']
    acct=c.cluster.run('sacct -j '+jid+' --format=JobID,State,ExitCode,Elapsed,AllocCPUS,CPUTimeRAW,NodeList -Pn')
    (OUT/'size9-accounting.txt').write_text(acct);print(acct)
    code='import pathlib,json; p=pathlib.Path('+repr(REMOTE)+'); print(json.dumps({str(f.relative_to(p)):f.read_text(errors="replace")[-3000:] for f in p.glob("size9-*.out")}))'
    logs=json.loads(c.cluster.run('python3 -c '+shlex.quote(code)));save('size9-logs.json',logs);print(json.dumps({k:v.splitlines()[-1:] for k,v in logs.items()},indent=2))
    if action=='collect':
        records={r.split('|')[0]:r.split('|') for r in acct.splitlines()}
        if not all(records.get(jid+'_'+str(i),['',''])[1:3]==['COMPLETED','0:0'] for i in range(3)):
            print('Collection deferred: scheduler has not confirmed all tasks completed successfully.');sys.exit(0)
        manifest={}
        for key in ['5000','20000','23400']:
            dirname='size9-'+key
            code='import pathlib,json,hashlib,base64; p=pathlib.Path('+repr(REMOTE+'/'+dirname)+'); print(json.dumps({str(f.relative_to(p)):{"sha256":hashlib.sha256(f.read_bytes()).hexdigest(),"base64":base64.b64encode(f.read_bytes()).decode()} for f in p.rglob("*") if f.is_file()}))'
            files=json.loads(c.cluster.run('python3 -c '+shlex.quote(code)))
            for n,d in files.items():
                b=base64.b64decode(d['base64']);assert sha(b)==d['sha256'];dest=OUT/dirname/n;dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists():assert dest.read_bytes()==b
                else:dest.write_bytes(b)
                manifest[dirname+'/'+n]=d['sha256']
            print('Collected',dirname,len(files),flush=True)
        save('size9-collection.json',dict(job_id=jid,files=manifest,unix=time.time()))
