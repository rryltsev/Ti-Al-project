"""Exclusive staging and bounded CPU Slurm inference; checkpoint and DFT data immutable."""
from pathlib import Path
import sys,json,hashlib,shlex,time,base64
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'pbe_checkpoint_sweep'
REMOTE='/home6/u3129/tial-testsets/pbe-checkpoint-sweep-20261025-01'
sys.path.insert(0,str(ROOT/'testset/dft'))
import cluster_test50 as c
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
save=lambda n,v:(OUT/n).write_text(json.dumps(v,indent=2),encoding='utf-8')
action=sys.argv[1]
def submit(mode):
    intent=OUT/(mode+'-submission-intent.json');assert not intent.exists(),'Reconcile prior submission'
    args=['sbatch','--parsable','--partition=apollo','--account=imet-2026-2','--qos=normal',
      '--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=12G','--time=00:30:00',
      '--array='+('0-1%2' if mode=='pilot' else '0-39%8'),'--no-requeue',
      '--job-name=pbe-sweep-'+mode,'--chdir='+REMOTE,'--output='+REMOTE+'/'+mode+'-%A_%a.out',REMOTE+'/job.sbatch',mode]
    save(intent.name,{'command':args,'unix':time.time()})
    jid=c.cluster.run(shlex.join(args)).strip().split(';')[0];assert jid.isdigit(),jid
    save(mode+'-submission.json',{'job_id':jid,'remote':REMOTE,'unix':time.time()});print(jid)
if action=='stage':
    assert not (OUT/'staging-intent.json').exists()
    m=read(OUT/'input-manifest.json');rows=read(OUT/'rows.json')
    cfg={'models':{k:{'path':REMOTE+'/checkpoint-'+k+'.pt','sha256':v['sha256']} for k,v in m['models'].items()},
         'total':m['total'],'geometry_sha256':m['geometry_sha256'],
         'pilot_indices':[r['i'] for r in rows if r['split']=='test20']+[0,100,200,300,400,500,600,700,800,850,900,950]}
    job=(ROOT/'testset/monitor/alignment/train-offsets-v1/training_job.sbatch').read_text().replace('python predict_training.py','python checkpoint_sweep_remote.py "$1"')
    files={'remote-input.json':json.dumps(cfg,indent=2).encode(),'job.sbatch':job.encode(),
           'checkpoint_sweep_remote.py':(HERE/'checkpoint_sweep_remote.py').read_bytes(),
           'geometries.npz':(OUT/'geometries.npz').read_bytes()}
    for k,v in m['models'].items():
        p=Path(v['local_path']);assert sha(p)==v['sha256'];files['checkpoint-'+k+'.pt']=p.read_bytes()
    hashes={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
    save('staging-intent.json',{'remote':REMOTE,'files':hashes,'bytes':{n:len(b) for n,b in files.items()},'unix':time.time()})
    c.cluster.run('umask 077; mkdir '+shlex.quote(REMOTE))
    for n,b in files.items():
        c.upload(b,REMOTE+'/'+n)
        if n in ['remote-input.json','job.sbatch','checkpoint_sweep_remote.py']:(OUT/n).write_bytes(b)
        print('Uploaded',n,len(b),flush=True)
    code='import pathlib,hashlib,json; p=pathlib.Path('+repr(REMOTE)+'); h='+repr(hashes)+'; assert all(hashlib.sha256((p/k).read_bytes()).hexdigest()==v for k,v in h.items()); print("all hashes verified")'
    print(c.cluster.run('python3 -c '+shlex.quote(code)));save('staging-complete.json',{'verified':True,'unix':time.time()})
    submit('pilot')
elif action=='submit-full':
    assert read(OUT/'pilot-verification.json')['status']=='passed'
    submit('full')
elif action=='resources':
    print(c.cluster.run("sinfo -p apollo -N -h -o '%N %t %C %m'"))
else:
    mode=sys.argv[2];jid=read(OUT/(mode+'-submission.json'))['job_id']
    acct=c.cluster.run('sacct -j '+jid+' --format=JobID,State,ExitCode,Elapsed,AllocCPUS,CPUTimeRAW,NodeList -Pn')
    (OUT/(mode+'-accounting.txt')).write_text(acct)
    top=[line.split('|') for line in acct.splitlines() if '.' not in line.split('|')[0]]
    print(json.dumps({'job':jid,'array_entries':[{ 'task':r[0],'state':r[1],'elapsed':r[3],'node':r[-1]} for r in top] if mode=='pilot' else {state:sum(r[1]==state for r in top) for state in set(r[1] for r in top)}}),flush=True)
    if action=='logs':
        code='import pathlib,json; p=pathlib.Path('+repr(REMOTE)+'); print(json.dumps({f.name:f.read_text(errors="replace")[-5000:] for f in p.glob('+repr(mode+'-*.out')+')}))'
        logs=json.loads(c.cluster.run('python3 -c '+shlex.quote(code)));save(mode+'-logs.json',logs);print(json.dumps(logs,indent=2))
    if action=='progress':
        running=[int(r[0].split('_')[-1]) for r in top if r[1]=='RUNNING' and r[0].split('_')[-1].isdigit()]
        code='import pathlib,json; p=pathlib.Path('+repr(REMOTE)+'); result={};\nfor i in '+repr(running)+':\n f=p/('+repr(mode+'-'+jid+'_')+'+str(i)+".out"); lines=f.read_text(errors="replace").splitlines() if f.exists() else []; values=[s for s in lines if len(s.split())==3 and s.split()[0].isdigit() and s.split()[1].isdigit()]; result[i]={"evaluated_frames":len(values),"last":values[-1] if values else lines[-1:]};\nprint(json.dumps(result))'
        data=json.loads(c.cluster.run('python3 -c '+shlex.quote(code)));save(mode+'-progress.json',data);print(json.dumps(data,indent=2))
    if action in ['collect','collect-ready']:
        records={r.split('|')[0]:r.split('|') for r in acct.splitlines()};count=2 if mode=='pilot' else 40
        done=[i for i in range(count) if records.get(jid+'_'+str(i),['',''])[1:3]==['COMPLETED','0:0']]
        if action=='collect':assert len(done)==count
        prefixes=[['pilot-5000','pilot-23400'][i] if mode=='pilot' else 'result-'+str(i) for i in done]
        collected={}
        for prefix in prefixes:
            names=[prefix+'.json',prefix+'.npz']
            if all((OUT/n).exists() for n in names):
                for n in names:collected[n]=sha(OUT/n)
                continue
            code='import pathlib,json,base64,hashlib; p=pathlib.Path('+repr(REMOTE)+'); print(json.dumps({n:{"sha256":hashlib.sha256((p/n).read_bytes()).hexdigest(),"base64":base64.b64encode((p/n).read_bytes()).decode()} for n in '+repr(names)+'}))'
            files=json.loads(c.cluster.run('python3 -c '+shlex.quote(code)))
            for n,d in files.items():
                b=base64.b64decode(d['base64']);assert hashlib.sha256(b).hexdigest()==d['sha256']
                p=OUT/n
                if p.exists():assert p.read_bytes()==b
                else:p.write_bytes(b)
                collected[n]=d['sha256']
            print('Collected',prefix,flush=True)
        save(mode+('-collection.json' if action=='collect' else '-partial-collection.json'),{'job_id':jid,'files':collected,'unix':time.time()})
