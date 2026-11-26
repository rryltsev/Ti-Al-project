"""Exclusive staging, bounded CPU Slurm array, and verified collection."""
from pathlib import Path
import sys,json,hashlib,shlex,time,base64
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];BASEOUT=HERE/'pbe_property_screen';OUT=BASEOUT
REMOTE='/home6/u3129/tial-testsets/pbe-property-screen-20261025-01'
retry=len(sys.argv)>2 and sys.argv[2]=='retry15k'
if retry:OUT=BASEOUT/'retry15k';OUT.mkdir(exist_ok=True);REMOTE+='-retry15k'
sys.path.insert(0,str(ROOT/'testset/dft'));import cluster_test50 as c
sha=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:json.loads(p.read_text())
def save(n,d):(OUT/n).write_text(json.dumps(d,indent=2))
action=sys.argv[1]
if action=='stage':
    assert not (OUT/'staging-intent.json').exists(),'Reconcile previous submission first'
    cfg=read(BASEOUT/'protocol.json')
    job=(HERE/'pbe_phonons_dft_geometry/job.sbatch').read_text().replace('python dft_geometry_phonons_remote.py','python property_screen_remote.py')
    files={n:(BASEOUT/n).read_bytes() for n in ['protocol.json','rows.json','geometries.npz']}
    files.update({'job.sbatch':job.encode(),'property_screen_remote.py':(HERE/'property_screen_remote.py').read_bytes()})
    hashes={n:sha(b) for n,b in files.items()}
    save('staging-intent.json',{'remote':REMOTE,'files':hashes,'unix':time.time()})
    c.cluster.run('umask 077; mkdir '+shlex.quote(REMOTE))
    for n,b in files.items():c.upload(b,REMOTE+'/'+n)
    code=('import pathlib,hashlib; p=pathlib.Path('+repr(REMOTE)+'); h='+repr(hashes)
          +'; assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==v for n,v in h.items()); '
          +'models='+repr(cfg['models'])+'; assert all(hashlib.sha256(pathlib.Path(m["path"]).read_bytes()).hexdigest()==m["sha256"] for m in models.values()); print("verified")')
    print(c.cluster.run('python3 -c '+shlex.quote(code)),flush=True)
    save('staging-complete.json',{'verified':True,'files':hashes,'unix':time.time()})
    args=['sbatch','--parsable','--partition=apollo','--account=imet-2026-2','--qos=normal','--nodes=1','--ntasks=1',
          '--cpus-per-task=4','--mem=12G','--time=00:30:00','--array='+('1-1' if retry else '0-1%2'),'--no-requeue',
          '--job-name=pbe-properties','--chdir='+REMOTE,'--output='+REMOTE+'/run-%A_%a.out',REMOTE+'/job.sbatch']
    save('submission-intent.json',{'command':args,'unix':time.time()})
    jid=c.cluster.run(shlex.join(args)).strip().split(';')[0];assert jid.isdigit(),jid
    save('submission.json',{'job_id':jid,'remote':REMOTE,'unix':time.time()});print('Submitted',jid)
elif action=='resources':
    print(c.cluster.run("sinfo -p apollo -N -h -o '%N %t %C %m'"))
else:
    jid=read(OUT/'submission.json')['job_id']
    acct=c.cluster.run('sacct -j '+jid+' --format=JobID,State,ExitCode,Elapsed,AllocCPUS,CPUTimeRAW,NodeList -Pn')
    (OUT/'accounting.txt').write_text(acct);print(acct,flush=True)
    code='import pathlib,json; p=pathlib.Path('+repr(REMOTE)+'); print(json.dumps({f.name:f.read_text(errors="replace")[-6000:] for f in p.glob("run-*.out")}))'
    logs=json.loads(c.cluster.run('python3 -c '+shlex.quote(code)));save('logs.json',logs)
    print(json.dumps({k:v.splitlines()[-4:] for k,v in logs.items()},indent=2))
    if action=='collect':
        records={r.split('|')[0]:r.split('|') for r in acct.splitlines()}
        for i,key in enumerate(['5000','15000']):
            if retry and key=='5000':continue
            if records.get(jid+'_'+str(i),['',''])[1:3]!=['COMPLETED','0:0']:
                print('Collection deferred',key);continue
            code=('import pathlib,json,hashlib,base64; p=pathlib.Path('+repr(REMOTE+'/'+key)+'); print(json.dumps({str(f.relative_to(p)):'
                  +'{"sha256":hashlib.sha256(f.read_bytes()).hexdigest(),"base64":base64.b64encode(f.read_bytes()).decode()} for f in p.rglob("*") if f.is_file()}))')
            files=json.loads(c.cluster.run('python3 -c '+shlex.quote(code)));manifest={}
            for n,d in files.items():
                b=base64.b64decode(d['base64']);assert sha(b)==d['sha256']
                dest=OUT/key/n;assert OUT.resolve() in dest.resolve().parents
                dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists():assert dest.read_bytes()==b
                else:dest.write_bytes(b)
                manifest[n]=d['sha256']
            save('collection-'+key+'.json',{'job_id':jid,'files':manifest,'unix':time.time()});print('Collected',key,len(files),'files')
