"""Exclusive staging/submission/collection for fixed DFT geometry phonons."""
from pathlib import Path
import sys,json,hashlib,shlex,time,base64
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'pbe_phonons_dft_geometry'
REMOTE='/home6/u3129/tial-phonons/pbe-dft-geometry-20261025-01'
sys.path.insert(0,str(ROOT/'testset/dft'));import cluster_test50 as c
sha=lambda b:hashlib.sha256(b).hexdigest();save=lambda n,d:(OUT/n).write_text(json.dumps(d,indent=2),encoding='utf-8')
OUT.mkdir(exist_ok=True);action=sys.argv[1]
if action=='stage':
    assert not (OUT/'staging-intent.json').exists()
    source=Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project\TiAl-papers\phonons\DFT_Ti_Ti3Al\Ti_hcp')
    pos=(source/'POSCAR').read_bytes();old=json.loads((HERE/'pbe_checkpoint_phonons/protocol.json').read_text())
    cfg={'id':'pbe-dft-geometry-20261025-01','checkpoints':old['checkpoints'],'models':old['models'],
       'geometry_source':str(source/'POSCAR'),'geometry_sha256':sha(pos),'relaxation':False,
       'supercell':[9,9,9],'delta_A':.01,'mass_amu':47.867,
       'postprocessing_meshes':[[40,40,30],[50,50,50]],'sigma_THz':.05,
       'note':'Exact archived DFT cell and fractional positions, no relaxation; 40x40x30 matches the documented DFT mesh dimensions; raw mesh.yaml is unavailable.'}
    job=(HERE/'pbe_checkpoint_phonons/job.sbatch').read_text().replace('phonon_checkpoint_remote.py','dft_geometry_phonons_remote.py')
    files={'protocol.json':json.dumps(cfg,indent=2).encode(),'POSCAR-DFT':pos,'job.sbatch':job.encode(),'dft_geometry_phonons_remote.py':(HERE/'dft_geometry_phonons_remote.py').read_bytes()}
    h={n:sha(b) for n,b in files.items()};save('staging-intent.json',dict(remote=REMOTE,files=h,unix=time.time()))
    c.cluster.run('umask 077; mkdir '+shlex.quote(REMOTE))
    for n,b in files.items():c.upload(b,REMOTE+'/'+n);(OUT/n).write_bytes(b)
    refs={}
    for n in ['total_dos.dat','phonopy.yaml','mesh.conf','make_total_dos_from_mesh.py','force_run_INCAR','force_run_KPOINTS']:
        b=(source/n).read_bytes();(OUT/('DFT-'+n)).write_bytes(b);refs[n]={'source':str(source/n),'sha256':sha(b)}
    save('reference-manifest.json',refs)
    code='import pathlib,hashlib; p=pathlib.Path('+repr(REMOTE)+'); h='+repr(h)+'; assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==v for n,v in h.items()); print("verified")'
    print(c.cluster.run('python3 -c '+shlex.quote(code)));save('staging-complete.json',dict(files=h,unix=time.time()))
    args=['sbatch','--parsable','--partition=apollo','--account=imet-2026-2','--qos=normal','--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=16G','--time=00:30:00','--array=0-2%3','--no-requeue','--job-name=pbe-dftgeom','--chdir='+REMOTE,'--output='+REMOTE+'/run-%A_%a.out',REMOTE+'/job.sbatch']
    save('submission-intent.json',dict(command=args,unix=time.time()))
    jid=c.cluster.run(shlex.join(args)).strip().split(';')[0];assert jid.isdigit(),jid
    save('submission.json',dict(job_id=jid,remote=REMOTE,unix=time.time()));print(jid)
else:
    jid=json.loads((OUT/'submission.json').read_text())['job_id']
    acct=c.cluster.run('sacct -j '+jid+' --format=JobID,State,ExitCode,Elapsed,AllocCPUS,CPUTimeRAW,NodeList -Pn')
    (OUT/'accounting.txt').write_text(acct);print(acct)
    code='import pathlib,json; p=pathlib.Path('+repr(REMOTE)+'); print(json.dumps({str(f.relative_to(p)):f.read_text(errors="replace")[-6000:] for f in p.glob("run-*.out")}))'
    logs=json.loads(c.cluster.run('python3 -c '+shlex.quote(code)));save('logs.json',logs);print(json.dumps({k:v.splitlines()[-1:] for k,v in logs.items()},indent=2))
    if action=='collect':
        records={r.split('|')[0]:r.split('|') for r in acct.splitlines()}
        if not all(records.get(jid+'_'+str(i),['',''])[1:3]==['COMPLETED','0:0'] for i in range(3)):
            print('Collection deferred until all tasks complete successfully.');sys.exit(0)
        manifest={}
        for key in ['5000','20000','23400']:
            code='import pathlib,json,hashlib,base64; p=pathlib.Path('+repr(REMOTE+'/'+key)+'); print(json.dumps({str(f.relative_to(p)):{"sha256":hashlib.sha256(f.read_bytes()).hexdigest(),"base64":base64.b64encode(f.read_bytes()).decode()} for f in p.rglob("*") if f.is_file()}))'
            files=json.loads(c.cluster.run('python3 -c '+shlex.quote(code)))
            for n,d in files.items():
                b=base64.b64decode(d['base64']);assert sha(b)==d['sha256'];dest=OUT/key/n;dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists():assert dest.read_bytes()==b
                else:dest.write_bytes(b)
                manifest[key+'/'+n]=d['sha256']
            print('Collected',key,len(files),flush=True)
        save('collection.json',dict(job_id=jid,files=manifest,unix=time.time()))
