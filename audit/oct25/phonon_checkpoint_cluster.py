"""Stage and collect the separately named CPU-only hcp Ti checkpoint comparison."""
from pathlib import Path
import sys,json,hashlib,shlex,time,base64
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'pbe_checkpoint_phonons'
REMOTE='/home6/u3129/tial-phonons/pbe-checkpoints-20261025-01'
sys.path.insert(0,str(ROOT/'testset/dft'));import cluster_test50 as c
sha=lambda b:hashlib.sha256(b).hexdigest()
save=lambda n,v:(OUT/n).write_text(json.dumps(v,indent=2),encoding='utf-8')
OUT.mkdir(exist_ok=True);action=sys.argv[1]
if action=='stage':
    assert not (OUT/'staging-intent.json').exists()
    source=Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project')
    old=source/'results-scripts-logs/models_check_wsl2/results/phonons/dpa4/neo/finetune/alpha_Ti/pbe'
    cif=(old/'relaxed_structure.cif').read_bytes()
    prev=json.loads((HERE/'pbe_checkpoint_sweep/remote-input.json').read_text())
    cfg={'id':'pbe-checkpoints-20261025-01','checkpoints':['5000','20000','23400'],
      'models':{k:prev['models'][k] for k in ['5000','20000','23400']},'common_source':str(old/'relaxed_structure.cif'),'common_sha256':sha(cif),
      'relaxation':'Independent a and c at zero stress, hcp symmetry retained; all six stress components <0.005 GPa.',
      'cases':'Common archived 5k geometry and each native equilibrium geometry; 5^3 delta .01 A; controls 5^3 delta .005 A and 7^3 delta .01 A.',
      'mass_amu':47.867,'path':'GKMGA, 200 common reduced q points','resources':'CPU only, Apollo, 3 tasks x 4 cores, 16 GiB, 30 min each'}
    job=(ROOT/'testset/monitor/alignment/train-offsets-v1/training_job.sbatch').read_text().replace('python predict_training.py','python phonon_checkpoint_remote.py')
    files={'protocol.json':json.dumps(cfg,indent=2).encode(),'common.cif':cif,'job.sbatch':job.encode(),
       'phonon_checkpoint_remote.py':(HERE/'phonon_checkpoint_remote.py').read_bytes()}
    hashes={n:sha(b) for n,b in files.items()};save('staging-intent.json',dict(remote=REMOTE,files=hashes,unix=time.time()))
    c.cluster.run('umask 077; mkdir '+shlex.quote(REMOTE))
    for n,b in files.items():c.upload(b,REMOTE+'/'+n);(OUT/n).write_bytes(b)
    for n in ['phonon_bands.dat','phonon_dos.dat','metrics.json','command.txt','relaxation_log.dat']:(OUT/('archived-'+n)).write_bytes((old/n).read_bytes())
    exp=source/'MD-results/phonons_DPA4-neo/finetune/alpha_Ti/pbe/phonon_bands_data_alpha_Ti_dpa4-neo-finetune_pbe.csv'
    (OUT/'reference-data.csv').write_bytes(exp.read_bytes())
    code='import pathlib,hashlib; p=pathlib.Path('+repr(REMOTE)+'); h='+repr(hashes)+'; assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==v for n,v in h.items()); print("staging verified")'
    print(c.cluster.run('python3 -c '+shlex.quote(code)))
    save('staging-complete.json',dict(files=hashes,unix=time.time()))
    args=['sbatch','--parsable','--partition=apollo','--account=imet-2026-2','--qos=normal','--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=16G','--time=00:30:00','--array=0-2%3','--no-requeue','--job-name=pbe-ti-phonons','--chdir='+REMOTE,'--output='+REMOTE+'/run-%A_%a.out',REMOTE+'/job.sbatch']
    save('submission-intent.json',dict(command=args,unix=time.time()))
    jid=c.cluster.run(shlex.join(args)).strip().split(';')[0];assert jid.isdigit(),jid
    save('submission.json',dict(job_id=jid,remote=REMOTE,unix=time.time()));print(jid)
else:
    jid=json.loads((OUT/'submission.json').read_text())['job_id']
    acct=c.cluster.run('sacct -j '+jid+' --format=JobID,State,ExitCode,Elapsed,AllocCPUS,CPUTimeRAW,NodeList -Pn')
    (OUT/'accounting.txt').write_text(acct);print(acct)
    code='import pathlib,json; p=pathlib.Path('+repr(REMOTE)+'); print(json.dumps({str(f.relative_to(p)):f.read_text(errors="replace")[-4000:] for f in list(p.glob("run-*.out"))+list(p.glob("*/progress.json"))}))'
    data=json.loads(c.cluster.run('python3 -c '+shlex.quote(code)));save('latest-progress.json',data)
    print(json.dumps({k:v.splitlines()[-1:] for k,v in data.items() if k.startswith('run-')},indent=2))
    if action=='collect':
        records={r.split('|')[0]:r.split('|') for r in acct.splitlines()}
        if not all(records.get(jid+'_'+str(i),['',''])[1:3]==['COMPLETED','0:0'] for i in range(3)):
            print('Collection deferred: not all array tasks completed successfully.');sys.exit(0)
        manifest={}
        for key in ['5000','20000','23400']:
            (OUT/key).mkdir(exist_ok=True)
            # Preserve raw displacement caches, force constants and exact geometries.
            code='import pathlib,json,base64,hashlib; p=pathlib.Path('+repr(REMOTE+'/'+key)+'); print(json.dumps({str(f.relative_to(p)):{"sha256":hashlib.sha256(f.read_bytes()).hexdigest(),"base64":base64.b64encode(f.read_bytes()).decode()} for f in p.rglob("*") if f.is_file()}))'
            files=json.loads(c.cluster.run('python3 -c '+shlex.quote(code)))
            for n,d in files.items():
                b=base64.b64decode(d['base64']);assert sha(b)==d['sha256'];dest=OUT/key/n;dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists():assert dest.read_bytes()==b
                else:dest.write_bytes(b)
                manifest[key+'/'+n]=d['sha256']
            print('Collected',key,len(files))
        save('collection.json',dict(job_id=jid,files=manifest,unix=time.time()))
