"""Immutable staging/submission and SHA-verified collection for matched DPA4."""
from pathlib import Path
import base64,hashlib,json,shlex,subprocess,sys
from datetime import datetime,timezone
import cluster as c

XC=sys.argv[2]
assert XC in ['PBE','r2SCAN']
LOCAL=c.RECORD/('DPA4-'+XC+'-matched-20261101-01')
REMOTE=c.REMOTE+'/'+LOCAL.name
DFT=c.ROOT/'results'/(XC+'-DFT-pair-inputs-20261101')
def sha(b): return hashlib.sha256(b).hexdigest()
def save(n,v): (LOCAL/n).write_text(json.dumps(v,indent=2)+'\n')
action=sys.argv[1]
if action=='submit':
    LOCAL.mkdir(exist_ok=False)
    models=json.loads((c.ROOT.parent/'audit/oct25/all_xc_audit/protocol.json').read_text())['models']
    model=models['PBE-5000' if XC=='PBE' else 'r2SCAN-23200'];assert model['article_checkpoint']
    files={key+'/POSCAR':(DFT/key/'POSCAR').read_bytes() for key in ['force000','force001','force002']}
    files['phonopy_disp.yaml']=(DFT/'phonopy_disp.yaml').read_bytes()
    files['worker.py']=(c.ROOT/'scripts/dpa4_matched_forces.py').read_bytes()
    launch=(c.ROOT.parent/'audit/oct25/pbe_phonons_dft_geometry/job.sbatch').read_text()
    launch=launch.replace('$SLURM_ARRAY_TASK_ID','$SLURM_JOB_ID').replace('python dft_geometry_phonons_remote.py','python worker.py')
    files['job.sbatch']=launch.encode()
    cfg=dict(id=LOCAL.name,remote=REMOTE,model=model,geometry='exact DFT-'+XC+' POSCARs; no relaxation',
             supercell=[4,4,3],displacement_A=.01,input_hashes={k:sha(v) for k,v in files.items()})
    files['protocol.json']=(json.dumps(cfg,indent=2)+'\n').encode()
    for n,b in files.items():
        p=LOCAL/n;p.parent.mkdir(exist_ok=True,parents=True);p.write_bytes(b)
    save('staging-intent.json',dict(remote=REMOTE,files={n:sha(b) for n,b in files.items()},utc=datetime.now(timezone.utc).isoformat()))
    print(c.run('umask 077; mkdir '+shlex.quote(REMOTE)),flush=True)
    print(c.run('mkdir '+' '.join(shlex.quote(REMOTE+'/'+k) for k in ['force000','force001','force002'])),flush=True)
    for n,b in files.items():
        subprocess.run(c.ssh.ssh_args()+['-o','BatchMode=yes','mlip-umt','set -e; set -o noclobber; cat > '+shlex.quote(REMOTE+'/'+n)],input=b,check=True)
    code='import pathlib,hashlib; p=pathlib.Path('+repr(REMOTE)+'); h='+repr({n:sha(b) for n,b in files.items()})+'; assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==v for n,v in h.items()); print("verified")'
    print(c.run('python3 -c '+shlex.quote(code)),flush=True)
    args=['sbatch','--parsable','--partition=apollo','--account=imet-2026-2','--qos=normal','--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=12G','--time=00:15:00','--no-requeue','--job-name=ti-DPA4-'+XC,'--chdir='+REMOTE,'--output='+REMOTE+'/run-%j.out',REMOTE+'/job.sbatch']
    save('submission-intent.json',dict(command=args,utc=datetime.now(timezone.utc).isoformat()))
    job=c.run(shlex.join(args)).strip().split(';')[0];assert job.isdigit(),job
    save('submission.json',dict(job_id=job,remote=REMOTE));print('Submitted',job,flush=True)
else:
    job=json.loads((LOCAL/'submission.json').read_text())['job_id']
    acct=c.run('sacct -j '+job+' -Pn --format=JobID,State,ExitCode,Elapsed,NodeList')
    (LOCAL/'accounting.txt').write_text(acct);print(acct,flush=True)
    print(c.run('tail -20 '+shlex.quote(REMOTE+'/run-'+job+'.out')+' 2>/dev/null || true'),flush=True)
    if action=='collect':
        rows={a[0]:a for a in (line.split('|') for line in acct.splitlines())}
        assert rows[job][1:3]==['COMPLETED','0:0']
        code='import pathlib,json,base64,hashlib; p=pathlib.Path('+repr(REMOTE+'/results')+'); print(json.dumps({str(f.relative_to(p)):{"data":base64.b64encode(f.read_bytes()).decode(),"sha256":hashlib.sha256(f.read_bytes()).hexdigest()} for f in p.rglob("*") if f.is_file()}))'
        data=json.loads(c.run('python3 -c '+shlex.quote(code)))
        for name,v in data.items():
            b=base64.b64decode(v['data']);assert sha(b)==v['sha256']
            p=LOCAL/'results'/name;assert (LOCAL/'results').resolve() in p.resolve().parents
            p.parent.mkdir(exist_ok=True,parents=True)
            if p.exists():assert p.read_bytes()==b
            else:p.write_bytes(b)
        manifest=json.loads((LOCAL/'results/manifest.json').read_text())
        assert all(sha((LOCAL/'results'/n).read_bytes())==h for n,h in manifest.items())
        save('collection.json',dict(job_id=job,files={n:v['sha256'] for n,v in data.items()},utc=datetime.now(timezone.utc).isoformat()))
        print('Collected and verified',flush=True)
