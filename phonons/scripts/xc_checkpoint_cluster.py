"""CPU-only matched checkpoint sweep, one sequential Slurm job per functional."""
import base64,hashlib,json,shlex,subprocess,sys
from datetime import datetime,timezone
import cluster as c

action,xc=sys.argv[1:3]
assert xc in ['PBEsol','r2SCAN']
p=c.RECORD/('DPA4-'+xc+'-checkpoints-20261102-01');remote=c.REMOTE+'/'+p.name
def sha(b):return hashlib.sha256(b).hexdigest()
def save(n,v):(p/n).write_text(json.dumps(v,indent=2)+'\n')
def remote_python(code):
    r=subprocess.run(c.ssh.ssh_args()+['-o','BatchMode=yes','mlip-umt','python3 -'],input=code,text=True,capture_output=True)
    assert r.returncode==0,r.stdout+r.stderr
    return r.stdout
if action=='submit':
    p.mkdir(exist_ok=False)
    geom=c.ROOT/'results'/('PBEsol-phonons-preliminary-20261031-02/inputs' if xc=='PBEsol' else 'r2SCAN-DFT-pair-inputs-20261101')
    registry=json.loads((c.ROOT.parent/'audit/oct25/all_xc_audit/protocol.json').read_text())['models']
    files={}
    for step in [5000,10000,15000,20000]:
        prefix=str(step)+'/'
        local={k+'/POSCAR':(geom/k/'POSCAR').read_bytes() for k in ['force000','force001','force002']}
        local['phonopy_disp.yaml']=(geom/'phonopy_disp.yaml').read_bytes()
        local['worker.py']=(c.ROOT/'scripts/dpa4_matched_forces.py').read_bytes()
        cfg=dict(id=p.name+'-'+str(step),model=registry[xc+'-'+str(step)],geometry='Exact DFT-'+xc+' equilibrium; no relaxation',supercell=[4,4,3],displacement_A=.01,input_hashes={n:sha(b) for n,b in local.items()})
        local['protocol.json']=(json.dumps(cfg,indent=2)+'\n').encode()
        files.update({prefix+n:b for n,b in local.items()})
    launcher=(c.ROOT.parent/'audit/oct25/pbe_phonons_dft_geometry/job.sbatch').read_text().replace('$SLURM_ARRAY_TASK_ID','$SLURM_JOB_ID').replace('python dft_geometry_phonons_remote.py','python sweep.py')
    files['job.sbatch']=launcher.encode()
    files['sweep.py']=b'import subprocess,sys\nfrom pathlib import Path\nfor step in [5000,10000,15000,20000]:\n print("CHECKPOINT",step,flush=True)\n subprocess.run([sys.executable,"worker.py"],cwd=Path(str(step)),check=True)\nprint("SWEEP COMPLETE",flush=True)\n'
    for n,b in files.items():
        f=p/n;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(b)
    save('staging-intent.json',dict(remote=remote,utc=datetime.now(timezone.utc).isoformat(),hashes={n:sha(b) for n,b in files.items()}))
    data={n:base64.b64encode(b).decode() for n,b in files.items()}
    code='import pathlib,base64,hashlib,os\nos.umask(0o077)\np=pathlib.Path('+repr(remote)+');p.mkdir()\n'
    code+='for n,s in '+repr(data)+'.items():\n f=p/n;f.parent.mkdir(parents=True,exist_ok=True);b=base64.b64decode(s);f.write_bytes(b);assert hashlib.sha256(f.read_bytes()).hexdigest()==hashlib.sha256(b).hexdigest()\nprint("Staged and verified")\n'
    print(remote_python(code),flush=True)
    queue=c.run("squeue -h -u u3129 -o '%i|%Z|%T'");assert remote not in queue
    (p/'pre-submit-queue.txt').write_text(queue)
    args=['sbatch','--parsable','--partition=apollo','--account=imet-2026-2','--qos=normal','--nodes=1','--ntasks=1','--cpus-per-task=4','--mem=12G','--time=00:15:00','--no-requeue','--job-name=ti-ckpt-'+xc,'--chdir='+remote,'--output='+remote+'/run-%j.out',remote+'/job.sbatch']
    save('submission-intent.json',dict(command=args,utc=datetime.now(timezone.utc).isoformat()))
    job=c.run(shlex.join(args)).strip().split(';')[0];assert job.isdigit(),job
    save('submission.json',dict(job_id=job,remote=remote));print(xc,'Submitted',job,flush=True)
else:
    job=json.loads((p/'submission.json').read_text())['job_id']
    acct=c.run('sacct -j '+job+' -Pn --format=JobID,State,ExitCode,Elapsed,NodeList')
    (p/'accounting.txt').write_text(acct);print(acct,flush=True)
    if action=='collect':
        rows={v[0]:v for v in (s.split('|') for s in acct.splitlines())};assert rows[job][1:3]==['COMPLETED','0:0']
        code='import pathlib,json,base64,hashlib\np=pathlib.Path('+repr(remote)+')\nprint(json.dumps({str(f.relative_to(p)):{"data":base64.b64encode(f.read_bytes()).decode(),"sha256":hashlib.sha256(f.read_bytes()).hexdigest()} for f in p.glob("*/results/**/*") if f.is_file()}))'
        data=json.loads(remote_python(code))
        for n,v in data.items():
            b=base64.b64decode(v['data']);assert sha(b)==v['sha256'];f=p/n;assert p.resolve() in f.resolve().parents
            f.parent.mkdir(parents=True,exist_ok=True)
            if f.exists():assert f.read_bytes()==b
            else:f.write_bytes(b)
        for step in [5000,10000,15000,20000]:
            results=p/str(step)/'results'
            for n,h in json.loads((results/'manifest.json').read_text()).items():assert sha((results/n).read_bytes())==h
        save('collection.json',dict(job_id=job,utc=datetime.now(timezone.utc).isoformat(),files={n:v['sha256'] for n,v in data.items()}))
        print('Collected and verified',flush=True)
