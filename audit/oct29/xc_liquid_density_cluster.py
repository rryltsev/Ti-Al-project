"""Idempotent staging, one submission, and incremental collection of completed points."""
from pathlib import Path
import hashlib,json,shlex,subprocess,sys,tarfile,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OUT=HERE/'xc_liquid_density_extended'
sys.path.insert(0,str(ROOT/'testset/dft'))
import cluster_test50 as c
cfg=json.loads((OUT/'protocol.json').read_text());remote=cfg['remote']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,data):(OUT/name).write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
def run(command):
    p=subprocess.run(c.cluster.ssh.ssh_args()+['-o','BatchMode=yes','mlip-umt',command],capture_output=True,encoding='utf-8',errors='replace',timeout=150)
    if p.returncode:raise RuntimeError(p.stdout+p.stderr)
    return p.stdout
def py(code):return run('python3 -c '+shlex.quote(code))
action=sys.argv[1]
if action=='stage':
    assert not (OUT/'staging-intent.json').exists()
    names=['protocol.json','tasks.json','md_common.py','workflow.py','job.sbatch','preparation-verification.json','archived_liquid_points.json']+[s['initial_file'] for s in cfg['states']]
    files={n:(OUT/n).read_bytes() for n in names};hashes={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
    save('staging-intent.json',dict(remote=remote,files=hashes,unix=time.time()))
    run('umask 077; mkdir '+shlex.quote(remote))
    for n,b in files.items():c.upload(b,remote+'/'+n)
    code='import pathlib,hashlib; p=pathlib.Path('+repr(remote)+');h='+repr(hashes)+';assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==v for n,v in h.items());models='+repr(cfg['models'])+';assert all(hashlib.sha256(pathlib.Path(m["path"]).read_bytes()).hexdigest()==m["sha256"] for m in models.values());print("Files and both final-XC checkpoints verified")'
    print(py(code));run('bash -n '+shlex.quote(remote+'/job.sbatch'))
    save('staging-complete.json',dict(files=hashes,unix=time.time()))
elif action=='submit':
    assert (OUT/'staging-complete.json').exists() and not (OUT/'submission-intent.json').exists()
    status=run('scontrol show node tesla-a101 -o; scontrol show partition v100 -o')
    (OUT/'availability-before-submit.txt').write_text(status)
    assert 'Gres=gpu:a100:2' in status and 'MaxTime=20:00:00' in status
    args=['sbatch','--parsable','--partition=v100','--nodelist=tesla-a101','--account=imet-2026-2','--qos=normal',
          '--nodes=1','--ntasks=1','--cpus-per-task=4','--gres=gpu:a100:1','--mem=32G','--time=20:00:00',
          '--no-requeue','--job-name=Ti-rhoT-sol-scan','--chdir='+remote,'--output='+remote+'/allocation-%j.log',remote+'/job.sbatch']
    save('submission-intent.json',dict(command=args,unix=time.time()))
    result=run(shlex.join(args));jid=result.strip().split(';')[0];assert jid.isdigit(),result
    save('submission.json',dict(job_id=jid,command=args,unix=time.time()));print('Submitted',jid)
elif action=='status':
    jid=json.loads((OUT/'submission.json').read_text())['job_id']
    scheduler=run('squeue -j '+jid+' -o "%.18i %.18j %.2t %.10M %.10l %R"; sacct -j '+jid+' --format=JobID,State,ExitCode,Elapsed,AllocTRES,NodeList -Pn')
    (OUT/'scheduler-latest.txt').write_text(scheduler);print(scheduler)
    code='''import pathlib,json
p=pathlib.Path(REMOTE);d={'metadata':{},'logs':{}}
for n in ['preflight.json','progress.json','summary.json']:
 f=p/n
 if f.exists():d['metadata'][n]=json.loads(f.read_text())
for f in p.glob('*.log'):
 with f.open('rb') as fp:
  fp.seek(max(0,f.stat().st_size-3500));d['logs'][f.name]=fp.read().decode(errors='replace')
print(json.dumps(d))
'''.replace('REMOTE',repr(remote))
    d=json.loads(py(code));save('status.json',d);print(json.dumps(d,indent=2))
elif action=='collect':
    dest=OUT/'collected';dest.mkdir(exist_ok=True)
    code='''import pathlib,json,hashlib
p=pathlib.Path(REMOTE);files=[]
for summary in (p/'results').glob('*/summary.json'):
 s=json.loads(summary.read_text());assert s['status']=='completed'
 files += [f for f in summary.parent.iterdir() if f.is_file() and f.suffix in ['.json','.jsonl','.traj']]
if files:files += [p/n for n in ['preflight.json','cpu_reference.npz','protocol.json','tasks.json']]
print(json.dumps({str(f.relative_to(p)):{'size':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in files}))
'''.replace('REMOTE',repr(remote))
    manifest=json.loads(py(code));save('collection-manifest.json',manifest)
    names=[n for n in manifest if not (dest/n).exists()]
    if names:
        archive=OUT/'incremental-download.tar.gz'
        command=shlex.join(['tar','-czf','-','-C',remote,'--']+names)
        with archive.open('wb') as fp:subprocess.run(c.cluster.ssh.ssh_args()+['-o','BatchMode=yes','mlip-umt',command],stdout=fp,check=True,timeout=300)
        with tarfile.open(archive,'r:gz') as tf:
            assert sorted(m.name for m in tf.getmembers())==sorted(names)
            for m in tf.getmembers():
                target=dest/m.name;assert m.isfile() and dest.resolve() in target.resolve().parents
                b=tf.extractfile(m).read();assert hashlib.sha256(b).hexdigest()==manifest[m.name]['sha256']
                target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists();target.write_bytes(b)
    assert all(sha(dest/n)==v['sha256'] for n,v in manifest.items())
    if manifest:assert (dest/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
    d=dict(status='verified',new_files=len(names),files=len(manifest),completed_points=sum(n.endswith('/summary.json') for n in manifest),unix=time.time())
    save('collection.json',d);print(json.dumps(d))
else:raise ValueError(action)
