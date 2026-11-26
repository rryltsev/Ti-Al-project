"""Stage once, submit one A100 allocation, inspect and collect its own results."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
import importlib.util,json,hashlib,shlex,subprocess,time,tarfile,io
ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[2]
spec=importlib.util.spec_from_file_location('ssh_helpers',WORK.parent/'MLIP-research/scripts/cluster_ssh.py')
ssh=importlib.util.module_from_spec(spec);spec.loader.exec_module(ssh)
cfg=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'));REMOTE=cfg['remote']
sha=lambda b:hashlib.sha256(b).hexdigest()

def run(command,content=None,timeout=180,binary=False):
    r=subprocess.run(ssh.ssh_args()+['-o','BatchMode=yes','mlip-umt',command],input=content,capture_output=True,timeout=timeout)
    if r.returncode: raise RuntimeError(r.stdout.decode(errors='replace')+r.stderr.decode(errors='replace'))
    return r.stdout if binary else r.stdout.decode('utf-8',errors='replace')
def save(name,obj):
    (ROOT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def py(code):return run('python3 -c '+shlex.quote(code))
def jobid():return json.loads((ROOT/'submission.json').read_text())['job_id']

action=sys.argv[1]
if action=='stage':
    assert not (ROOT/'staging-intent.json').exists()
    names=['protocol.json','baseline_5k.json','workflow.py','analyze_batches.py','job.sbatch','README_RU.txt','test_analysis.py']+[s['initial_file'] for s in cfg['states']]
    files={n:(ROOT/n).read_bytes() for n in names};hashes={n:sha(b) for n,b in files.items()}
    save('staging-intent.json',dict(remote=REMOTE,files=hashes,unix=time.time()))
    run('umask 077; mkdir '+shlex.quote(REMOTE))
    for n,b in files.items():run('set -e; set -o noclobber; cat > '+shlex.quote(REMOTE+'/'+n),content=b)
    code='import pathlib,hashlib; p=pathlib.Path('+repr(REMOTE)+'); h='+repr(hashes)+'; assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==v for n,v in h.items()); models='+repr([cfg['checkpoint'],cfg['frozen_model']])+'; assert all(hashlib.sha256(pathlib.Path(m["path"]).read_bytes()).hexdigest()==m["sha256"] for m in models); print("All input/model hashes verified")'
    print(py(code));print(run('bash -n '+shlex.quote(REMOTE+'/job.sbatch')))
    save('staging-complete.json',dict(remote=REMOTE,files=hashes,unix=time.time()))
elif action=='setup-analysis':
    # Private dependency directory: do not mutate the shared LAMMPS environment.
    p='/home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python'
    command='umask 077; '+p+' -m pip install --no-deps --target '+shlex.quote(REMOTE+'/analysis_deps')+" stacie==1.3.0 'attrs>=23.1.0' --disable-pip-version-check"
    print(run(command,timeout=180))
    code='import sys,importlib.metadata;sys.path.insert(0,'+repr(REMOTE+'/analysis_deps')+');import stacie;from stacie import estimate_acint,LorentzModel,compute_spectrum;print({p:importlib.metadata.version(p) for p in ["stacie","attrs","scipy","numpy","matplotlib"]})'
    print(run(p+' -c '+shlex.quote(code)))
    save('analysis-environment.json',dict(stacie='1.3.0',target=REMOTE+'/analysis_deps',checked=True))
elif action=='test':
    p='/home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python'
    command='cd '+shlex.quote(REMOTE)+'; OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLCONFIGDIR='+shlex.quote(REMOTE+'/cache/matplotlib')+' '+p+' test_analysis.py'
    result=run(command,timeout=180);print(result);(ROOT/'remote-tests.txt').write_text(result,encoding='utf-8')
elif action=='submit':
    assert (ROOT/'staging-complete.json').exists() and (ROOT/'analysis-environment.json').exists()
    assert (ROOT/'remote-tests.txt').exists() and not (ROOT/'submission-intent.json').exists()
    availability=run('scontrol show node tesla-a101 -o; scontrol show partition v100 -o')
    (ROOT/'availability-before-submit.txt').write_text(availability)
    assert 'gpu:a100:2' in availability and 'MaxTime=20:00:00' in availability
    a=cfg['allocation']
    args=['sbatch','--parsable','--partition='+a['partition'],'--nodelist='+a['node'],
        '--account='+a['account'],'--qos='+a['qos'],'--nodes=1','--ntasks=1',
        '--cpus-per-task='+str(a['cpus']),'--gres=gpu:'+a['gpu'],'--mem='+a['memory'],
        '--time='+a['walltime'],'--requeue','--job-name=TiAl-visc15k-iter',
        '--chdir='+REMOTE,'--output='+REMOTE+'/allocation-%j.log','--open-mode=append',REMOTE+'/job.sbatch']
    save('submission-intent.json',dict(command=args,unix=time.time()))
    jid=run(shlex.join(args)).strip().split(';')[0];assert jid.isdigit(),jid
    save('submission.json',dict(job_id=jid,command=args,unix=time.time()));print('Submitted single job',jid)
elif action in ['status','collect']:
    jid=jobid()
    account=run('squeue -j '+jid+' -o "%.18i %.24j %.2t %.10M %.10l %R"; sacct -j '+jid+' --format=JobID,State,ExitCode,Elapsed,Timelimit,AllocTRES,NodeList -Pn')
    (ROOT/'scheduler-latest.txt').write_text(account);print(account)
    code='''import pathlib,json
p=pathlib.Path(REMOTE)
d={"metadata":{},"logs":{}}
for n in ["preflight.json","progress.json","final.json","failure.json","analysis/latest.json","analysis/failure.json"]:
 f=p/n
 if f.exists():
  try:d["metadata"][n]=json.loads(f.read_text())
  except json.JSONDecodeError:pass
for f in list(p.glob("*.log"))+list((p/"analysis").glob("*.log"))[-2:]:
 with f.open("rb") as fp:fp.seek(max(0,f.stat().st_size-2500));d["logs"][str(f.relative_to(p))]=fp.read().decode(errors="replace")
print(json.dumps(d))
'''.replace('REMOTE',repr(REMOTE))
    data=json.loads(py(code));save('status.json',data)
    print(json.dumps(data,ensure_ascii=False,indent=2))
    if action=='collect':
        code='''import pathlib,tarfile,sys
p=pathlib.Path(REMOTE)
with tarfile.open(fileobj=sys.stdout.buffer,mode="w|gz") as t:
 for name in ["analysis","preflight.json","progress.json","final.json","failure.json"]:
  f=p/name
  if f.exists():t.add(f,arcname=name,recursive=True)
'''.replace('REMOTE',repr(REMOTE))
        payload=run('python3 -c '+shlex.quote(code),binary=True)
        dest=ROOT/'collected';dest.mkdir(exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as t:
            for m in t.getmembers():
                target=(dest/m.name).resolve()
                assert target.is_relative_to(dest.resolve()) and not m.issym() and not m.islnk()
                if m.isfile():target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(t.extractfile(m).read())
        save('collection.json',dict(unix=time.time(),bytes=len(payload),job_id=jid))
elif action=='stop-after-batch':
    control={'stop_after_batch':True,'reason':sys.argv[2] if len(sys.argv)>2 else 'Requested stop'}
    code='import pathlib,json; p=pathlib.Path('+repr(REMOTE)+'); (p/"control.tmp").write_text('+repr(json.dumps(control))+'); (p/"control.tmp").replace(p/"control.json")'
    print(py(code));save('stop-request.json',control)
elif action=='exec':
    assert len(sys.argv)==3
    print(run(Path(sys.argv[2]).read_text(encoding='utf-8'),timeout=180))
else: raise ValueError(action)
