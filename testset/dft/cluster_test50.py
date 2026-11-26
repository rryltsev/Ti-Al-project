"""Private staging and one-time Slurm submissions for the frozen test set."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, shlex, subprocess, time
BASE=Path(__file__).resolve().parent
ROOT=BASE/'test50-dft-20261021-01'
CFG=json.loads((ROOT/'protocol.json').read_text())
REMOTE=CFG['remote_root']
spec=importlib.util.spec_from_file_location('cluster',BASE.parents[1]/'phonons/scripts/cluster.py')
cluster=importlib.util.module_from_spec(spec);spec.loader.exec_module(cluster)
SOURCE=Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project')

def save(name,data):(ROOT/name).write_text(json.dumps(data,indent=2)+'\n')
def upload(content,dest):
    subprocess.run(cluster.ssh.ssh_args()+['-o','BatchMode=yes','mlip-umt','set -e; set -o noclobber; cat > '+shlex.quote(dest)],input=content,check=True)
def stage():
    files={p.relative_to(ROOT).as_posix():p.read_bytes() for p in ROOT.rglob('*') if p.is_file() and p.suffix not in ['.pyc'] and '__pycache__' not in p.parts}
    files['private/POTCAR_Al']=(SOURCE/'VASP-input/POTCAR_Al').read_bytes()
    files['private/POTCAR_Ti']=(SOURCE/'VASP-input/POTCAR_Ti_pv').read_bytes()
    for el in ['Al','Ti']:assert hashlib.sha256(files['private/POTCAR_'+el]).hexdigest()==CFG['paw_sha256'][el]
    hashes={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
    save('staging-intent.json',dict(remote_root=REMOTE,files=hashes,unix=time.time()))
    dirs=sorted({str(Path(n).parent).replace('\\','/') for n in files if str(Path(n).parent)!='.'})
    cluster.run('umask 077; mkdir '+shlex.quote(REMOTE)+'; mkdir -p '+' '.join(shlex.quote(REMOTE+'/'+d) for d in dirs))
    for name,data in files.items():upload(data,REMOTE+'/'+name)
    code="import pathlib,json,hashlib; p=pathlib.Path("+repr(REMOTE)+"); h="+repr(hashes)+"; assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==v for n,v in h.items()); print(json.dumps({'verified_files':len(h)}))"
    result=cluster.run('python3 -c '+shlex.quote(code));save('staging-result.json',json.loads(result));print(result)

def submit(xc):
    assert xc in ['setup','setup-retry1',*CFG['functionals']]
    intent=ROOT/(xc+'-submission-intent.json')
    assert not intent.exists(),'Existing submission intent: reconcile scheduler, do not resubmit blindly'
    args=['sbatch','--parsable','--partition=apollo','--account=imet-2026-2','--qos=normal','--nodes=1','--chdir='+REMOTE,'--job-name=t50-'+xc,'--output='+REMOTE+'/'+xc+'-%A_%a.out','--no-requeue']
    if xc.startswith('setup'):args+=['--ntasks=1','--cpus-per-task=1','--mem=4G','--time=00:30:00',REMOTE+'/setup.sbatch']
    else:
        pre=json.loads(cluster.run('cat '+shlex.quote(REMOTE+'/preflight.json')))
        assert pre['status']=='PASS' and pre['calculations']==150 and pre['identical_geometries']
        save('preflight.json',pre)
        args+=['--ntasks=24','--cpus-per-task=1','--mem=64G','--time=20:00:00','--array=0-49%2',REMOTE+'/production.sbatch',xc]
    save(intent.name,dict(command=args,unix=time.time(),state='submission_intent'))
    result=cluster.run(shlex.join(args)).strip();assert result.split(';')[0].isdigit(),result
    record=dict(job_id=result.split(';')[0],command=args,unix=time.time(),state='submitted')
    save(xc+'-submission.json',record);print(json.dumps(record))

def snapshot():
    jobs=[json.loads(p.read_text())['job_id'] for p in ROOT.glob('*-submission.json')]
    command='squeue -j '+','.join(jobs)+' -o "%.18i %.18j %.2t %.10M %.10l %.6D %R"; sacct -j '+','.join(jobs)+' --format=JobID,JobName,State,ExitCode,Elapsed,AllocCPUS,CPUTimeRAW -P'
    result=cluster.run(command);(ROOT/'scheduler-latest.txt').write_text(result);print(result)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['stage','submit','snapshot']);parser.add_argument('xc',nargs='?');a=parser.parse_args()
    if a.action=='stage':stage()
    elif a.action=='submit':submit(a.xc)
    else:snapshot()
