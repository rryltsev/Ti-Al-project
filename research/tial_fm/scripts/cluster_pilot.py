"""Trusted SSH transport; explicit stage/submit/read-only status and safe collection."""
from pathlib import Path
import argparse,datetime,hashlib,io,json,shlex,subprocess,sys,tarfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parents[1].parent/'MLIP-research/scripts'))
import cluster_ssh
def ssh(command,data=None):
    p=subprocess.run(cluster_ssh.ssh_args()+['-o','BatchMode=yes','mlip-umt',command],input=data,capture_output=True,timeout=180)
    if p.returncode:raise RuntimeError(p.stderr.decode(errors='replace')+p.stdout.decode(errors='replace'))
    if p.stderr:print(p.stderr.decode(errors='replace'),file=sys.stderr)
    return p.stdout
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['stage','stage-training','submit-baseline','submit-training','status','collect','collect-harmonics','command']);ap.add_argument('--run',default='dpa4-n200-pilot-20261106-02');ap.add_argument('--command');a=ap.parse_args()
    run=ROOT/'runs'/a.run;cfg=json.loads((run/'package.json').read_text());remote=cfg['remote_root'];q=shlex.quote(remote)
    if a.action in ['stage','stage-training']:
        if a.action=='stage-training':cfg=json.loads((run/'training-package.json').read_text());remote=cfg['remote_root']
        archive=Path(cfg['path']);assert hashlib.sha256(archive.read_bytes()).hexdigest()==cfg['sha256']
        code="import pathlib,tarfile,sys,hashlib,io,json; p=pathlib.Path("+repr(remote)+"); p.mkdir(parents=True,exist_ok=False); data=sys.stdin.buffer.read(); assert hashlib.sha256(data).hexdigest()=="+repr(cfg['sha256'])+"; t=tarfile.open(fileobj=io.BytesIO(data)); t.extractall(p,filter='data'); h=json.loads((p/'FILES_SHA256.json').read_text()); assert all(hashlib.sha256((p/k).read_bytes()).hexdigest()==v for k,v in h.items()); print('STAGED',len(h))"
        output=ssh('/home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python -c '+shlex.quote(code),archive.read_bytes()).decode();dump(run/(a.action+'.json'),dict(output=output,remote=remote,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()));print(output)
    elif a.action=='submit-baseline':
        assert not (run/'baseline-submission.json').exists()
        command=f'cd {q} && test ! -e baseline-submission.txt && sbatch --parsable --account=imet-2026-2 --partition=apollo --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=24G --time=00:45:00 --job-name=TiAl-FM-B0-H0 --output=baseline-%j.log baseline.sbatch | tee baseline-submission.txt'
        out=ssh(command).decode().strip();assert out.split(';')[0].isdigit(),out
        dump(run/'baseline-submission.json',dict(job_id=out.split(';')[0],remote_root=remote,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command));print(out)
    elif a.action=='submit-training':
        assert not (run/'training-submission.json').exists()
        # Check the stored numerical gates at the point of submission, not just preparation.
        code="import json; from pathlib import Path; p=Path("+repr(remote)+"); assert all(json.loads((p/x/'summary.json').read_text())['status']=='PASS' for x in ['baseline','harmonics'])"
        command='/home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python -c '+shlex.quote(code)+f' && cd {q}/training-plan01 && test ! -e submission.txt && sbatch --parsable --account=imet-2026-2 --partition=v100 --gres=gpu:1 --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=40G --time=04:00:00 --array=0-3%1 --job-name=TiAl-N200-pilot --output=pilot-%A_%a.log train-v2.sbatch | tee submission.txt'
        out=ssh(command).decode().strip();assert out.split(';')[0].isdigit(),out
        dump(run/'training-submission.json',dict(job_id=out.split(';')[0],remote_root=remote,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command));print(out)
    elif a.action=='status':
        jid=json.loads((run/'baseline-submission.json').read_text())['job_id']
        print(ssh(f'squeue -j {jid} -o "%.18i %.9T %.10M %.40R"; sacct -j {jid} --format=JobID,State,Elapsed,AllocCPUS,MaxRSS,ExitCode -P; tail -n 15 {q}/baseline-{jid}.log').decode(errors='replace'))
    elif a.action=='command':print(ssh(a.command).decode(errors='replace'))
    else:
        folders=['harmonics','harmonic-early','runtime-mask-preflight'] if a.action=='collect-harmonics' else ['baseline','harmonics','harmonic-early','runtime-mask-preflight','trainer-zero-preflight']
        code="from pathlib import Path; import tarfile,sys; p=Path("+repr(remote)+"); t=tarfile.open(fileobj=sys.stdout.buffer,mode='w|'); [t.add(f,arcname=f.relative_to(p).as_posix()) for x in "+repr(folders)+" for f in (p/x).rglob('*') if f.is_file() and not f.is_symlink() and f.suffix not in ['.pt','.pt2']]; t.close()"
        data=ssh('/home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python -c '+shlex.quote(code));out=run/'collected';out.mkdir(exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data)) as t:
            for m in t.getmembers():
                target=(out/m.name).resolve();assert target.is_relative_to(out.resolve()) and m.isfile()
                target.parent.mkdir(parents=True,exist_ok=True);payload=t.extractfile(m).read()
                if target.exists():assert target.read_bytes()==payload,('immutable collected file changed',m.name)
                else:target.write_bytes(payload)
        dump(run/'collection.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),bytes=len(data)));print('COLLECTED',len(data))
if __name__=='__main__':main()
