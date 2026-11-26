from pathlib import Path
import hashlib,json,shlex
import cluster_pilot as c
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/dpa4-n200-pilot-20261106-02';remote=json.loads((RUN/'package.json').read_text())['remote_root']
source=(ROOT/'scripts/resume_baseline.py').read_text();launcher=(RUN/'package/baseline.sbatch').read_text().split('python baseline_worker.py')[0]+'python resume_baseline.py\n'
payload={'resume_baseline.py':source,'baseline-continuation.sbatch':launcher};payload['baseline-continuation-inputs.json']=json.dumps({k:hashlib.sha256(v.encode()).hexdigest() for k,v in payload.items()},indent=2)+'\n'
code="import json,pathlib,sys; root=pathlib.Path("+repr(remote)+"); files=json.load(sys.stdin); [(root/k).open('x').write(v) for k,v in files.items()]; print('STAGED_BASELINE_CONTINUATION')"
print(c.ssh('/home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python -c '+shlex.quote(code),json.dumps(payload).encode()).decode())
cmd=f'cd {shlex.quote(remote)} && sbatch --parsable --dependency=afterany:20988440 --account=imet-2026-2 --partition=apollo --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=24G --time=00:45:00 --job-name=TiAl-FM-B0-resume --output=baseline-continuation-%j.log baseline-continuation.sbatch'
out=c.ssh(cmd).decode().strip();assert out.split(';')[0].isdigit();jid=out.split(';')[0]
(RUN/'baseline-continuation-submission.json').write_text(json.dumps(dict(job_id=jid,command=cmd,reason='CPU freeze plus native prediction throughput may exceed initial45min. Only a TIMEOUT resumes completed system files; other unexplained failures stop. Already-complete B0 returns without duplicated evaluations.'),indent=2)+'\n')
print(c.ssh('scontrol update JobId=20988442 Dependency=afterok:'+jid).decode());print(jid)
