/home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python -B -c '
import hashlib,json,pathlib,subprocess,time
p=pathlib.Path('"'"'/home6/u3129/tial-testsets/pbe15k-viscosity-1873-20261027-01'"'"')
r=p/'"'"'repairs/type-map-20261027'"'"'
receipt=r/'"'"'replacement-submission.json'"'"'
if receipt.exists():
    print(receipt.read_text())
    raise SystemExit(0)
queue=subprocess.check_output(['"'"'squeue'"'"','"'"'--me'"'"','"'"'-h'"'"','"'"'-o'"'"','"'"'%i|%j'"'"'],text=True)
assert not any(line.split('"'"'|'"'"')[0]=='"'"'20961934'"'"' or line.split('"'"'|'"'"')[1]=='"'"'TiAl-visc15k-iter'"'"' for line in queue.splitlines()),queue
acct=subprocess.check_output(['"'"'sacct'"'"','"'"'-j'"'"','"'"'20961934'"'"','"'"'--format=JobID,State,ExitCode'"'"','"'"'-Pn'"'"'],text=True)
assert '"'"'20961934|FAILED|1:0'"'"' in acct,acct
repair=json.loads((r/'"'"'repair.json'"'"').read_text())
assert hashlib.sha256((p/'"'"'workflow.py'"'"').read_bytes()).hexdigest()==repair['"'"'new_sha256'"'"']
assert not (p/'"'"'final.json'"'"').exists()
args=['"'"'sbatch'"'"', '"'"'--parsable'"'"', '"'"'--partition=v100'"'"', '"'"'--nodelist=tesla-a101'"'"', '"'"'--account=imet-2026-2'"'"', '"'"'--qos=normal'"'"', '"'"'--nodes=1'"'"', '"'"'--ntasks=1'"'"', '"'"'--cpus-per-task=6'"'"', '"'"'--gres=gpu:a100:1'"'"', '"'"'--mem=32G'"'"', '"'"'--time=20:00:00'"'"', '"'"'--requeue'"'"', '"'"'--job-name=TiAl-visc15k-iter'"'"', '"'"'--chdir=/home6/u3129/tial-testsets/pbe15k-viscosity-1873-20261027-01'"'"', '"'"'--output=/home6/u3129/tial-testsets/pbe15k-viscosity-1873-20261027-01/allocation-%j.log'"'"', '"'"'--open-mode=append'"'"', '"'"'/home6/u3129/tial-testsets/pbe15k-viscosity-1873-20261027-01/job.sbatch'"'"']
with (r/'"'"'replacement-intent.json'"'"').open('"'"'x'"'"') as fp:
    json.dump(dict(command=args,previous_job_id='"'"'20961934'"'"',reason='"'"'Failed job was purged; scontrol requeue returned Invalid job id'"'"',unix=time.time(),queue=queue,accounting=acct),fp,indent=2)
result=subprocess.run(args,capture_output=True,text=True)
(r/'"'"'replacement-sbatch-result.json'"'"').write_text(json.dumps(dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr),indent=2))
assert result.returncode==0,result.stderr
jid=result.stdout.strip().split('"'"';'"'"')[0]
assert jid.isdigit(),result.stdout
record=dict(job_id=jid,command=args,unix=time.time(),previous_job_id='"'"'20961934'"'"',reason='"'"'Replacement after corrected preflight assertion; original job purged'"'"')
receipt.write_text(json.dumps(record,indent=2)+'"'"'\n'"'"')
print(receipt.read_text())
'
