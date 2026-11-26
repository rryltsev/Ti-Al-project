"""One A100, 1775 K, 100 ps production; no reservation of the second GPU."""
from pathlib import Path
import sys,json,subprocess,shlex,hashlib,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'pbe15k_melting_1775'
sys.path.insert(0,str(ROOT/'testset/dft'));import cluster_test50 as c
cfg=json.loads((OUT/'protocol.json').read_text());remote=cfg['remote']
def run(cmd):
 r=subprocess.run(c.cluster.ssh.ssh_args()+['-o','BatchMode=yes','mlip-umt',cmd],capture_output=True,encoding='utf-8',errors='replace')
 if r.returncode:raise RuntimeError(r.stdout+r.stderr)
 return r.stdout
def save(name,d):(OUT/name).write_text(json.dumps(d,indent=2),encoding='utf-8')
action=sys.argv[1]
if action=='stage':
 assert not (OUT/'staging-intent.json').exists()
 names=['initial.data','reference_two_phase.data','workflow.py','job.sbatch','protocol.json','README_RU.txt']
 files={n:(OUT/n).read_bytes() for n in names};hashes={n:hashlib.sha256(v).hexdigest() for n,v in files.items()}
 save('staging-intent.json',{'remote':remote,'files':hashes,'unix':time.time()})
 run('umask 077; mkdir '+shlex.quote(remote))
 for n,b in files.items():c.upload(b,remote+'/'+n)
 code='import pathlib,hashlib; p=pathlib.Path('+repr(remote)+'); h='+repr(hashes)+'; assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==v for n,v in h.items()); print("Staging verified")'
 print(run('python3 -c '+shlex.quote(code)))
 model=cfg['frozen_model']
 code='import pathlib,hashlib,shutil; s=pathlib.Path('+repr(model['source'])+'); d=pathlib.Path('+repr(remote+'/model_15000.pt2')+'); assert not d.exists(); assert hashlib.sha256(s.read_bytes()).hexdigest()=='+repr(model['sha256'])+'; shutil.copyfile(s,d); assert hashlib.sha256(d.read_bytes()).hexdigest()=='+repr(model['sha256'])+'; print("Validated model copy")'
 print(run('python3 -c '+shlex.quote(code)));hashes['model_15000.pt2']=model['sha256'];save('staging-complete.json',hashes)
elif action=='submit':
 assert (OUT/'staging-complete.json').exists() and not (OUT/'submission-intent.json').exists()
 avail=run('scontrol show node tesla-a101 -o');(OUT/'availability-before-submit.txt').write_text(avail)
 # Submit one GPU even if occupied; scheduler will wait without disturbing other jobs.
 args=['sbatch','--parsable','--partition=v100','--nodelist=tesla-a101','--account=imet-2026-2','--qos=normal','--nodes=1','--ntasks=1','--cpus-per-task=4','--gres=gpu:a100:1','--mem=32G','--time=08:00:00','--no-requeue','--job-name=Ti-melt15k-1775','--chdir='+remote,'--output='+remote+'/allocation-%j.log',remote+'/job.sbatch']
 save('submission-intent.json',{'command':args,'unix':time.time()});jid=run(shlex.join(args)).strip().split(';')[0];assert jid.isdigit(),jid
 save('submission.json',{'job_id':jid,'command':args,'unix':time.time()});print('Submitted',jid)
elif action=='status':
 jid=json.loads((OUT/'submission.json').read_text())['job_id'];acct=run('sacct -j '+jid+' --format=JobID,State,ExitCode,Elapsed,AllocTRES,NodeList -Pn');(OUT/'accounting.txt').write_text(acct);print(acct)
 code='''import pathlib,json
p=pathlib.Path(REMOTE)
d={"metadata":{},"logs":{},"stage_logs":{}}
for f in p.glob("*.json"):
 if f.name not in ["protocol.json"]:
  try:d["metadata"][f.name]=json.loads(f.read_text())
  except Exception:pass
for f in p.glob("*.log"):
 with f.open("rb") as fp:fp.seek(max(0,f.stat().st_size-3000));d["logs"][f.name]=fp.read().decode(errors="replace")
for f in p.glob("T*K/*/log.lammps"):
 with f.open("rb") as fp:fp.seek(max(0,f.stat().st_size-1800));d["stage_logs"][str(f.relative_to(p))]=fp.read().decode(errors="replace")
print(json.dumps(d))
'''.replace('REMOTE',repr(remote))
 data=json.loads(run('python3 -c '+shlex.quote(code)));save('status.json',data);print(json.dumps(data,indent=2))
