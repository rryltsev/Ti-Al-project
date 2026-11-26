"""A single Slurm allocation owns both A100 workers and all 22 MD trajectories."""
from pathlib import Path
import sys,json,hashlib,shlex,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'pbe_full_enthalpy_gpu'
sys.path.insert(0,str(ROOT/'testset/dft'));import cluster_test50 as c
cfg=json.loads((OUT/'protocol.json').read_text());REMOTE=cfg['remote']
save=lambda n,d:(OUT/n).write_text(json.dumps(d,indent=2))
sha=lambda b:hashlib.sha256(b).hexdigest()
action=sys.argv[1]
if action=='stage':
    assert not (OUT/'staging-intent.json').exists()
    names=['protocol.json','tasks.json','baseline_5k.json','md_common.py','gpu_enthalpy_worker.py','gpu_enthalpy_job.sbatch']+[s['initial_file'] for s in cfg['states']]
    files={n:(OUT/n).read_bytes() for n in names};hashes={n:sha(b) for n,b in files.items()}
    save('staging-intent.json',{'files':hashes,'remote':REMOTE,'unix':time.time()})
    c.cluster.run('umask 077; mkdir '+shlex.quote(REMOTE))
    for n,b in files.items():c.upload(b,REMOTE+'/'+n)
    code='import pathlib,hashlib; p=pathlib.Path('+repr(REMOTE)+'); h='+repr(hashes)+'; assert all(hashlib.sha256((p/n).read_bytes()).hexdigest()==v for n,v in h.items()); print("Verified staging")'
    print(c.cluster.run('python3 -c '+shlex.quote(code)));save('staging-complete.json',{'files':hashes,'unix':time.time()})
elif action=='submit':
    assert not (OUT/'submission-intent.json').exists()
    availability=c.cluster.run('scontrol show node tesla-a101 -o');(OUT/'availability-before-submit.txt').write_text(availability)
    assert 'State=IDLE' in availability and 'CPUAlloc=0 ' in availability,'A100 host no longer idle; reassess'
    args=['sbatch','--parsable','--partition=v100','--nodelist=tesla-a101','--account=imet-2026-2','--qos=normal',
      '--nodes=1','--ntasks=2','--cpus-per-task=4','--gres=gpu:a100:2','--mem=32G','--time=24:00:00',
      '--no-requeue','--job-name=pbe-full-H-15k20k','--chdir='+REMOTE,'--output='+REMOTE+'/allocation-%j.log',REMOTE+'/gpu_enthalpy_job.sbatch']
    save('submission-intent.json',{'command':args,'unix':time.time()})
    jid=c.cluster.run(shlex.join(args)).strip().split(';')[0];assert jid.isdigit(),jid
    save('submission.json',{'job_id':jid,'command':args,'unix':time.time()});print('Submitted one two-A100 allocation',jid)
elif action=='status':
    jid=json.loads((OUT/'submission.json').read_text())['job_id']
    acct=c.cluster.run('sacct -j '+jid+' --format=JobID,State,ExitCode,Elapsed,AllocCPUS,AllocTRES,NodeList -Pn');(OUT/'accounting.txt').write_text(acct);print(acct)
    code='''import pathlib,json
p=pathlib.Path(REMOTE)
d={"logs":{f.name:f.read_text(errors="replace")[-5000:] for f in p.glob("*.log")},"preflight":{},"progress":{},"thermo":{}}
for step in [15000,20000]:
 for name in ["preflight.json","progress.json","summary.json"]:
  f=p/str(step)/name
  if f.exists():d["preflight" if name=="preflight.json" else "progress"][str(step)+"/"+name]=json.loads(f.read_text())
 for f in (p/str(step)).glob("results/*/*-thermo.jsonl"):
  if f.stat().st_size:
   lines=f.read_text().splitlines()
   for line in reversed(lines[-2:]):
    try:d["thermo"][str(f.relative_to(p))]=json.loads(line);break
    except json.JSONDecodeError:pass
print(json.dumps(d))
'''.replace('REMOTE',repr(REMOTE))
    d=json.loads(c.cluster.run('python3 -c '+shlex.quote(code)));save('status.json',d)
    print(json.dumps({'preflight':d['preflight'],'progress':d['progress'],'thermo':d['thermo'],'logs':{k:v[-1800:] for k,v in d['logs'].items()}},indent=2))
