import pathlib,json,hashlib,subprocess,time,fcntl
p=pathlib.Path('/home6/u3129/tial-testsets/pbe15k-ti-1941-20261105-01')
(p/'submission.guard').mkdir(mode=0o700)
if (p/'submission.json').exists():
 print((p/'submission.json').read_text());raise SystemExit(0)
if (p/'submit-intent.json').exists():assert json.loads((p/'submit-intent.json').read_text())['status']=='transport_recovery_before_sbatch'
cfg=json.loads((p/'protocol.json').read_text());a=cfg['allocation'];h=json.loads((p/'staging-manifest.json').read_text());sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();assert all(sha(p/n)==v for n,v in h.items());assert '"status": "PASS"' in (p/'cpu-tests.txt').read_text()
for k in ['checkpoint','frozen_model']:assert sha(pathlib.Path(cfg[k]['path']))==cfg[k]['sha256']
jobname='Ti1941-visc-PBE15k';queue=subprocess.check_output(['squeue','-h','-u','u3129','-o','%i|%j|%k'],universal_newlines=True);assert jobname not in queue and cfg['run_id'] not in queue,'Matching queued job exists'
args=['sbatch','--parsable','--partition='+a['partition'],'--nodelist='+a['node'],'--account='+a['account'],'--qos='+a['qos'],'--nodes=1','--ntasks=1','--cpus-per-task='+str(a['cpus']),'--gres=gpu:'+a['gpu'],'--mem='+a['memory'],'--time='+a['walltime'],'--requeue','--job-name='+jobname,'--comment='+cfg['run_id'],'--chdir='+str(p),'--output='+str(p/'allocation-%j.log'),'--open-mode=append',str(p/'job.sbatch')]
intent=dict(unix=time.time(),remote=str(p),run_id=cfg['run_id'],args=args,status='attempting_once');(p/'submit-intent.json').write_text(json.dumps(intent,indent=2))
r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True);receipt=dict(intent,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
if r.returncode==0:
 jid=r.stdout.strip().split(';')[0];assert jid.isdigit() and jid!='20962153';receipt.update(job_id=jid,status='submitted',new_sbatch_created=True)
else:receipt['status']='submission_failed'
(p/'submission.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt));assert r.returncode==0
