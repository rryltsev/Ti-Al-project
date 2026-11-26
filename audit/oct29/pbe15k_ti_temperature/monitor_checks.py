"""Read-only verification of the two explicitly authorized viscosity campaigns."""
from pathlib import Path
import json,importlib.util
W=Path(__file__).resolve().parents[3]
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
spec=importlib.util.spec_from_file_location('c',Path(__file__).with_name('cluster.py'));c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
roots=[W/'audit/oct29/pbe15k_ti_temperature',W/'audit/nov05/pbe15k_ti_1941']
specs=[(read(p/'protocol.json')['remote'],str(read(p/'submission.json')['job_id']),read(p/'submission.json').get('previous_remote',read(p/'protocol.json')['remote'])) for p in roots]
code=r"""from pathlib import Path
import json,hashlib,subprocess,time,re
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
def ti(p):
 s=p.read_text();assert '512 atoms' in s
 a=s.split('Atoms')[1].split('\n',1)[1].split('Velocities')[0]
 rows=[l.split() for l in a.splitlines() if l.strip() and not l.startswith('#')]
 rows=[r for r in rows if len(r)>=5 and r[0].isdigit()]
 return len(rows)==512 and all(r[1]=='2' for r in rows)
out=[]
for root,jid,workdir in SPECS:
 p=Path(root);cfg=read(p/'protocol.json');h=sha(p/'protocol.json');checks={n:sha(p/n)==v for n,v in read(p/'staging-manifest.json').items()}
 for k in ['checkpoint','frozen_model']:checks[k]=sha(Path(cfg[k]['path']))==cfg[k]['sha256']
 pre=read(p/'preflight.json');checks['preflight']=pre['status']=='passed' and 'A100' in pre['device']['name'] and pre['device']['job_id']==jid
 checks['physics']=all(cfg[k]==v for k,v in {'atoms':512,'dt_ps':.001,'initial_npt_steps':20000,'reservoir_spacing_steps':5000,'nvt_equil_steps':1000,'production_steps':40000}.items())
 checks['normalization']='volume_A3*1e-30/(1.380649e-23*temperature_K)' in (p/'analysis_common.py').read_text() and 'a.channels(pressure*1e5)' in (p/'analyze_point.py').read_text() and '[a.prefactor(vol,T)]*5' in (p/'analyze_point.py').read_text()
 job=subprocess.check_output(['scontrol','show','job',jid,'-o'],universal_newlines=True);fields=dict(re.findall(r'(\w+)=(\S+)',job))
 checks['scope']=fields['JobId']==jid and fields['WorkDir']==workdir
 checks['oneA100']=all('gres/gpu=1' in fields[k] and 'gres/gpu:a100=1' in fields[k] for k in ['ReqTRES','AllocTRES'])
 completed={};reps=[]
 for s in cfg['states']:
  sid=s['id'];T=s['temperature_K'];checks['initial_'+sid]=ti(p/s['initial_file']) and s['counts']=={'Al':0,'Ti':512}
  ss=sorted((p/'states'/sid/'replicas').glob('replica_*/summary.json'));completed[sid]=len(ss)
  if ss:
   m=read(ss[-1]);f=p/m['stress_file'];prod=f.parent
   valid=m['status']=='completed' and m['temperature_K']==T and m['n_samples']==40001 and m['protocol_sha256']==h and m['checkpoint_sha256']==cfg['checkpoint']['sha256'] and m['device']['job_id']==jid and 'A100' in m['device']['name'] and sha(f)==m['stress_sha256'] and ti(prod/'mol.data') and ('fix sample all nvt temp %d %d 0.2'%(T,T)) in (prod/'run.in').read_text() and 'run 40000' in (prod/'run.in').read_text()
   checks['latest_'+sid]=valid;reps.append(dict(id=sid,replica=m['replica'],target_K=T,prefactor_V_over_kBT=m['volume_A3']*1e-30/(1.380649e-23*T),checks='PASS' if valid else 'FAIL'))
 progress=read(p/'progress.json');s=next(x for x in cfg['states'] if x['id']==progress['composition']);T=s['temperature_K'];at=p/'states'/s['id']/'replicas'/('replica_%04d'%progress['replica'])/'attempt_001'
 if progress['phase']=='production':checks['current_target']=('fix sample all nvt temp %d %d 0.2'%(T,T)) in (at/'production/run.in').read_text()
 elif progress['phase']=='replica_preparation':checks['current_target']=('fix prepare all npt temp %d %d 0.1'%(T,T)) in (at/'run.in').read_text()
 else:checks['current_target']=False
 an=read(p/'analysis/latest.json');checks['no_failure']=not any((p/n).exists() for n in ['failure.json','analysis/failure.json'])
 assert all(checks.values()),checks
 out.append(dict(checked_unix=time.time(),checks=checks,raw_job=job,job_fields=fields,completed=completed,latest_replicas=reps,progress=progress,analyzed={k:v['n_replicas'] for k,v in an['points'].items()}))
print(json.dumps(out))
"""
results=json.loads(c.py(code.replace('SPECS',repr(specs))))
for p,d in zip(roots,results):
 (p/'scheduler-check-latest.json').write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
 print(p.name,'PASS',d['job_fields']['JobState'],d['completed'],d['analyzed'])
