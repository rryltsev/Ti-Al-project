"""Hash-verified staging and durable archival before reference-force jobs."""
import base64,json,shlex
from datetime import datetime,timezone
from cluster import ROOT,RECORD,REMOTE,CFG,run,save

files={}
for name in ['execution-policy-09.json','execution-policy-10.json']:
    files[name]=(ROOT/'config'/name).read_bytes()
for name in ['production_step_v9.py','production_step_v10.py','launch_v15.sbatch','archive_first_r2scan_wavecar_20261101.py']:
    files[name]=(ROOT/'scripts'/name).read_bytes()
import hashlib
hashes={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}
save('reference-production06-staging-intent-20261101-attempt02.json',dict(files=hashes,utc=datetime.now(timezone.utc).isoformat()))
payload={n:base64.b64encode(b).decode() for n,b in files.items()}
code='''import base64,json,hashlib,pathlib
p=pathlib.Path(ROOT)
for n,v in DATA.items():
 b=base64.b64decode(v); target=p/n
 if target.exists(): assert target.read_bytes()==b
 else:
  with target.open('xb') as f:f.write(b)
 assert hashlib.sha256(target.read_bytes()).hexdigest()==HASHES[n]
print('Staged source hashes verified')
'''.replace('ROOT',repr(REMOTE)).replace('DATA',repr(payload)).replace('HASHES',repr(hashes))
print(run(CFG['python_base']+' -c '+shlex.quote(code)),flush=True)
for xc,driver in [('PBE','production_step_v9.py'),('r2SCAN','production_step_v10.py')]:
    print(run('bash -n '+REMOTE+'/launch_v15.sbatch && TIAL_LAUNCH_CHECK=1 bash '+REMOTE+'/launch_v15.sbatch '+driver+' '+xc+' force-000'),flush=True)
intent=RECORD/'r2SCAN-first-wavecar-archive-20261101-intent.json'
if intent.exists():raise RuntimeError('Archive intent exists; reconcile before retry')
save(intent.name,dict(source=REMOTE+'/r2SCAN/production-02/force-001-restart-04/WAVECAR',
    proof=REMOTE+'/r2SCAN/production-05/force-002/restart-provenance.json',
    reason='Validated second displacement has its own fresh checkpoint; preserve prior WAVECAR on Lustre and reclaim home6 space for reference job',utc=datetime.now(timezone.utc).isoformat()))
result=json.loads(run(CFG['python_base']+' '+REMOTE+'/archive_first_r2scan_wavecar_20261101.py'))
save('r2SCAN-first-wavecar-archive-20261101-result.json',dict(result=result))
print(json.dumps(result),flush=True)
print(run('quota -w'),flush=True)
