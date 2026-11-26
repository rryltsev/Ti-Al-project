"""Verify the five new raw trajectories for each supplied T,N pair; read-only SSH."""
from pathlib import Path
import json,sys,importlib.util
W=Path(__file__).resolve().parents[3];read=lambda p:json.loads(p.read_text(encoding='utf-8'))
s=importlib.util.spec_from_file_location('c',Path(__file__).with_name('cluster.py'));c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
assert len(sys.argv)>1 and (len(sys.argv)-1)%2==0
local=[];specs=[]
for t,n in zip(sys.argv[1::2],sys.argv[2::2]):
 T,N=int(t),int(n);assert T in [1873,1941,2000,2100] and N>=5 and N%5==0
 p=W/('audit/nov05/pbe15k_ti_1941' if T==1941 else 'audit/oct29/pbe15k_ti_temperature');specs.append((read(p/'protocol.json')['remote'],str(read(p/'submission.json')['job_id']),T,N));local.append((p,T,N))
code=r"""from pathlib import Path
import json,hashlib
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def ti(p):
 s=p.read_text();assert '512 atoms' in s
 rows=[l.split() for l in s.split('Atoms')[1].split('\n',1)[1].split('Velocities')[0].splitlines() if l.strip()];rows=[r for r in rows if len(r)>=5 and r[0].isdigit()]
 return len(rows)==512 and all(r[1]=='2' for r in rows)
out=[]
for root,jid,T,N in SPECS:
 p=Path(root);cfg=read(p/'protocol.json');d=read(p/f'analysis/Ti_{T}K/batch_{N:04d}/summary.json');sources={r['replica']:r['sha256'] for r in d['points'][0]['sources']};reps=[]
 for n in range(N-4,N+1):
  m=read(p/f'states/Ti_{T}K/replicas/replica_{n:04d}/summary.json');f=p/m['stress_file'];prod=f.parent
  valid=m['status']=='completed' and m['temperature_K']==T and m['n_samples']==40001 and m['protocol_sha256']==sha(p/'protocol.json') and m['checkpoint_sha256']==cfg['checkpoint']['sha256'] and m['device']['job_id']==jid and 'A100' in m['device']['name'] and sha(f)==m['stress_sha256']==sources[n] and ti(prod/'mol.data') and f'fix sample all nvt temp {T} {T} 0.2' in (prod/'run.in').read_text() and 'run 40000' in (prod/'run.in').read_text()
  assert valid,m
  reps.append(dict(replica=n,temperature_K=T,volume_A3=m['volume_A3'],prefactor_V_over_kBT=m['volume_A3']*1e-30/(1.380649e-23*T),stress_sha256=sha(f),checks='PASS'))
 out.append(reps)
print(json.dumps(out))
"""
res=json.loads(c.py(code.replace('SPECS',repr(specs))))
for (p,T,N),reps in zip(local,res):
 out=p/f'reviews/Ti_{T}K_N{N}';out.mkdir(parents=True,exist_ok=True);d=read(p/'scheduler-check-latest.json');d['new_batch_replicas']=reps;(out/'runtime-check.json').write_text(json.dumps(d,indent=2)+'\n');print(T,N,'new5raw/model/protocol/512Ti/targetT/normalization PASS')
