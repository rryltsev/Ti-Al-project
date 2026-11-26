"""Collect all stage logs/trajectories after successful completion; hash each file."""
from pathlib import Path
import sys,json,subprocess,shlex,hashlib,tarfile,time
sys.argv=['melting_1775_cluster.py','inspect']
import melting_1775_cluster as c
OUT=c.OUT/'collected';OUT.mkdir(exist_ok=True);remote=c.remote
code='''import pathlib,json,hashlib
p=pathlib.Path(REMOTE);fs=[]
for T in [1775]:
 case=p/f'T{T}K';s=json.loads((case/'summary.json').read_text());assert s['status']=='completed'
 fs.extend(f for f in case.rglob('*') if f.is_file() and (f.suffix in ['.json','.log','.in','.data','.lammpstrj','.txt'] or f.name=='log.lammps'))
fs.extend(f for f in p.iterdir() if f.is_file() and f.suffix in ['.json','.log','.py','.sbatch','.txt'])
print(json.dumps({str(f.relative_to(p)):{'size':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in fs}))
'''.replace('REMOTE',repr(remote))
manifest=json.loads(c.run('python3 -c '+shlex.quote(code)));(OUT/'remote-manifest.json').write_text(json.dumps(manifest,indent=2))
names=[n for n,d in manifest.items() if not (OUT/n).exists()]
if names:
 archive=OUT/('outputs-'+str(int(time.time()))+'.tar.gz');cmd=shlex.join(['tar','-czf','-','-C',remote,'--']+names)
 with archive.open('wb') as f:subprocess.run(c.c.cluster.ssh.ssh_args()+['-o','BatchMode=yes','mlip-umt',cmd],stdout=f,check=True)
 with tarfile.open(archive,'r:gz') as tf:
  for m in tf.getmembers():
   assert m.isfile() and m.name in manifest;dest=OUT/m.name;assert OUT.resolve() in dest.resolve().parents
   b=tf.extractfile(m).read();assert hashlib.sha256(b).hexdigest()==manifest[m.name]['sha256'];dest.parent.mkdir(parents=True,exist_ok=True)
   if dest.exists():assert dest.read_bytes()==b
   else:dest.write_bytes(b)
assert all(hashlib.sha256((OUT/n).read_bytes()).hexdigest()==v['sha256'] for n,v in manifest.items())
d={'status':'verified','files':len(manifest),'bytes':sum(v['size'] for v in manifest.values()),'unix':time.time(),'remote':remote,'note':'All stage trajectories/logs/data copied. Binary restarts and compiled model retained on cluster.'};(OUT/'collection.json').write_text(json.dumps(d,indent=2));print(json.dumps(d),flush=True)

