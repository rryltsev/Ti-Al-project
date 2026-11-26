"""Download completed density outputs and verify every file by SHA256."""
from pathlib import Path
import hashlib, json, shlex, subprocess, sys, tarfile, time
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
OUT=HERE/'pbe15k_density_1873'; DEST=OUT/'collected'; DEST.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'testset/dft'))
import cluster_test50 as c
remote=json.loads((OUT/'protocol.json').read_text())['remote']
code='''import pathlib,json,hashlib
p=pathlib.Path(REMOTE)
s=json.loads((p/'15000/summary.json').read_text())
assert s['status']=='completed' and len(s['completed'])==11 and not s['failed']
files=[f for f in (p/'15000').rglob('*') if f.is_file() and f.suffix in ['.traj','.json','.jsonl','.npz']]
files += [f for f in p.iterdir() if f.is_file() and f.suffix in ['.json','.log','.py','.sbatch','.csv','.txt']]
print(json.dumps({str(f.relative_to(p)):{'size':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in files}))
'''.replace('REMOTE',repr(remote))
manifest=json.loads(c.cluster.run('python3 -c '+shlex.quote(code)))
(OUT/'collection-manifest.json').write_text(json.dumps(manifest,indent=2))
names=[n for n,d in manifest.items() if not (DEST/n).exists()]
print('Downloading',len(names),'files;',sum(manifest[n]['size'] for n in names),'bytes',flush=True)
if names:
    archive=OUT/'results-download.tar.gz'
    command=shlex.join(['tar','-czf','-','-C',remote,'--']+names)
    with archive.open('wb') as f:
        subprocess.run(c.cluster.ssh.ssh_args()+['-o','BatchMode=yes','mlip-umt',command],stdout=f,check=True)
    with tarfile.open(archive,'r:gz') as tf:
        for member in tf.getmembers():
            assert member.isfile() and member.name in manifest
            dest=DEST/member.name
            assert DEST.resolve() in dest.resolve().parents
            content=tf.extractfile(member).read()
            assert hashlib.sha256(content).hexdigest()==manifest[member.name]['sha256']
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists(): assert dest.read_bytes()==content
            else: dest.write_bytes(content)
assert all(hashlib.sha256((DEST/n).read_bytes()).hexdigest()==v['sha256'] for n,v in manifest.items())
assert (DEST/'protocol.json').read_bytes()==(OUT/'protocol.json').read_bytes()
summary=dict(status='verified',files=len(manifest),bytes=sum(v['size'] for v in manifest.values()),unix=time.time())
(OUT/'collection.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary),flush=True)
