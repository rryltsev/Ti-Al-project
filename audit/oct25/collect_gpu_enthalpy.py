"""Download complete GPU outputs with per-file SHA256 verification, never trimming stages."""
from pathlib import Path
import sys,json,subprocess,shlex,hashlib,tarfile,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'pbe_full_enthalpy_gpu'
sys.path.insert(0,str(ROOT/'testset/dft'));import cluster_test50 as c
remote=json.loads((OUT/'protocol.json').read_text())['remote'];mode=sys.argv[1]
assert mode in ['scalars','trajectories']
code='''import pathlib,json,hashlib
p=pathlib.Path(REMOTE);fs=[]
for step in [15000,20000]:
 s=json.loads((p/str(step)/"summary.json").read_text());assert s["status"]=="completed" and len(s["completed"])==11 and not s["failed"]
 fs.extend(f for f in (p/str(step)).rglob("*") if f.is_file() and ((f.suffix==".traj") if MODE=="trajectories" else (f.suffix in [".json",".jsonl",".npz"])))
if MODE=="scalars":fs.extend(p.glob("*.log"))
print(json.dumps({str(f.relative_to(p)):{"size":f.stat().st_size,"sha256":hashlib.sha256(f.read_bytes()).hexdigest()} for f in fs}))
'''.replace('REMOTE',repr(remote)).replace('MODE',repr(mode))
manifest=json.loads(c.cluster.run('python3 -c '+shlex.quote(code)));(OUT/(mode+'-remote-manifest.json')).write_text(json.dumps(manifest,indent=2))
names=[n for n,d in manifest.items() if not (OUT/n).exists() or hashlib.sha256((OUT/n).read_bytes()).hexdigest()!=d['sha256']]
if names:
    archive=OUT/(mode+'-download.tar.gz')
    cmd=shlex.join(['tar','-czf','-','-C',remote,'--']+names)
    with archive.open('wb') as f:subprocess.run(c.cluster.ssh.ssh_args()+['-o','BatchMode=yes','mlip-umt',cmd],stdout=f,check=True)
    with tarfile.open(archive,'r:gz') as tf:
        for m in tf.getmembers():
            assert m.isfile() and m.name in manifest;dest=OUT/m.name;assert OUT.resolve() in dest.resolve().parents
            b=tf.extractfile(m).read();assert hashlib.sha256(b).hexdigest()==manifest[m.name]['sha256']
            dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert dest.read_bytes()==b,'Refusing to overwrite differing local output'
            else:dest.write_bytes(b)
assert all(hashlib.sha256((OUT/n).read_bytes()).hexdigest()==d['sha256'] for n,d in manifest.items())
summary={'status':'verified','mode':mode,'files':len(manifest),'bytes':sum(d['size'] for d in manifest.values()),'unix':time.time()}
(OUT/(mode+'-collection.json')).write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
