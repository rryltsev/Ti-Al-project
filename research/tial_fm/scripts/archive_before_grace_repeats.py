"""Archive only explicit completed-run intermediates, with local payload verification."""
from pathlib import Path
import hashlib,io,json,shlex,subprocess,tarfile
import cluster_pilot as c
import manage_expansion as m
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/grace-pbe-seeds-20261110-01/storage-recovery'
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if not (OUT/'plan.json').exists():
        code='''from pathlib import Path
import json,hashlib
r=Path('/home6/u3129/tial-fm-research');plan={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for rid in ['dpa4-n200-pilot-20261106-02','dpa4-replicates-full-20261106-02','dpa4-n694-seeds-20261107-01']:
 for w in (r/rid/'training-plan01/cases').glob('PBE*'):
  s=json.loads((w/'training-summary.json').read_text());e=json.loads((w/'evaluation/summary.json').read_text());assert s['status']==e['status']=='PASS'
  final=Path(s['final_checkpoint']);assert sha(final)==e['model_sha256'];files=[]
  files += [p for p in (w/'merged').glob('*.pt') if p.resolve()!=final.resolve()]
  files += [p for p in (w/'checkpoints').glob('*.pt') if not p.name.endswith('-'+str(s['steps'])+'.pt')]
  plan[rid+'__'+w.name]=dict(work=str(w),protected={str(final):sha(final)},files={str(p.relative_to(w)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files})
rid='grace-n200-mask-ablation-20261109-01'
for w in (r/rid/'cases').iterdir():
 s=json.loads((w/'training-summary.json').read_text());assert s['status']=='PASS' and json.loads((w/'evaluation-summary.json').read_text())['status']=='PASS'
 files=[p for p in (w/'seed/10/checkpoints').glob('checkpoint*') if p.is_file()]
 if not files:continue
 assert len(files)==2
 protected={str(w/'final-checkpoint'/n):h for n,h in s['hashes'].items()}
 assert all(sha(Path(n))==h for n,h in protected.items())
 plan[rid+'__'+w.name]=dict(work=str(w),protected=protected,files={str(p.relative_to(w)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files})
assert len(plan)==15
print(json.dumps(plan))
'''
        dump(OUT/'plan.json',json.loads(m.remote_python(code)))
    plan=json.loads((OUT/'plan.json').read_text());verified={}
    for key,d in plan.items():
        archive=OUT/(key+'.tar.gz')
        if not archive.exists():
            code='''from pathlib import Path
import hashlib,sys,tarfile
d=RECORD;p=Path(d['work'])
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz',compresslevel=1) as t:
 for name,meta in d['files'].items():
  f=p/name;assert not f.is_symlink() and hashlib.sha256(f.read_bytes()).hexdigest()==meta['sha256'];t.add(f,arcname=name)
'''.replace('RECORD',repr(d))
            part=archive.with_suffix('.part')
            with part.open('wb') as dest:
                proc=subprocess.run(c.cluster_ssh.ssh_args()+['-o','BatchMode=yes','mlip-umt',m.PY+' -c '+shlex.quote(code)],stdout=dest,stderr=subprocess.PIPE,timeout=1800)
            assert proc.returncode==0,proc.stderr.decode();part.rename(archive)
        seen=[]
        with tarfile.open(archive) as t:
            for f in t:
                assert f.isfile() and f.name in d['files'];h=hashlib.sha256();s=t.extractfile(f)
                for block in iter(lambda:s.read(8*2**20),b''):h.update(block)
                assert h.hexdigest()==d['files'][f.name]['sha256'];seen.append(f.name)
        assert set(seen)==set(d['files'])
        verified[key]=dict(path=str(archive),sha256=digest(archive),payloads_verified=True)
        dump(OUT/'verified.json',verified);print('VERIFIED',key,flush=True)
    code='''from pathlib import Path
import hashlib,json,subprocess
plan=PLAN;verified=VERIFIED;r=Path('/home6/u3129/tial-fm-research').resolve();released=[]
active=subprocess.check_output(['squeue','-h','-u','u3129','-o','%Z'],text=True).splitlines()
for key,d in plan.items():
 w=Path(d['work']).resolve();assert w.is_relative_to(r)
 assert not any(str(w) in str(Path(x).resolve()) for x in active if x.startswith('/'))
 for n,h in d['protected'].items():assert hashlib.sha256(Path(n).read_bytes()).hexdigest()==h
 for n,meta in d['files'].items():
  p=(w/n).resolve();assert p.is_relative_to(w) and not (w/n).is_symlink() and str(p) not in {str(Path(x).resolve()) for x in d['protected']}
  if p.exists():
   assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256'];p.unlink();released.append(dict(path=str(p),bytes=meta['bytes']))
 (w/'INTERMEDIATE_LOCAL_ARCHIVE_20261110.json').write_text(json.dumps(dict(archive=verified[key],files=d['files'],protected=d['protected']),indent=2))
print(json.dumps(dict(status='PASS',bytes_released=sum(x['bytes'] for x in released),released=released)))
'''.replace('PLAN',repr(plan)).replace('VERIFIED',repr(verified))
    result=json.loads(c.ssh(m.PY+' -',code.encode()));dump(OUT/'released.json',result);print('RELEASED',result['bytes_released'],flush=True)
if __name__=='__main__':main()
