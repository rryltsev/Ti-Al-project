"""Recoverable local storage of completed intermediates; no scientific reruns."""
import json
import archive_before_grace_repeats as arch
from prepare_grace_n694 import RUN

def main():
    arch.OUT=RUN/'storage-recovery';arch.OUT.mkdir(parents=True,exist_ok=True)
    path=arch.OUT/'plan.json'
    if not path.exists():
        code='''from pathlib import Path
import json,hashlib
r=Path('/home6/u3129/tial-fm-research');plan={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for rid in ['dpa4-n200-pilot-20261106-02','dpa4-replicates-full-20261106-02','dpa4-n694-seeds-20261107-01']:
 for w in (r/rid/'training-plan01/cases').glob('r2SCAN*'):
  s=json.loads((w/'training-summary.json').read_text());e=json.loads((w/'evaluation/summary.json').read_text());assert s['status']==e['status']=='PASS'
  final=Path(s['final_checkpoint']);assert sha(final)==e['model_sha256']
  files=[p for p in (w/'merged').glob('*.pt') if p.resolve()!=final.resolve()]
  files += [p for p in (w/'checkpoints').glob('*.pt') if not p.name.endswith('-'+str(s['steps'])+'.pt')]
  assert files
  plan[rid+'__'+w.name]=dict(work=str(w),protected={str(final):sha(final)},files={str(p.relative_to(w)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files})
rid='grace-pbe-seeds-20261110-01'
for w in (r/rid/'cases').iterdir():
 s=json.loads((w/'training-summary.json').read_text());assert s['status']=='PASS' and json.loads((w/'evaluation-summary.json').read_text())['status']=='PASS'
 folder=w/'seed'/str(s['case']['seed'])/'checkpoints';p=folder/'optimizer-final.tar.gz';a=json.loads((folder/'ARCHIVE.json').read_text());assert sha(p)==a['archive_sha256']
 protected={str(w/'final-checkpoint'/n):h for n,h in s['hashes'].items()};assert all(sha(Path(n))==h for n,h in protected.items())
 plan[rid+'__'+w.name]=dict(work=str(w),protected=protected,files={str(p.relative_to(w)):dict(bytes=p.stat().st_size,sha256=sha(p))})
assert len(plan)==20
print(json.dumps(plan))
'''
        arch.dump(path,json.loads(arch.m.remote_python(code)))
    arch.main()
if __name__=='__main__':main()
