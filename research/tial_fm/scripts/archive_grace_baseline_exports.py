"""Relocate redundant baseline exports, preserving native foundation weights."""
import json
import archive_before_grace_repeats as arch
from prepare_grace_n694 import RUN,REMOTE

def main():
    arch.OUT=RUN/'storage-baseline-exports';arch.OUT.mkdir(parents=True,exist_ok=True);path=arch.OUT/'plan.json'
    if not path.exists():
        code='''from pathlib import Path
import json,hashlib
r=Path('/home6/u3129/tial-fm-research');plan={}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for rid in ['grace-n200-fp64-pilot-20261109-02','grace-pbe-n694-20261110-01']:
 root=r/rid;w=root/'baseline';assert json.loads((w/'summary.json').read_text())['status']=='PASS' and json.loads((w/'numerical-QC.json').read_text())['status']=='PASS'
 source=json.loads((root/'plan.json').read_text());protected={str(Path(source['source_checkpoint'])/n):h for n,h in source['source_hashes'].items()}
 assert all(sha(Path(n))==h for n,h in protected.items())
 files=[p for p in (w/'original-export').rglob('*') if p.is_file()];assert files
 plan[rid+'__baseline-export']=dict(work=str(w),protected=protected,files={str(p.relative_to(w)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files})
print(json.dumps(plan))
'''
        arch.dump(path,json.loads(arch.m.remote_python(code)))
    arch.main()
if __name__=='__main__':main()
