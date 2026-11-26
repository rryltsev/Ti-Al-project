"""Apply provenance-only postprocessing revision after smoke, before production."""
import io,json,tarfile,shlex
from prepare_grace_pbe_seeds import RUN,REMOTE,load,dump,sha
import cluster_pilot as c
from manage_expansion import PY
record=load(RUN/'postprocessing-revision.json');names=record['changed_files']
old={n:sha(RUN/'recovery/postprocessing-provenance'/n) for n in names}
new={n:sha(RUN/'package'/n) for n in names}
buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w:gz') as t:
    for n in names:t.add(RUN/'package'/n,arcname=n)
code='''from pathlib import Path
import ast,hashlib,io,json,shutil,sys,tarfile
p=Path(REMOTE);old=OLD;new=NEW
assert json.loads((p/'optimizer-gate.json').read_text())['status']=='PASS'
assert not any((w/'attempt.json').exists() for w in (p/'cases').iterdir())
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
for n,h in old.items():assert sha(p/n)==h,n
r=p/'recovery/postprocessing-provenance';r.mkdir(parents=True,exist_ok=False)
for n in old:
 f=r/n;f.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p/n,f)
tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read())).extractall(p,filter='data')
for n,h in new.items():assert sha(p/n)==h,n
for n,h in json.loads((p/'FILES_SHA256.json').read_text()).items():assert sha(p/n)==h,n
for f in p.rglob('*.py'):ast.parse(f.read_text())
print(json.dumps(dict(status='PASS',changed_files=new,training_configurations_unchanged=True)))
'''.replace('REMOTE',repr(REMOTE)).replace('OLD',repr(old)).replace('NEW',repr(new))
result=json.loads(c.ssh(PY+' -c '+shlex.quote(code),buffer.getvalue()));dump(RUN/'postprocessing-revision-deployed.json',result);print(result)
