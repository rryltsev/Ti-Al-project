"""Copy and hash qualified aligned checkpoints within the user's cluster project."""
import json
import manage_expansion as m
from prepare_grace_mask_ablation import RUN,REMOTE,PARENT_REMOTE

code=r'''from pathlib import Path
import ast,hashlib,json,shutil,subprocess
p=Path(REMOTE);parent=Path(PARENT);files={}
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
assert subprocess.check_output(['id','-un'],text=True).strip()=='u3129'
for manifest in ['FILES_SHA256.json','TRAINING_FILES_SHA256.json']:
 for n,h in json.loads((p/manifest).read_text()).items():assert sha(p/n)==h,n
for xc in ['PBE','r2SCAN']:
 for n,h in json.loads((p/'aligned'/xc/'alignment.json').read_text())['checkpoint_hashes'].items():
  source=parent/'aligned'/xc/n;dest=p/'aligned'/xc/n;assert sha(source)==h
  if not dest.exists():shutil.copyfile(source,dest)
  assert sha(dest)==h;files[dest.relative_to(p).as_posix()]=h
 for part in ['train','validation']:assert sha(p/'data'/xc/(part+'.extxyz'))==sha(parent/'data'/xc/(part+'.extxyz'))
assert json.loads((p/'baseline/summary.json').read_text())['status']=='PASS'
assert json.loads((p/'baseline/harmonics/summary.json').read_text())['status']=='PASS'
for f in p.glob('*.sbatch'):subprocess.run(['bash','-n',str(f)],check=True)
for f in p.glob('*.py'):ast.parse(f.read_text())
(p/'CHECKPOINT_FILES_SHA256.json').write_text(json.dumps(files,indent=2)+'\n')
print(json.dumps(dict(status='PASS',checkpoints=files,source_baseline=str(parent),identical_data=True)))
'''.replace('REMOTE',repr(REMOTE)).replace('PARENT',repr(PARENT_REMOTE))
if __name__=='__main__':
    d=json.loads(m.remote_python(code));(RUN/'remote-preflight.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
