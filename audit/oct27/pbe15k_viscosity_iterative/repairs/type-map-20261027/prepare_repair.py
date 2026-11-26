"""Prepare a hash-guarded repair; does not submit or execute remote work."""
import ast,base64,hashlib,json,shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
source=(ROOT/'workflow.py').read_bytes()
tree=ast.parse(source)
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='required_type_indices')
scope={}
exec(compile(ast.Module(body=[fn],type_ignores=[]),'workflow.py','exec'),scope)
check=scope['required_type_indices']
names=['H','He','Li','Be','B','C','N','O','F','Ne','Na','Mg','Al','Si','P','S','Cl','Ar','K','Ca','Sc','Ti']
assert check(names,['Al','Ti'])=={'Al':12,'Ti':21}
assert check(['Ti','Al'],['Al','Ti'])=={'Al':1,'Ti':0}
for bad in [['Al'],['Ti'],['Al','Ti','Al']]:
    try:check(bad,['Al','Ti'])
    except AssertionError:pass
    else:raise AssertionError(f'Invalid map accepted: {bad}')
old_hash=hashlib.sha256((HERE/'workflow.py.before').read_bytes()).hexdigest()
new_hash=hashlib.sha256(source).hexdigest()
remote=json.loads((ROOT/'protocol.json').read_text())['remote']
code=f'''
import ast,base64,hashlib,json,pathlib,subprocess,time
p=pathlib.Path({remote!r})
queue=subprocess.check_output(['squeue','--me','-h','-o','%i|%j'],text=True)
assert not any(line.split('|')[0]=='20961934' or line.split('|')[1]=='TiAl-visc15k-iter' for line in queue.splitlines()), queue
acct=subprocess.check_output(['sacct','-j','20961934','--format=JobID,State,ExitCode','-Pn'],text=True)
assert '20961934|FAILED|1:0' in acct, acct
assert not list(p.glob('states/*/replicas/replica_*/summary.json'))
old=p/'workflow.py'
assert hashlib.sha256(old.read_bytes()).hexdigest()=={old_hash!r}
b=base64.b64decode({base64.b64encode(source).decode()!r})
assert hashlib.sha256(b).hexdigest()=={new_hash!r}
ast.parse(b)
dest=p/'repairs/type-map-20261027'
dest.mkdir(parents=True,exist_ok=False)
(dest/'workflow.py.before').write_bytes(old.read_bytes())
for name in ['failure.json']:
    if (p/name).exists():(p/name).rename(dest/name)
(p/'workflow.py.repair').write_bytes(b)
(p/'workflow.py.repair').replace(old)
record=dict(reason='DPA4 retains full 118-element type map; validate required elements instead of exact binary map',old_sha256={old_hash!r},new_sha256={new_hash!r},physics_changed=False,completed_replicas=0,unix=time.time(),previous_job_id='20961934',accounting=acct)
(dest/'repair.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record,indent=2))
'''
(HERE/'apply_repair.sh').write_text('/home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python -B -c '+shlex.quote(code)+'\n',encoding='utf-8')
(HERE/'requeue.sh').write_text("scontrol requeue 20961934\n",encoding='utf-8')
(HERE/'local-validation.json').write_text(json.dumps(dict(status='PASS',tests=['periodic table Al/Ti indices','reordered binary type map','missing element rejected','duplicate element rejected','AST parse'],old_sha256=old_hash,new_sha256=new_hash),indent=2)+'\n')
print('Type-map regression tests PASS; guarded remote repair prepared.')
