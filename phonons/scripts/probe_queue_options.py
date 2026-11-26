"""Read-only scheduler projections: sbatch --test-only never submits jobs."""
import json
import shlex
from datetime import datetime, timezone
from cluster import CFG, run, save

options=[
    ('apollo-current32','apollo',1,32,300,'20:00:00'),
    ('apollo-32-compact','apollo',1,32,240,'20:00:00'),
    ('apollo-32-split2','apollo',2,32,128,'20:00:00'),
    ('apollo-32-split4','apollo',4,32,80,'20:00:00'),
    ('apollo-64-compact','apollo',2,64,240,'20:00:00'),
    ('apollo-64-split4','apollo',4,64,128,'20:00:00'),
    ('apollo-32-split2-long','apollo',2,32,128,'48:00:00'),
    ('apollo-32-compact-long','apollo',1,32,240,'48:00:00'),
    ('tesla-32-split4','tesla',4,32,80,'20:00:00'),
    ('tesla-32-split8','tesla',8,32,40,'20:00:00'),
    ('v100-cpu32','v100',1,32,240,'20:00:00'),
]
code='''
import json,subprocess
options=OPTIONS
result=[]
for name,partition,nodes,ranks,mem,wall in options:
    args=['sbatch','--test-only','--partition='+partition,'--account=imet-2026-2','--qos=normal',
          '--nodes='+str(nodes),'--ntasks='+str(ranks),'--ntasks-per-node='+str(ranks//nodes),
          '--cpus-per-task=1','--mem='+str(mem)+'G','--time='+wall,'--wrap=/bin/true']
    p=subprocess.run(args,text=True,capture_output=True,timeout=30)
    result.append(dict(name=name,command=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
print(json.dumps(result))
'''.replace('OPTIONS',repr(options))
result=json.loads(run(CFG['python_base']+' -c '+shlex.quote(code)))
save('queue-options-20261023.json',dict(checked_utc=datetime.now(timezone.utc).isoformat(),options=result))
for r in result: print(r['name'],r['returncode'],r['stdout'].strip(),r['stderr'].strip(),flush=True)
