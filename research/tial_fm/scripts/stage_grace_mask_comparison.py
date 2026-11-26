"""Stage a reproducible CPU comparison and submit it after the phonon array."""
import ast,hashlib,json,shlex
from prepare_grace_mask_ablation import ROOT,RUN,OUT,REMOTE
import manage_expansion as m
import cluster_pilot as c

def main():
    script=(ROOT/'scripts/grace_mask_aggregate.py').read_text();ast.parse(script)
    batch=(OUT/'optimizer-gate.sbatch').read_text().replace('grace_optimizer_gate.py','grace_mask_aggregate.py')
    files={'grace_mask_aggregate.py':script,'comparison.sbatch':batch}
    folder=RUN/'analysis-package';folder.mkdir(exist_ok=True)
    hashes={k:hashlib.sha256(v.encode()).hexdigest() for k,v in files.items()};files['ANALYSIS_FILES_SHA256.json']=json.dumps(hashes,indent=2)+'\n'
    for n,s in files.items():(folder/n).write_text(s,encoding='utf8',newline='\n')
    code=r'''from pathlib import Path
import json,sys,hashlib,subprocess
p=Path(REMOTE)
for n,s in json.load(sys.stdin).items():
 f=p/n
 if f.exists():assert f.read_text()==s,n
 else:f.write_text(s)
for n,h in json.loads((p/'ANALYSIS_FILES_SHA256.json').read_text()).items():assert hashlib.sha256((p/n).read_bytes()).hexdigest()==h
subprocess.run(['bash','-n',str(p/'comparison.sbatch')],check=True)
print('ANALYSIS_STAGED')
'''.replace('REMOTE',repr(REMOTE))
    print(c.ssh(m.PY+' -c '+shlex.quote(code),json.dumps(files).encode()).decode())
    job=json.loads((RUN/'submissions/trained-harmonics.json').read_text())['job_id'];m.RUN=RUN
    m.submit(REMOTE,'comparison',['sbatch','--parsable','--account=imet-2026-2','--partition=apollo','--nodes=1','--ntasks=1','--cpus-per-task=2','--mem=8G','--time=00:10:00','--dependency=afterok:'+job,'--job-name=TiAl-GRACE-mask-compare','--output=comparison-%j.log','comparison.sbatch'])

if __name__=='__main__':main()
