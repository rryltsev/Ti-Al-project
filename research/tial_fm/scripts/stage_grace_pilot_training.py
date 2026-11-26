import ast,hashlib,json,shlex
import cluster_pilot as c
from prepare_grace_pilot import ROOT,RUN,OUT,REMOTE,ENV,OLD
def main():
    folder=RUN/'training-package';folder.mkdir(exist_ok=True);files={}
    for name in ['grace_pilot_train.py','grace_pilot_evaluate.py','grace_optimizer_gate.py']:files[name]=(ROOT/'scripts'/name).read_text(encoding='utf8');ast.parse(files[name])
    files['backend-manifest.json']=(OLD/'backend-manifest.json').read_text()
    common=(OUT/'baseline.sbatch').read_text().split(ENV+'/bin/python grace_pilot_baseline.py')[0]
    names=[c['name'] for c in json.loads((OUT/'plan.json').read_text())['cases']]
    pick='RUN='+REMOTE+'\nCASES=('+' '.join(names)+')\nCASE_NAME=${CASES[$SLURM_ARRAY_TASK_ID]}\n'
    files['smoke.sbatch']=common+pick+'mkdir -p "$RUN/smoke/$CASE_NAME"\n'+ENV+'/bin/python "$RUN/grace_pilot_train.py" --case "$CASE_NAME" --smoke > "$RUN/smoke/$CASE_NAME/train.log" 2>&1\n'
    files['optimizer-gate.sbatch']=common+'export CUDA_VISIBLE_DEVICES=""\n/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python grace_optimizer_gate.py\n'
    gate=ENV+'''/bin/python -c 'import json; assert json.load(open("optimizer-gate.json"))["status"] == "PASS"'
'''
    files['train.sbatch']=common+pick+gate+ENV+'/bin/python "$RUN/grace_pilot_train.py" --case "$CASE_NAME" > "$RUN/cases/$CASE_NAME/train.log" 2>&1\n'+ENV+'/bin/python "$RUN/grace_pilot_evaluate.py" --case "$CASE_NAME" > "$RUN/cases/$CASE_NAME/evaluate.log" 2>&1\n'
    files['trained-harmonics.sbatch']=common+pick+'export CUDA_VISIBLE_DEVICES=""\n/home6/u3129/tial-phonons/envs/phonopy-4.8.2/bin/python "$RUN/grace_pilot_harmonics.py" --case "$CASE_NAME"\n'
    files['TRAINING_FILES_SHA256.json']=json.dumps({k:hashlib.sha256(v.encode()).hexdigest() for k,v in files.items()},indent=2)+'\n'
    for name,s in files.items():(folder/name).write_text(s,encoding='utf8',newline='\n')
    code='''import json,sys,hashlib,ast,subprocess
from pathlib import Path
p=Path(REMOTE)
for name,s in json.load(sys.stdin).items():
 f=p/name
 if f.exists() and f.read_text()!=s:
  assert not list((p/'smoke').glob('*/attempt.json')) and not list((p/'cases').glob('*/attempt.json'))
  backup=p/(name+'.prelaunch-v1')
  assert not backup.exists();backup.write_bytes(f.read_bytes());f.write_text(s)
 elif not f.exists():f.write_text(s)
for f in p.glob('*.sbatch'):subprocess.run(['bash','-n',str(f)],check=True)
for name,h in json.loads((p/'TRAINING_FILES_SHA256.json').read_text()).items():assert hashlib.sha256((p/name).read_bytes()).hexdigest()==h
print('TRAINING_CODE_STAGED')
'''.replace('REMOTE',repr(REMOTE))
    print(c.ssh('/home/u3129/miniconda3/envs/lammps_deepmd_dpa4/bin/python -c '+shlex.quote(code),json.dumps(files).encode()).decode())
if __name__=='__main__':main()
