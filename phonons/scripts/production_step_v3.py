"""One traceable force calculation per allocation; no cumulative resource cap."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
import workflow as w

POLICY = json.loads((w.ROOT / 'execution-policy-03.json').read_text())

def main(xc, step):
    if xc != 'PBEsol' or step not in POLICY['steps']: raise ValueError((xc,step))
    source = w.ROOT / xc / 'recovery-01' / step
    root = w.ROOT / xc / POLICY['revision']
    root.mkdir(exist_ok=True)
    target = root / step
    target.mkdir()  # Never overwrite a previous attempt.
    resources = POLICY['dense_kmesh' if step == 'check-kmesh' else 'standard']
    state = {'xc':xc,'step':step,'job_id':os.environ['SLURM_JOB_ID'],'state':'preparing',
             'source':str(source),'resources':resources,'cumulative_budget':None}
    def status(value):
        state['state']=value
        w.save(target / 'status.json',state)
        print(json.dumps(state),flush=True)
    try:
        expected = json.loads((source / 'input-hashes.json').read_text())
        for name, sha in expected.items():
            if w.digest(source/name) != sha: raise RuntimeError('Source input hash mismatch: '+name)
        for name in ['POSCAR','POTCAR','KPOINTS','potcar-provenance.json']:
            shutil.copyfile(source/name,target/name)
        text=(source/'INCAR').read_text()
        for tag,value in [('KPAR',resources['KPAR']),('NCORE',resources['NCORE']),('NBANDS',576),('LCHARG','.TRUE.')]:
            pat=rf'(?m)^\s*{tag}\s*=.*$'
            text=re.sub(pat,f'{tag} = {value}',text) if re.search(pat,text) else text+f'{tag} = {value}\n'
        (target/'INCAR').write_text(text)
        hashes={name:w.digest(target/name) for name in ['POSCAR','POTCAR','KPOINTS','INCAR']}
        w.save(target/'input-hashes.json',hashes)
        w.save(target/'provenance.json',{'source_hashes':expected,'input_hashes':hashes,
               'policy_sha256':w.digest(w.ROOT/'execution-policy-03.json'),
               'driver_sha256':w.digest(Path(__file__)),
               'changes':['KPAR=8','64 ranks on 2 nodes; 32 per node','NBANDS=576 explicitly preserves previous band count','LCHARG=True for restart']})
        status('running')
        start=time.monotonic()
        job = os.environ['SLURM_JOB_ID']
        info = subprocess.check_output(['scontrol','show','job','-o',job],text=True)
        end = re.search(r'EndTime=(\S+)',info).group(1)
        deadline = subprocess.check_output(['date','-d',end,'+%s'],text=True).strip()
        watchdog = subprocess.Popen([sys.executable,str(w.ROOT/'checkpoint_guard_v2.py'),str(target),job,deadline])
        try:
            with (target/'vasp.stdout').open('w') as out:
                subprocess.run(['srun','--kill-on-bad-exit=1','--wait=60',
                                '--ntasks='+str(resources['ranks']),'vasp_std'],
                               cwd=target,stdout=out,stderr=subprocess.STDOUT,check=True)
        finally:
            if watchdog.poll() is None:
                watchdog.terminate()
                watchdog.wait()
        text = (target/'OUTCAR').read_text(errors='replace')
        if 'aborting loop because EDIFF is reached' not in text:
            status('checkpointed_needs_review' if (target/'checkpoint-request.json').exists() else 'unconverged_needs_review')
            return
        result,quality=w.validate(target)
        state['wall_seconds']=time.monotonic()-start
        state['quality']=quality
        status('validated')
        w.save(target/'output-manifest.json',{p.name:w.digest(p) for p in target.iterdir() if p.is_file()})
    except BaseException:
        state['error']=traceback.format_exc()
        status('failed')
        raise

if __name__=='__main__': main(*sys.argv[1:])
