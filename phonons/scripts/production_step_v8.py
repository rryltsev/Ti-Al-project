"""Second r2SCAN displacement seeded from independently accepted first displacement."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback
import numpy as np
from ase.calculators.vasp import VaspChargeDensity
from ase.io import read
import workflow as w
from wavecar_check import inspect_wavecar
from restart_step_v7 import quota
from restart_step_v2 import converged

POLICY=json.loads((w.ROOT/'execution-policy-08.json').read_text())

def main(xc,step):
    if xc!='r2SCAN' or step!='force-002': raise ValueError((xc,step))
    if os.environ.get('SLURM_JOB_PARTITION')!='apollo': raise RuntimeError('Apollo only; A100 forbidden')
    parent=POLICY['seed_job_id']
    seed=w.ROOT/POLICY['seed_source']
    frozen=w.ROOT/xc/'recovery-01'/step
    target=w.ROOT/xc/POLICY['revision']/step
    target.parent.mkdir(exist_ok=True)
    target.mkdir()
    state=dict(xc=xc,step=step,job_id=os.environ['SLURM_JOB_ID'],parent_job=parent,
               source=str(frozen),seed_source=str(seed),resources=POLICY['standard'])
    def status(value):
        state['state']=value
        w.save(target/'status.json',state)
        print(json.dumps(state),flush=True)
    try:
        status('checking_seed')
        rows=subprocess.check_output(['sacct','-X','-nP','-j',parent,'--format=JobIDRaw,State,ExitCode'],text=True)
        if not any(r.split('|')[:3]==[parent,'COMPLETED','0:0'] for r in rows.splitlines()):
            raise RuntimeError('Seed job not cleanly completed: '+rows)
        if not converged(seed) or json.loads((seed/'status.json').read_text())['state']!='validated':
            raise RuntimeError('Seed has no validated EDIFF convergence')
        if 'General timing and accounting informations' not in (seed/'OUTCAR').read_text():
            raise RuntimeError('Seed did not terminate normally')
        for folder in [seed,frozen]:
            for name,sha in json.loads((folder/'input-hashes.json').read_text()).items():
                if w.digest(folder/name)!=sha: raise RuntimeError('Changed input: '+str(folder/name))
        original=w.ROOT/xc/'recovery-01/force-001/POSCAR'
        if w.digest(seed/'POSCAR')!=w.digest(original): raise RuntimeError('Seed geometry mismatch')
        atoms=read(seed/'POSCAR')
        displaced=read(frozen/'POSCAR')
        if atoms.get_chemical_symbols()!=displaced.get_chemical_symbols() or not np.allclose(atoms.cell,displaced.cell,atol=1e-10,rtol=0):
            raise RuntimeError('Seed and new displacement must have identical cell/species')
        delta=displaced.get_scaled_positions(wrap=False)-atoms.get_scaled_positions(wrap=False)
        delta-=np.round(delta)
        norms=np.linalg.norm(delta@atoms.cell.array,axis=1)
        if abs(norms[0]-.02)>1e-8 or np.max(norms[1:])>1e-8:
            raise RuntimeError('Expected opposite .01A displacement of atom1')
        names=['CHGCAR','TAUCAR','WAVECAR']
        for name in names[:2]:
            density=VaspChargeDensity(str(seed/name))
            if len(density.chg)!=1 or not np.isfinite(density.chg[0]).all(): raise RuntimeError('Nonfinite seed density')
            if (density.atoms[0].get_chemical_symbols()!=atoms.get_chemical_symbols()
                or not np.allclose(density.atoms[0].cell,atoms.cell,atol=1e-6,rtol=0)
                or not np.allclose(density.atoms[0].positions,atoms.positions,atol=3e-5,rtol=0)):
                raise RuntimeError('Seed density geometry mismatch')
            if name=='CHGCAR':
                charge=float(density.chg[0].mean()*atoms.get_volume())
                if abs(charge-960)>.01 or not density.aug: raise RuntimeError('Charge/PAW mismatch')
                state['seed_electrons']=charge
        state['wave_header']=inspect_wavecar(seed/'WAVECAR',atoms.cell.tolist())
        density_bytes=sum((seed/n).stat().st_size for n in names[:2])
        needed=sum((seed/n).stat().st_size for n in names)+2*density_bytes+4*1024**3
        free,q=quota()
        state.update(quota_before_copy=q,required_additional_bytes=needed)
        if free<needed: raise RuntimeError('Insufficient quota for immutable seed copy')
        expected=json.loads((seed/'output-manifest.json').read_text())
        copied={}
        for name in names:
            sha=w.digest(seed/name)
            if sha!=expected[name]: raise RuntimeError('Changed converged seed: '+name)
            shutil.copyfile(seed/name,target/name)
            if w.digest(target/name)!=sha: raise RuntimeError('Seed copy hash mismatch: '+name)
            copied[name]=dict(sha256=sha,bytes=(target/name).stat().st_size)
        if quota()[0]<4*1024**3: raise RuntimeError('Quota headroom shrank')
        for name in ['POSCAR','POTCAR','KPOINTS','potcar-provenance.json']:
            shutil.copyfile(frozen/name,target/name)
            if w.digest(target/name)!=w.digest(frozen/name): raise RuntimeError('Frozen input copy mismatch')
        text=(frozen/'INCAR').read_text()
        for key,value in [('ICHARG',1),('ISTART',1),('KPAR',8),('NCORE',4),('NBANDS',576),('LCHARG','.TRUE.'),('LWAVE','.TRUE.')]:
            pat=rf'(?m)^\s*{key}\s*=.*$'
            text=re.sub(pat,f'{key} = {value}',text) if re.search(pat,text) else text+f'{key} = {value}\n'
        (target/'INCAR').write_text(text)
        w.save(target/'input-hashes.json',{n:w.digest(target/n) for n in ['INCAR','POSCAR','KPOINTS','POTCAR']})
        w.save(target/'restart-provenance.json',dict(parent_job=parent,source=str(seed),checkpoints=copied,
            frozen_source=str(frozen),driver_sha256=w.digest(Path(__file__)),policy_sha256=w.digest(w.ROOT/'execution-policy-08.json'),
            changes=['Opposite displacement from frozen phonopy inputs; same scientific settings',
                     'ISTART1/ICHARG1: converged first displacement as initial guess; self-consistency retained',
                     '64 ranks KPAR8 NCORE4 NBANDS576; LWAVE/LCHARG enabled']))
        job=state['job_id']
        info=subprocess.check_output(['scontrol','show','job','-o',job],text=True)
        deadline=subprocess.check_output(['date','-d',re.search(r'EndTime=(\S+)',info).group(1),'+%s'],text=True).strip()
        guard=subprocess.Popen([sys.executable,str(w.ROOT/'checkpoint_guard_v2.py'),str(target),job,deadline])
        status('running')
        try:
            with (target/'vasp.stdout').open('w') as out:
                process=subprocess.Popen(['srun','--kill-on-bad-exit=1','--wait=60','--ntasks=64','vasp_std'],cwd=target,stdout=out,stderr=subprocess.STDOUT)
                while True:
                    try:
                        code=process.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        log=(target/'vasp.stdout').read_text(errors='replace')
                        if 'WAVECAR not read' in log or 'WAVECAR not re' in log:
                            process.terminate()
                            try: process.wait(timeout=90)
                            except subprocess.TimeoutExpired: process.kill();process.wait()
                            raise RuntimeError('WAVECAR restart rejected')
                if code: raise RuntimeError('VASP/srun failed: '+str(code))
        finally:
            if guard.poll() is None: guard.terminate();guard.wait()
        if 'General timing and accounting informations' not in (target/'OUTCAR').read_text(errors='replace'):
            raise RuntimeError('No normal termination')
        w.save(target/'output-manifest.json',{n:w.digest(target/n) for n in ['OUTCAR','vasprun.xml','CONTCAR']+names})
        if not converged(target):
            status('checkpointed_needs_review' if (target/'checkpoint-request.json').exists() else 'unconverged_needs_review')
            return
        _,state['quality']=w.validate(target)
        status('validated')
    except BaseException:
        state['error']=traceback.format_exc()
        status('failed_needs_review')
        raise

if __name__=='__main__': main(*sys.argv[1:])
