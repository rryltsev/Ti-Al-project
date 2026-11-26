"""PBE retry after node failure and r2SCAN continuation from its improved orbital checkpoint."""
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
from restart_step_v2 import converged

PLANS = {
    'PBE': ('20959753','force-001-restart-03','force-001-restart-05'),
    'r2SCAN': ('20964090','force-001-restart-03','force-001-restart-04'),
}


def quota():
    text=subprocess.check_output(['quota','-w'],text=True)
    row=next(s for s in text.splitlines() if ':/home6' in s)
    used,soft,hard=[int(s.rstrip('*'))*1024 for s in row.split()[1:4]]
    return hard-used,text


def main(xc,step,parent):
    if xc not in PLANS or step!='force-001' or parent!=PLANS[xc][0]:
        raise ValueError('Unreviewed restart plan')
    if os.environ.get('SLURM_JOB_PARTITION')!='apollo':
        raise RuntimeError('VASP must run on Apollo; A100 forbidden')
    source=w.ROOT/xc/'production-02'/PLANS[xc][1]
    target=source.with_name(PLANS[xc][2])
    target.mkdir()
    job=os.environ['SLURM_JOB_ID']
    state=dict(xc=xc,step=step,job_id=job,parent_job=parent,source=str(source),
               resources=dict(ranks=64,nodes=2,ranks_per_node=32,KPAR=8,NCORE=4,NBANDS=576,memory_GiB_per_node=240))
    def status(value):
        state['state']=value
        w.save(target/'status.json',state)
        print(json.dumps(state),flush=True)
    try:
        status('checking_checkpoint')
        scheduler=subprocess.check_output(['sacct','-X','-nP','-j',parent,'--format=JobIDRaw,State'],text=True)
        if not any(row.split('|')[:2]==[parent,'COMPLETED'] for row in scheduler.splitlines()):
            raise RuntimeError('Parent did not complete cleanly: '+scheduler)
        state['parent_scheduler']=scheduler
        for name,sha in json.loads((source/'input-hashes.json').read_text()).items():
            if w.digest(source/name)!=sha:raise RuntimeError('Changed parent input: '+name)
        if converged(source):
            w.validate(source)
            status('parent_already_converged')
            return
        if not (source/'checkpoint-request.json').exists() or 'General timing and accounting informations' not in (source/'OUTCAR').read_text(errors='replace'):
            raise RuntimeError('No verified planned checkpoint termination')
        atoms=read(source/'POSCAR')
        names=['CHGCAR']+(['TAUCAR'] if xc=='r2SCAN' else [])
        for name in names:
            density=VaspChargeDensity(str(source/name))
            if len(density.chg)!=1 or not np.isfinite(density.chg[0]).all():raise RuntimeError('Invalid density '+name)
            if (not np.allclose(density.atoms[0].cell,atoms.cell,atol=1e-6,rtol=0) or
                not np.allclose(density.atoms[0].positions,atoms.positions,atol=3e-5,rtol=0) or
                density.atoms[0].get_chemical_symbols()!=atoms.get_chemical_symbols()):
                raise RuntimeError('Density geometry mismatch')
            if name=='CHGCAR':
                charge=float(density.chg[0].mean()*atoms.get_volume())
                if abs(charge-10*len(atoms))>.01 or not density.aug:raise RuntimeError('Charge/PAW mismatch')
                state['checkpoint_electrons']=charge
        density_bytes=sum((source/n).stat().st_size for n in names)
        if xc=='r2SCAN':
            state['source_wavecar']=inspect_wavecar(source/'WAVECAR',atoms.cell.tolist())
            names.append('WAVECAR')
        needed=sum((source/n).stat().st_size for n in names)+2*density_bytes+4*1024**3
        free,quota_text=quota()
        state.update(quota_before_copy=quota_text,required_additional_bytes=needed)
        if free<needed:raise RuntimeError('Insufficient quota for immutable checkpoint copy and output')
        checkpoints={}
        for name in names:
            sha=w.digest(source/name)
            shutil.copyfile(source/name,target/name)
            if w.digest(target/name)!=sha:raise RuntimeError('Checkpoint copy hash mismatch: '+name)
            checkpoints[name]=dict(sha256=sha,bytes=(target/name).stat().st_size)
        if quota()[0]<4*1024**3:raise RuntimeError('Quota headroom shrank during checkpoint copy')
        for name in ['POSCAR','KPOINTS','POTCAR','potcar-provenance.json']:
            shutil.copyfile(source/name,target/name)
            if w.digest(target/name)!=w.digest(source/name):raise RuntimeError('Input copy mismatch')
        text=(source/'INCAR').read_text()
        for key,value in [('ICHARG',1),('ISTART',1 if xc=='r2SCAN' else 0),('KPAR',8),('NCORE',4),('NBANDS',576),('LCHARG','.TRUE.'),('LWAVE','.TRUE.' if xc=='r2SCAN' else '.FALSE.')]:
            pat=rf'(?m)^\s*{key}\s*=.*$'
            text=re.sub(pat,f'{key} = {value}',text) if re.search(pat,text) else text+f'{key} = {value}\n'
        (target/'INCAR').write_text(text)
        w.save(target/'input-hashes.json',{n:w.digest(target/n) for n in ['INCAR','POSCAR','KPOINTS','POTCAR']})
        w.save(target/'restart-provenance.json',dict(parent_job=parent,source=str(source),checkpoints=checkpoints,
               driver_sha256=w.digest(Path(__file__)),changes=['64 ranks, KPAR8, NCORE4; scientific settings unchanged',
               'r2SCAN ISTART1/ICHARG1 with WAVECAR+CHGCAR+TAUCAR' if xc=='r2SCAN' else 'PBE ISTART0/ICHARG1; density-only restart with faster resource layout']))
        info=subprocess.check_output(['scontrol','show','job','-o',job],text=True)
        state['scheduler_allocation']=info
        end=re.search(r'EndTime=(\S+)',info).group(1)
        deadline=subprocess.check_output(['date','-d',end,'+%s'],text=True).strip()
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
                        if xc=='r2SCAN' and ('WAVECAR not read' in log or 'WAVECAR not re' in log):
                            process.terminate()
                            try:process.wait(timeout=90)
                            except subprocess.TimeoutExpired:
                                process.kill();process.wait()
                            raise RuntimeError('WAVECAR restart rejected; stop instead of repeating fresh SCF')
                if code:raise RuntimeError('VASP/srun failed: '+str(code))
        finally:
            if guard.poll() is None:
                guard.terminate();guard.wait()
        if 'General timing and accounting informations' not in (target/'OUTCAR').read_text(errors='replace'):
            raise RuntimeError('No clean termination')
        w.save(target/'output-manifest.json',{n:w.digest(target/n) for n in ['OUTCAR','vasprun.xml','CONTCAR']+names if (target/n).exists()})
        if not converged(target):
            status('checkpointed_needs_review' if (target/'checkpoint-request.json').exists() else 'unconverged_needs_review')
            return
        _,state['quality']=w.validate(target)
        status('validated')
    except BaseException:
        state['error']=traceback.format_exc()
        status('failed_needs_review')
        raise


if __name__=='__main__':main(*sys.argv[1:])
