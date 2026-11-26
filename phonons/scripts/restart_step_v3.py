"""Relocate the PBE CPU job from A100 to Apollo using its clean density checkpoint."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
from ase.calculators.vasp import VaspChargeDensity
from ase.io import read
import workflow as w


def converged(folder):
    text = (folder / 'OUTCAR').read_text(errors='replace')
    return ('aborting loop because EDIFF is reached' in text and
            'General timing and accounting informations' in text)


def main(xc, step, parent_job):
    if (xc, step, parent_job) != ('PBE', 'force-001', '20959336'):
        raise ValueError('This immutable relocation driver is scoped to PBE 20959336')
    allocation = subprocess.check_output(['scontrol','show','job','-o',os.environ['SLURM_JOB_ID']],text=True)
    if 'Partition=apollo ' not in allocation:
        raise RuntimeError('CPU VASP must run on Apollo; never occupy the scarce A100 host')
    source = w.ROOT / xc / 'production-02' / (step + '-restart-01')
    target = source.with_name(step + '-restart-02')
    target.mkdir()  # Immutable attempt, never replace an earlier continuation.
    state = dict(xc=xc, step=step, job_id=os.environ['SLURM_JOB_ID'],
                 parent_job=parent_job, source=str(source), state='checking_checkpoint')
    def status(value):
        state['state'] = value
        w.save(target / 'status.json', state)
    try:
        state['scheduler_allocation'] = allocation
        status('checking_checkpoint')
        scheduler = subprocess.check_output(['sacct', '-X', '-nP', '-j', parent_job,
                                             '--format=JobIDRaw,State'], text=True)
        states = [s.split('|')[1] for s in scheduler.splitlines() if s.startswith(parent_job + '|')]
        if len(states) != 1 or states[0] not in ('COMPLETED', 'TIMEOUT'):
            raise RuntimeError('Parent requires diagnosis, not automatic retry: ' + scheduler)
        state['parent_scheduler'] = scheduler
        for name, sha in json.loads((source / 'input-hashes.json').read_text()).items():
            if w.digest(source / name) != sha:
                raise RuntimeError('Parent input changed: ' + name)
        if converged(source):
            w.validate(source)
            status('parent_already_converged')
            return
        if not (source / 'checkpoint-request.json').exists():
            raise RuntimeError('No planned checkpoint request; investigate before restart')
        if 'General timing and accounting informations' not in (source / 'OUTCAR').read_text(errors='replace'):
            raise RuntimeError('No clean checkpoint termination; preserve evidence, do not restart blindly')
        # CHGCAR coordinates are printed at six decimal places by this VASP build.
        # Accept only rounding-scale errors; POSCAR itself remains hash-identical.
        atoms = read(source / 'POSCAR')
        names = ['CHGCAR'] + (['TAUCAR'] if xc == 'r2SCAN' else [])
        checkpoints = {}
        for name in names:
            density = VaspChargeDensity(str(source / name))
            if len(density.chg) != 1 or not np.isfinite(density.chg[0]).all():
                raise RuntimeError('Invalid density: ' + name)
            if (not np.allclose(density.atoms[0].cell, atoms.cell, atol=1e-6, rtol=0) or
                not np.allclose(density.atoms[0].positions, atoms.positions, atol=3e-5, rtol=0) or
                density.atoms[0].get_chemical_symbols() != atoms.get_chemical_symbols()):
                raise RuntimeError('Checkpoint structure mismatch')
            if name == 'CHGCAR':
                electrons = float(density.chg[0].mean() * atoms.get_volume())
                if abs(electrons - 10 * len(atoms)) > 0.01 or not density.aug:
                    raise RuntimeError('Charge integral or PAW occupancies invalid')
                state['checkpoint_electrons'] = electrons
            checkpoints[name] = dict(sha256=w.digest(source / name), bytes=(source / name).stat().st_size)
        # Reserve additional space against the actual 100 GiB user quota.
        quota = subprocess.check_output(['quota', '-w'], text=True)
        state['quota_before_copy'] = quota
        quota_line = next(s for s in quota.splitlines() if ':/home6' in s)
        numbers = quota_line.split()[1:4]
        used, soft, hard = [int(s.rstrip('*')) * 1024 for s in numbers]
        needed = 3 * sum(d['bytes'] for d in checkpoints.values()) + 1024**3
        if hard - used < needed:
            raise RuntimeError('Insufficient user quota for durable restart')
        for name in ['POSCAR', 'POTCAR', 'KPOINTS', 'potcar-provenance.json'] + names:
            shutil.copyfile(source / name, target / name)
            if w.digest(source / name) != w.digest(target / name):
                raise RuntimeError('Copy hash mismatch: ' + name)
        text = (source / 'INCAR').read_text()
        text = re.sub(r'(?m)^ICHARG\s*=.*$', 'ICHARG = 1', text)
        (target / 'INCAR').write_text(text)
        w.save(target / 'input-hashes.json', {n:w.digest(target/n) for n in ['INCAR','POSCAR','KPOINTS','POTCAR']})
        w.save(target / 'restart-provenance.json', dict(parent_job=parent_job, source=str(source),
               checkpoints=checkpoints, changes=['ICHARG=1; ISTART=0 (no parent WAVECAR)'],
               driver_sha256=w.digest(Path(__file__))))
        job = os.environ['SLURM_JOB_ID']
        info = subprocess.check_output(['scontrol','show','job','-o',job],text=True)
        end = re.search(r'EndTime=(\S+)',info).group(1)
        deadline = subprocess.check_output(['date','-d',end,'+%s'],text=True).strip()
        watchdog = subprocess.Popen([sys.executable,str(w.ROOT/'checkpoint_guard_v2.py'),str(target),job,deadline])
        status('running')
        try:
            with (target/'vasp.stdout').open('w') as output:
                subprocess.run(['srun','--kill-on-bad-exit=1','--wait=60','--ntasks=32','vasp_std'],
                               cwd=target,stdout=output,stderr=subprocess.STDOUT,check=True)
        finally:
            if watchdog.poll() is None:
                watchdog.terminate()
                watchdog.wait()
        if not converged(target):
            status('checkpointed_needs_review' if (target/'checkpoint-request.json').exists() else 'unconverged_needs_review')
            return
        _, quality = w.validate(target)
        state['quality'] = quality
        status('validated')
        w.save(target/'output-manifest.json',{p.name:w.digest(p) for p in target.iterdir() if p.is_file()})
    except BaseException:
        state['error'] = traceback.format_exc()
        status('failed_needs_review')
        raise


if __name__ == '__main__':
    main(*sys.argv[1:])
