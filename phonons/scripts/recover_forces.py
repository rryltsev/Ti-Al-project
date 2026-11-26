"""Continue from checked equilibrium cells, preserving the failed attempt."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import ase
import numpy as np
import workflow as w

def main(xc):
    if xc not in w.C['functionals']: raise ValueError(xc)
    parent = w.ROOT / xc
    root = parent / 'recovery-01'
    root.mkdir()
    state = {'xc': xc, 'job_id': os.environ['SLURM_JOB_ID'], 'state': 'running',
             'reason': 'Original supercell step OOM; resume with 192 GiB and fail-fast MPI.',
             'scientific_protocol_changed': False}
    start = time.monotonic()
    def progress(step):
        state['step'] = step
        w.save(root / 'status.json', state)
        print(xc, step, flush=True)
    try:
        # Verify equilibrium output hashes and independent geometry/force/stress parsing again.
        expected = json.loads((parent / 'equilibrium-02/validation.json').read_text())
        for name, sha in expected['outputs'].items():
            if w.digest(parent / 'equilibrium-02' / name) != sha:
                raise RuntimeError('Equilibrium output hash mismatch')
        atoms, eq = w.validate(parent / 'equilibrium-02')
        if eq['max_force_eV_A'] > w.C['relax_max_force_eV_A'] or eq['max_stress_kbar'] > w.C['relax_max_stress_kbar']:
            raise RuntimeError('Equilibrium quality gate failed')
        ph = w.phonon_object(parent / 'POSCAR-relaxed')
        if ph.symmetry.dataset.number != 194: raise RuntimeError('Wrong symmetry')
        shutil.copyfile(parent / 'phonopy_disp.yaml', root / 'phonopy_disp.yaml')
        def to_ase(cell):
            return ase.Atoms(symbols=cell.symbols, cell=cell.cell, scaled_positions=cell.scaled_positions, pbc=True)
        tasks = [(f'force-{i:03d}', to_ase(cell), w.C['force_kmesh'], {})
                 for i, cell in enumerate(ph.supercells_with_displacements, 1)]
        tasks.append(('force-000', to_ase(ph.supercell), w.C['force_kmesh'], {}))
        first = to_ase(ph.supercells_with_displacements[0])
        tasks.extend([
            ('check-cutoff', first, w.C['force_kmesh'], {'encut': w.C['diagnostics']['encut_eV']}),
            ('check-kmesh', first, w.C['diagnostics']['kmesh'], {}),
            ('check-smearing', first, w.C['force_kmesh'], {'sigma': w.C['diagnostics']['sigma_eV']})])
        for name, cell, mesh, options in tasks:
            w.inputs(root / name, cell, xc, mesh, **options)
        w.save(root / 'input-manifest.json', {str(p.relative_to(root)): w.digest(p)
               for p in root.rglob('*') if p.is_file()})
        forces = {}
        for name, cell, mesh, options in tasks:
            # Keep valid completed steps if the bounded allocation cannot finish the series.
            if time.monotonic() - start > 70 * 60:
                state['state'] = 'partial_allocation_limit'
                progress('waiting_for_continuation_before_' + name)
                return
            progress(name)
            with (root / name / 'vasp.stdout').open('w') as out:
                subprocess.run(['srun', '--kill-on-bad-exit=1', '--wait=60', '--ntasks=16', 'vasp_std'],
                               cwd=root / name, stdout=out, stderr=subprocess.STDOUT, check=True)
            result, quality = w.validate(root / name)
            forces[name] = result.get_forces()
            if 'force-001' in forces and 'force-002' in forces:
                np.save(root / 'displaced-forces-raw.npy', np.array([forces['force-001'], forces['force-002']]))
        diagnostics = []
        for name in ['check-cutoff', 'check-kmesh', 'check-smearing']:
            delta = float(np.linalg.norm(forces[name] - forces['force-001'], axis=1).max())
            diagnostics.append({'name': name, 'max_force_difference_eV_A': delta,
                                'within_target': delta <= w.C['diagnostic_force_tolerance_eV_A']})
        w.save(root / 'electronic-diagnostics.json', diagnostics)
        state['state'] = 'completed_raw_forces'
        state['electronic_diagnostics_pass'] = all(d['within_target'] for d in diagnostics)
        state['publication_qualified'] = False
        progress('ready_for_analysis_and_supercell_convergence_review')
        w.save(root / 'output-manifest.json', {str(p.relative_to(root)): w.digest(p)
               for p in root.rglob('*') if p.is_file()})
    except BaseException:
        state['state'] = 'failed'
        state['error'] = traceback.format_exc()
        w.save(root / 'status.json', state)
        raise

if __name__ == '__main__': main(sys.argv[1])
