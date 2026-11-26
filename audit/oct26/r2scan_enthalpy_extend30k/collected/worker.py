"""One allocated A100; same r2SCAN potential; continue each archived final frame."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import traceback
import numpy as np
import torch
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.melchionna import MelchionnaNPT
from deepmd.infer import DeepPot
from deepmd.pt.utils import env as dp_env

ROOT = Path.cwd()
cfg = json.loads(Path('protocol.json').read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert os.environ.get('SLURM_JOB_ID')
assert sha('baseline.json') == cfg['baseline_sha256']
assert sha(cfg['model']['path']) == cfg['model']['sha256']
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '4')))
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def save(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)

class Native(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']
    def __init__(self):
        super().__init__()
        model = torch.load(cfg['model']['path'], map_location='cpu', weights_only=True)
        assert model['model']['_extra_state']['train_infos']['step'] == cfg['model']['step']-1
        self.dp = DeepPot(cfg['model']['path'])
        self.names = self.dp.get_type_map()

    def calculate(self, atoms=None, properties=['forces'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        e, f, v = self.dp.eval(atoms.positions.reshape(1, -1), atoms.cell.array.reshape(1, 9),
                  [self.names.index(s) for s in atoms.get_chemical_symbols()])[:3]
        e, f, v = float(e.reshape(-1)[0]), f.reshape(-1, 3), v.reshape(3, 3)
        stress = -(v+v.T)/(2*atoms.get_volume())
        assert np.isfinite(e) and np.isfinite(f).all() and np.isfinite(stress).all()
        self.results = dict(energy=e, forces=f, stress=stress.flat[[0, 4, 8, 5, 2, 1]])

def initial(state, calc):
    assert sha(state['initial_file']) == state['initial_sha256']
    a = read(state['initial_file'])
    assert len(a) == cfg['n_atoms'] and a.has('momenta')
    assert np.max(abs(np.tril(a.cell.array, -1))) < 1e-12
    a.calc = calc
    return a

def predictions(a):
    return a.get_potential_energy(), a.get_forces(), a.get_stress()

if '--cpu-reference' in sys.argv:
    assert dp_env.DEVICE.type == 'cpu'
    calc, arrays = Native(), {}
    for i, state in enumerate(cfg['states']):
        e, f, s = predictions(initial(state, calc))
        arrays[f'E_{i}'], arrays[f'F_{i}'], arrays[f'S_{i}'] = e, f, s
    np.savez_compressed('cpu_reference.npz', **arrays)
    sys.exit(0)

assert torch.cuda.is_available() and dp_env.DEVICE.type == 'cuda'
assert torch.cuda.device_count() == 1
assert 'A100' in torch.cuda.get_device_name(0)
device = dict(name=torch.cuda.get_device_name(0), CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
              deepmd_device=str(dp_env.DEVICE))
print('GPU_WORKER', device, flush=True)
subprocess.run([sys.executable, __file__, '--cpu-reference'],
               env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'DEVICE': 'cpu'}, check=True)
reference, calc, checks = np.load('cpu_reference.npz'), Native(), []
for i, state in enumerate(cfg['states']):
    a = initial(state, calc)
    e, f, s = predictions(a)
    old = read(state['initial_file'])
    oe, of, ostress = predictions(old)
    check = dict(composition=state['composition'],
                 cpu_gpu_E_eV_atom=abs(e-float(reference[f'E_{i}']))/len(a),
                 cpu_gpu_F_rms_eV_A=float(np.sqrt(np.mean((f-reference[f'F_{i}'])**2))),
                 cpu_gpu_stress_max_GPa=float(np.max(abs(s-reference[f'S_{i}']))/units.GPa),
                 archive_E_eV_atom=abs(e-oe)/len(a),
                 archive_F_rms_eV_A=float(np.sqrt(np.mean((f-of)**2))),
                 archive_stress_max_GPa=float(np.max(abs(s-ostress))/units.GPa))
    assert check['cpu_gpu_E_eV_atom'] < 1e-5 and check['cpu_gpu_F_rms_eV_A'] < 1e-4 and check['cpu_gpu_stress_max_GPa'] < .002, check
    assert check['archive_E_eV_atom'] < 2e-5 and check['archive_F_rms_eV_A'] < 2e-4 and check['archive_stress_max_GPa'] < .005, check
    checks.append(check)
save('preflight.json', dict(status='passed', checks=checks, device=device,
     model_sha256=cfg['model']['sha256'], protocol_sha256=sha('protocol.json'),
     worker_sha256=sha(__file__), job_id=os.environ['SLURM_JOB_ID'],
     versions={p: importlib.metadata.version(p) for p in ['ase', 'deepmd-kit', 'torch', 'numpy']}))
print('PREFLIGHT_PASSED', flush=True)

def row(a, step):
    n, v = len(a), a.get_volume()
    e, k, temperature = a.get_potential_energy(), a.get_kinetic_energy(), a.get_temperature()
    pressure = -float(np.mean(a.get_stress(include_ideal_gas=True)[:3]))
    assert np.isfinite([v, e, k, temperature, pressure]).all()
    assert 10 < v/n < 40 and 200 < temperature < 6000, (step, v/n, temperature)
    return dict(step=step, production_step=30000+step, time_ps=step*.001,
                total_time_ps=30+step*.001, Epot_eV_atom=e/n, Ekin_eV_atom=k/n,
                H_external_eV_atom=(e+k+cfg['P_bar']*units.bar*v)/n,
                T_K=temperature, P_GPa=pressure/units.GPa, V_A3_atom=v/n,
                rho_g_cm3=float(a.get_masses().sum()/v*1.66053906660))

completed, failed = [], []
for state in cfg['states']:
    name = state['composition']
    out = ROOT/'results'/name
    out.mkdir(parents=True, exist_ok=False)
    try:
        a = initial(state, calc)
        before_positions, before_cell, momenta = a.positions.copy(), a.cell.array.copy(), a.get_momenta().copy()
        assert np.linalg.norm(momenta.sum(axis=0)) < 1e-6
        dyn = MelchionnaNPT(a, timestep=cfg['dt_fs']*units.fs, temperature_K=cfg['T_K'],
                 externalstress=cfg['P_bar']*units.bar, ttime=cfg['ttime_fs']*units.fs,
                 pfactor=(cfg['ttime_fs']*units.fs)**2*state['bulk_modulus_GPa']*units.GPa, logfile=None)
        dyn.set_fraction_traceless(0)
        # Constructor subtracts negligible residual center-of-mass momentum. Restore exact archived values.
        com_correction = float(np.max(abs(a.get_momenta()-momenta)))
        assert com_correction < 1e-8
        a.set_momenta(momenta)
        assert np.array_equal(a.positions, before_positions) and np.array_equal(a.cell.array, before_cell)
        assert np.array_equal(a.get_momenta(), momenta)
        save(out/'initialization.json', dict(exact_positions_cell_momenta=True,
             constructor_com_correction_restored=com_correction,
             thermostat_barostat_reset=True, source_frame=state['initial_frame'], additional_equilibration_steps=0))
        start = time.monotonic()
        print('START_COMPOSITION', name, flush=True)
        with (out/'production-thermo.jsonl').open('x') as fp, Trajectory(str(out/'production.traj'), 'w', a) as trajectory:
            def observe():
                r = row(a, dyn.nsteps)
                fp.write(json.dumps(r)+'\n')
                fp.flush()
                if dyn.nsteps % 1000 == 0:
                    save('progress.json', dict(status='running', active=name, step=dyn.nsteps,
                         completed=completed, failed=failed, job_id=os.environ['SLURM_JOB_ID']))
                    print(name, dyn.nsteps, r['T_K'], r['rho_g_cm3'], flush=True)
            dyn.attach(observe, interval=cfg['thermo_interval'])
            dyn.attach(trajectory.write, interval=cfg['trajectory_interval'])
            dyn.run(100)
            if not completed:
                elapsed = time.monotonic()-start
                save('timing.json', dict(seconds_first_100_steps=elapsed,
                     projected_series_hours=elapsed/100*cfg['additional_steps']*len(cfg['states'])/3600,
                     first_100_steps_included_in_production=True))
            dyn.run(cfg['additional_steps']-100)
        assert dyn.nsteps == 30000
        with Trajectory(str(out/'production.traj')) as trajectory:
            assert len(trajectory) == 301
        write(out/'production-final.traj', a)
        save(out/'summary.json', dict(status='completed', composition=name, steps=dyn.nsteps,
             wall_seconds=time.monotonic()-start, model_sha256=cfg['model']['sha256'],
             protocol_sha256=sha('protocol.json'), job_id=os.environ['SLURM_JOB_ID']))
        completed.append(name)
        print('COMPLETED_COMPOSITION', name, flush=True)
    except Exception:
        failure = dict(composition=name, traceback=traceback.format_exc())
        save(out/'failure.json', failure)
        failed.append(failure)
        print('FAILED_COMPOSITION', name, failure['traceback'], flush=True)
    save('progress.json', dict(status='running', completed=completed, failed=failed))
save('summary.json', dict(status='failed' if failed else 'completed', completed=completed, failed=failed))
if failed:
    sys.exit(1)
subprocess.run([sys.executable, 'summarize.py'], check=True)
