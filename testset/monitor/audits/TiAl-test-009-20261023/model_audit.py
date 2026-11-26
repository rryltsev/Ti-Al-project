"""Independent inference on original POSCAR, without changing models or cache."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import time
import numpy as np

assert os.environ.get('SLURM_JOB_ID'), 'Compute-node Slurm allocation required'

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def save(value):
    Path('model-result.json').write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

registry = json.loads(Path('model_registry.json').read_text())
model = registry['models']['PBE']
lines = Path('POSCAR').read_text().splitlines()
assert float(lines[1]) == 1 and lines[7].lower().startswith('d')
cell = np.array([list(map(float, x.split())) for x in lines[2:5]])
symbols = [s for s, n in zip(lines[5].split(), map(int, lines[6].split())) for _ in range(n)]
frac = np.array([list(map(float, x.split()[:3])) for x in lines[8:8 + len(symbols)]])
positions = frac @ cell
cache = json.loads(Path('cached.json').read_text())
assert cache['symbols'] == symbols and cache['poscar_sha256'] == sha('POSCAR')
assert cache['model_sha256'] == model['sha256']
result = {'status': 'running', 'job_id': os.environ['SLURM_JOB_ID'], 'started_unix': time.time(),
          'structure_id': cache['structure_id'], 'poscar_sha256': sha('POSCAR'),
          'script_sha256': sha(__file__), 'registry_sha256': sha('model_registry.json'),
          'cached_sha256': sha('cached.json'), 'models': {},
          'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'deepmd-kit', 'torch']}}
save(result)
from deepmd.infer import DeepPot
for role, path, expected in [('checkpoint', model['checkpoint_path'], model['checkpoint_sha256']),
                             ('frozen', model['path'], model['sha256'])]:
    assert sha(path) == expected
    dp = DeepPot(path)
    names = dp.get_type_map()
    types = [names.index(s) for s in symbols]
    assert dp.get_dim_fparam() == dp.get_dim_aparam() == 0
    def evaluate(coords, atom_types=types):
        e, f, v = dp.eval(coords.reshape(1, -1), cell.reshape(1, 9), atom_types)[:3]
        return float(e.ravel()[0]), f.reshape(-1, 3), v.reshape(3, 3)
    energy, forces, virial = evaluate(positions)
    result['models'][role] = {'path': path, 'sha256': expected, 'type_indices': {s: names.index(s) for s in set(symbols)},
        'energy_eV': energy, 'forces_eV_A': forces.tolist(), 'virial_eV': virial.tolist(),
        'cache_difference': {'energy_eV': energy - cache['energy_eV'],
            'force_max_eV_A': float(np.max(np.abs(forces - np.array(cache['forces_eV_A'])))),
            'virial_max_eV': float(np.max(np.abs(virial - np.array(cache['virial_eV']))))}}
    assert abs(energy - cache['energy_eV']) < 0.001
    assert np.max(np.abs(forces - np.array(cache['forces_eV_A']))) < 1e-4
    save(result)
    if role == 'frozen':
        permutation = np.random.default_rng(409).permutation(len(symbols))
        ep, fp, vp = evaluate(positions[permutation], [types[i] for i in permutation])
        translated = ((frac + np.array([0.237, -0.119, 0.341])) % 1) @ cell
        et, ft, vt = evaluate(translated)
        result['invariance'] = {
            'permutation_energy_difference_eV': ep - energy,
            'permutation_force_max_eV_A': float(np.max(np.abs(fp - forces[permutation]))),
            'translation_energy_difference_eV': et - energy,
            'translation_force_max_eV_A': float(np.max(np.abs(ft - forces)))}
        result['finite_difference'] = []
        atom, axis = np.unravel_index(np.argmax(np.abs(forces)), forces.shape)
        for h in [0.003, 0.001]:
            energies = []
            for sign in [-1, 1]:
                moved = positions.copy()
                moved[atom, axis] += sign * h
                energies.append(evaluate(moved)[0])
            fd = -(energies[1] - energies[0]) / (2 * h)
            result['finite_difference'].append({'atom': int(atom), 'axis': int(axis), 'h_A': h,
                'analytic_force_eV_A': float(forces[atom, axis]), 'numeric_force_eV_A': fd,
                'difference_eV_A': fd - float(forces[atom, axis])})
        assert abs(ep - energy) < 0.001 and abs(et - energy) < 0.001
        assert max(abs(x['difference_eV_A']) for x in result['finite_difference']) < 0.01
    del dp
result.update(status='passed', finished_unix=time.time())
save(result)
print(json.dumps({'status': result['status'], 'job_id': result['job_id'],
                  'energies': {k: v['energy_eV'] for k, v in result['models'].items()}}), flush=True)
