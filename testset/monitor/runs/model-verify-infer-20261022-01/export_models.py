"""Export user-specified checkpoints, verify numerical equivalence, then infer.

Run only within the dedicated Slurm run directory. Never trains or fits offsets.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

ROOT = Path.cwd()
XC = ['PBE', 'PBEsol', 'r2SCAN']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def main():
    assert os.environ.get('SLURM_JOB_ID'), 'Compute node required'
    import numpy as np
    import torch
    import importlib.metadata
    from deepmd.infer import DeepPot
    source = json.loads(Path('source-manifest.json').read_text())
    bundle = json.loads(Path('bundle.json').read_text())
    registry = json.loads(Path('registry-template.json').read_text())
    result = {'status': 'running', 'job_id': os.environ['SLURM_JOB_ID'],
              'started_unix': time.time(), 'versions': {'torch': torch.__version__,
              'deepmd-kit': importlib.metadata.version('deepmd-kit'), 'numpy': np.__version__},
              'models': {}, 'source_manifest_sha256': sha('source-manifest.json')}
    save('result.json', result)
    try:
        for xc in XC:
            checkpoint = ROOT / xc / 'checkpoint.pt'
            assert sha(checkpoint) == source['models'][xc]['sha256']
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            extra = state.get('model', state).get('_extra_state', {})
            save(ROOT / xc / 'checkpoint-metadata.json', extra)
            del state
            target = ROOT / xc / 'model_frozen.pt2'
            assert not target.exists(), 'Immutable export already exists'
            with (ROOT / xc / 'freeze.log').open('w') as log:
                subprocess.run([str(Path(sys.prefix) / 'bin/dp'), '--pt', 'freeze',
                                '-c', str(checkpoint), '-o', str(target.with_suffix(''))],
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
            assert target.is_file()
            model = registry['models'][xc]
            model.update(path=str(target), sha256=sha(target),
                         checkpoint_sha256=sha(checkpoint), checkpoint_path=str(checkpoint))
            result['models'][xc] = {'export_sha256': sha(target), 'status': 'exported'}
            save('result.json', result)
            print(xc, 'exported', flush=True)
        # Pin all exports before any DFT-error analysis or choice of checkpoint.
        registry.update(status='frozen', frozen_unix=time.time(),
                        source_manifest_sha256=sha('source-manifest.json'))
        save('model_registry.json', registry)
        for xc in XC:
            before = DeepPot(str(ROOT / xc / 'checkpoint.pt'))
            after = DeepPot(registry['models'][xc]['path'])
            assert before.get_type_map() == after.get_type_map()
            names = after.get_type_map()
            # Fine-tuned DPA4 retains the foundation model's full element map.
            # Resolve Al/Ti by name; do not assume binary indices 0 and 1.
            assert 'Al' in names and 'Ti' in names and len(names) == len(set(names)), names
            assert after.get_dim_fparam() == after.get_dim_aparam() == 0
            errors = []
            for r in bundle['records']:
                cell = np.asarray(r['cell_A'])
                coords = np.asarray(r['fractional_positions']) @ cell
                atypes = [names.index(s) for s in r['symbols']]
                args = (coords.reshape(1,-1), cell.reshape(1,9), atypes)
                a = before.eval(*args)[:3]; b = after.eval(*args)[:3]
                diffs = [float(np.max(np.abs(x-y))) for x,y in zip(a,b)]
                assert diffs[0]/len(atypes) < 1e-6 and diffs[1] < 1e-5 and diffs[2] < 1e-4, diffs
                assert all(np.isfinite(v).all() for v in b)
                errors.append({'structure_id': r['metadata']['structure_id'],
                               'max_abs_E_F_V': diffs})
            # Independent derivative/sign checks on the first fixed geometry.
            r = bundle['records'][0]
            cell = np.asarray(r['cell_A']); coords = np.asarray(r['fractional_positions']) @ cell
            atypes = [names.index(s) for s in r['symbols']]
            e,f,v = after.eval(coords.reshape(1,-1),cell.reshape(1,9),atypes)[:3]
            h = 1e-3
            energies = []
            component = int(np.argmax(np.abs(f)))
            atom, axis = divmod(component, 3)
            for sign in [-1,1]:
                moved = coords.copy(); moved[atom,axis] += sign*h
                energies.append(float(after.eval(moved.reshape(1,-1),cell.reshape(1,9),atypes)[0].ravel()[0]))
            fd_force = -(energies[1]-energies[0])/(2*h)
            force = float(f.reshape(-1,3)[atom,axis])
            energies = []
            for sign in [-1,1]:
                strain = np.eye(3); strain[0,0] += sign*h
                energies.append(float(after.eval((coords@strain).reshape(1,-1),(cell@strain).reshape(1,9),atypes)[0].ravel()[0]))
            fd_virial = -(energies[1]-energies[0])/(2*h)
            virial = float(v.reshape(3,3)[0,0])
            assert abs(fd_force-force) < .01, (fd_force, force)
            assert abs(fd_virial-virial) < .05, (fd_virial, virial)
            result['models'][xc].update(status='verified', type_map=names, equivalence=errors,
                derivative_check={'step':h,'atom':atom,'axis':axis,'force_eV_A':force,
                'force_finite_difference_eV_A':fd_force,'virial_xx_eV':virial,
                'virial_xx_finite_difference_eV':fd_virial,'stress_convention':'ASE tensile-positive = -virial/volume'})
            save('result.json', result)
            print(xc, 'equivalence and derivative checks passed', flush=True)
            del before, after
        subprocess.run([sys.executable,'predict.py','bundle.json','model_registry.json','predictions'],check=True)
        result.update(status='passed', finished_unix=time.time())
        save('result.json', result)
    except Exception:
        result.update(status='failed', traceback=traceback.format_exc())
        save('result.json', result)
        raise
    files = [p for p in ROOT.rglob('*') if p.is_file() and
             ('predictions' in p.parts or p.name in ['model_registry.json','model_frozen.pt2',
              'result.json','checkpoint-metadata.json','freeze.log'])]
    save('outputs.json', {p.relative_to(ROOT).as_posix():sha(p) for p in files})


if __name__ == '__main__':
    main()
