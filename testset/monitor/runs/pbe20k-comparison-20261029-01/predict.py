"""Evaluate pinned article models on a downloaded common-DFT bundle on a Slurm node."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bundle', type=Path)
    parser.add_argument('registry', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--functional', choices=['PBE','PBEsol','r2SCAN'])
    args = parser.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Run inference on a Slurm compute node, not on the login node.'
    registry = json.loads(args.registry.read_text())
    assert registry['status'] == 'frozen', 'Accessible article model paths and hashes must be frozen first.'
    bundle = json.loads(args.bundle.read_text())
    from deepmd.infer import DeepPot
    versions = {p: importlib.metadata.version(p) for p in ['numpy','deepmd-kit']}
    versions['python'] = platform.python_version()
    for xc in ([args.functional] if args.functional else ['PBE','PBEsol','r2SCAN']):
        model = registry['models'][xc]
        assert model['sha256'] and sha(model['path']) == model['sha256']
        dp = DeepPot(model['path'])
        names = dp.get_type_map()
        assert 'Al' in names and 'Ti' in names
        assert dp.get_dim_fparam() == dp.get_dim_aparam() == 0, 'Model requires additional parameters; do not guess them.'
        output = args.output / xc; output.mkdir(parents=True, exist_ok=True)
        for r in bundle['records']:
            metadata = r['metadata']; sid = metadata['structure_id']
            path = output / (sid + '.json')
            if path.exists():
                cached = json.loads(path.read_text())
                assert cached['model_sha256'] == model['sha256'] and cached['poscar_sha256'] == metadata['poscar_sha256']
                assert cached['symbols'] == r['symbols']
                continue
            cell = np.array(r['cell_A'],dtype=float)
            positions = np.array(r['fractional_positions'],dtype=float) @ cell
            atypes = [names.index(e) for e in r['symbols']]
            e,f,v = dp.eval(positions.reshape(1,-1),cell.reshape(1,9),atypes)[:3]
            energy = float(np.asarray(e).reshape(-1)[0])
            forces = np.asarray(f).reshape(len(atypes),3)
            virial = np.asarray(v).reshape(3,3)
            assert np.isfinite(energy) and np.isfinite(forces).all() and np.isfinite(virial).all()
            record = {'structure_id':sid,'functional':xc,'model_sha256':model['sha256'],
                      'poscar_sha256':metadata['poscar_sha256'],'symbols':r['symbols'],'type_map':names,
                      'energy_eV':energy,'forces_eV_A':forces.tolist(),'virial_eV':virial.tolist(),
                      'versions':versions,'unix':time.time(),'job_id':os.environ['SLURM_JOB_ID'],
                      'script_sha256':sha(__file__),'bundle_sha256':sha(args.bundle),
                      'energy_offset_applied_eV':0.0}
            temporary=path.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
            temporary.replace(path)
            print(xc,sid,'predicted',flush=True)
        del dp


if __name__ == '__main__':
    main()
