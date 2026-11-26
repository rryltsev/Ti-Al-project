"""Resume verification of already-created, immutable exports; no refreezing."""
from pathlib import Path
import json
import os
import sys
import time
import traceback
import numpy as np
from deepmd.infer import DeepPot
from export_models import sha, save

assert os.environ.get('SLURM_JOB_ID')
source = Path(sys.argv[1])
registry = json.loads((source / 'model_registry.json').read_text())
bundle = json.loads(Path('bundle.json').read_text())
result = {'status':'running','job_id':os.environ['SLURM_JOB_ID'],
          'export_run':str(source),'models':{},'started_unix':time.time()}
save('model_registry.json',registry)
try:
    for xc, model in registry['models'].items():
        assert sha(model['path']) == model['sha256']
        assert sha(model['checkpoint_path']) == model['checkpoint_sha256']
        before = DeepPot(model['checkpoint_path']); after = DeepPot(model['path'])
        names = after.get_type_map()
        assert names == before.get_type_map() and 'Al' in names and 'Ti' in names
        assert after.get_dim_fparam() == after.get_dim_aparam() == 0
        errors = []
        for r in bundle['records']:
            cell = np.asarray(r['cell_A']); coords = np.asarray(r['fractional_positions']) @ cell
            atypes = [names.index(s) for s in r['symbols']]
            a = before.eval(coords.reshape(1,-1),cell.reshape(1,9),atypes)[:3]
            b = after.eval(coords.reshape(1,-1),cell.reshape(1,9),atypes)[:3]
            diffs = [float(np.max(np.abs(x-y))) for x,y in zip(a,b)]
            assert all(np.isfinite(x).all() for x in a+b)
            errors.append({'structure_id':r['metadata']['structure_id'],'max_abs_E_F_V':diffs,
                           'energy_error_per_atom_eV':diffs[0]/len(atypes)})
        result['models'][xc] = {'type_map':names,'equivalence':errors}
        save('result.json',result)
        # Check both signs with finite differences at several step sizes to
        # expose single-precision roundoff and finite-step truncation.
        r = bundle['records'][0]
        cell = np.asarray(r['cell_A']);coords=np.asarray(r['fractional_positions'])@cell
        atypes=[names.index(s) for s in r['symbols']]
        e,f,v=after.eval(coords.reshape(1,-1),cell.reshape(1,9),atypes)[:3]
        atom,axis=divmod(int(np.argmax(np.abs(f))),3)
        force=float(f.reshape(-1,3)[atom,axis]);virial=float(v.reshape(3,3)[0,0])
        fd=[]
        for h in [0.003,0.001,0.0003]:
            ef=[];ev=[]
            for sign in [-1,1]:
                moved=coords.copy();moved[atom,axis]+=sign*h
                ef.append(float(after.eval(moved.reshape(1,-1),cell.reshape(1,9),atypes)[0].ravel()[0]))
                strain=np.eye(3);strain[0,0]+=sign*h
                ev.append(float(after.eval((coords@strain).reshape(1,-1),(cell@strain).reshape(1,9),atypes)[0].ravel()[0]))
            fd.append({'step':h,'force_eV_A':-(ef[1]-ef[0])/(2*h),'virial_xx_eV':-(ev[1]-ev[0])/(2*h)})
        result['models'][xc]['derivative_check']={'atom':atom,'axis':axis,'force_eV_A':force,
            'virial_xx_eV':virial,'finite_differences':fd,'stress_convention':'ASE tensile-positive = -virial/volume'}
        save('result.json',result)
        assert all(d['energy_error_per_atom_eV'] < 1e-6 and d['max_abs_E_F_V'][1] < 1e-5
                   and d['max_abs_E_F_V'][2] < 1e-4 for d in errors), errors
        assert sum(abs(d['force_eV_A']-force)<.01 for d in fd)>=2, fd
        assert sum(abs(d['virial_xx_eV']-virial)<.05 for d in fd)>=2, fd
        result['models'][xc]['status']='verified'
        save('result.json',result)
        print(xc,'verified',flush=True)
        del before,after
    result.update(status='passed',finished_unix=time.time())
    save('result.json',result)
except Exception:
    result.update(status='failed',traceback=traceback.format_exc())
    save('result.json',result)
    raise
paths=[Path('result.json'),Path('model_registry.json')]
for xc in registry['models']:
    Path(xc).mkdir(exist_ok=True)
    for name in ['model_frozen.pt2','checkpoint-metadata.json','freeze.log']:
        target=Path(xc)/name
        target.symlink_to(source/xc/name)
        paths.append(target)
save('outputs.json',{p.as_posix():sha(p) for p in paths})
