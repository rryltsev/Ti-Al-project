"""One unchanged foundation model, all fixed Test50 geometries, no DFT fitting."""
from pathlib import Path
import json
import os
import platform
import time
import traceback
import importlib.metadata
import numpy as np
import torch
from deepmd.infer import DeepPot
from predict import sha

assert os.environ.get('SLURM_JOB_ID'), 'Compute node required'
registry=json.loads(Path('pretrained_registry.json').read_text())
assert registry['status']=='frozen' and sha(registry['path'])==registry['sha256']
bundle=json.loads(Path('bundle.json').read_text())
state=torch.load(registry['path'],map_location='cpu',weights_only=True)
extra=state.get('model',state).get('_extra_state',{})
Path('checkpoint-metadata.json').write_text(json.dumps(extra,indent=2)+'\n')
del state
fine=json.loads(Path('finetuned-metadata.json').read_text())['model_params']
base=extra['model_params']
assert base['type'] in ['SeZM','dpa4'] and fine['type']=='dpa4'
assert base['type_map']==fine['type_map']
architecture_keys=['channels','n_radial','lmax','mmax','n_blocks','mixing_layers','rcut','n_focus','n_atten_head']
architecture={k:{'pretrained':base['descriptor'][k],'finetuned':fine['descriptor'][k]} for k in architecture_keys}
assert all(v['pretrained']==v['finetuned'] for v in architecture.values()), architecture
result={'status':'running','job_id':os.environ['SLURM_JOB_ID'],'model_sha256':registry['sha256'],
        'architecture_comparison':architecture,'unix':time.time()}
def save():Path('result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
save()
try:
    dp=DeepPot(registry['path']);names=dp.get_type_map()
    assert names==base['type_map'] and 'Al' in names and 'Ti' in names
    assert dp.get_dim_fparam()==dp.get_dim_aparam()==0
    versions={p:importlib.metadata.version(p) for p in ['numpy','deepmd-kit','torch']}
    versions['python']=platform.python_version()
    output=Path('predictions/pretrained');output.mkdir(parents=True)
    for r in bundle['records']:
        meta=r['metadata'];sid=meta['structure_id'];cell=np.asarray(r['cell_A'])
        coords=np.asarray(r['fractional_positions'])@cell;atypes=[names.index(s) for s in r['symbols']]
        e,f,v=dp.eval(coords.reshape(1,-1),cell.reshape(1,9),atypes)[:3]
        assert all(np.isfinite(x).all() for x in [e,f,v])
        record={'structure_id':sid,'model_role':'pretrained','model_sha256':registry['sha256'],
                'poscar_sha256':meta['poscar_sha256'],'symbols':r['symbols'],'type_map':names,
                'energy_eV':float(e.ravel()[0]),'forces_eV_A':f.reshape(-1,3).tolist(),
                'virial_eV':v.reshape(3,3).tolist(),'energy_offset_applied_eV':0.,
                'job_id':os.environ['SLURM_JOB_ID'],'versions':versions,'unix':time.time(),
                'script_sha256':sha(__file__),'bundle_sha256':sha('bundle.json')}
        (output/(sid+'.json')).write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        print(sid,'predicted',flush=True)
    r=bundle['records'][0];cell=np.asarray(r['cell_A']);coords=np.asarray(r['fractional_positions'])@cell
    atypes=[names.index(s) for s in r['symbols']]
    e,f,v=dp.eval(coords.reshape(1,-1),cell.reshape(1,9),atypes)[:3]
    atom,axis=divmod(int(np.argmax(np.abs(f))),3);force=float(f.reshape(-1,3)[atom,axis]);virial=float(v.reshape(3,3)[0,0])
    fd=[]
    for h in [.003,.001,.0003]:
        ef=[];ev=[]
        for sign in [-1,1]:
            moved=coords.copy();moved[atom,axis]+=sign*h
            ef.append(float(dp.eval(moved.reshape(1,-1),cell.reshape(1,9),atypes)[0].ravel()[0]))
            strain=np.eye(3);strain[0,0]+=sign*h
            ev.append(float(dp.eval((coords@strain).reshape(1,-1),(cell@strain).reshape(1,9),atypes)[0].ravel()[0]))
        fd.append({'step':h,'force_eV_A':-(ef[1]-ef[0])/(2*h),'virial_xx_eV':-(ev[1]-ev[0])/(2*h)})
    result['derivative_check']={'force_eV_A':force,'virial_xx_eV':virial,'finite_differences':fd};save()
    assert sum(abs(d['force_eV_A']-force)<.01 for d in fd)>=2,fd
    assert sum(abs(d['virial_xx_eV']-virial)<.05 for d in fd)>=2,fd
    result.update(status='passed',predictions=len(bundle['records']),versions=versions,finished_unix=time.time());save()
except Exception:
    result.update(status='failed',traceback=traceback.format_exc());save();raise
paths=list(output.glob('*.json'))+[Path('result.json'),Path('checkpoint-metadata.json'),Path('pretrained_registry.json')]
Path('outputs.json').write_text(json.dumps({p.as_posix():sha(p) for p in paths},indent=2)+'\n')
