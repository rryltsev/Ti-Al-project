"""Evaluate only the explicitly selected PBE 15k checkpoint on all fixed Test50 cells."""
from pathlib import Path
import hashlib
import json
import os
import time
import numpy as np
import torch
from deepmd.infer import DeepPot
from predict import main as predict


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def save(p,v): Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


assert os.environ.get('SLURM_JOB_ID'), 'Compute allocation required'
cfg=read('inference-input.json')
assert sha(cfg['checkpoint']['path'])==cfg['checkpoint']['sha256']
assert sha('all50-geometries.json')==cfg['geometry_sha256']
state=torch.load(cfg['checkpoint']['path'],map_location='cpu',weights_only=True)
extra=state.get('model',state)['_extra_state']
assert extra['train_infos']['step']==14999,extra['train_infos']
save('checkpoint-metadata.json',extra)
del state
registry=read('registry-template.json')
save('model_registry.json',registry)
bundle=read('all50-geometries.json');assert len(bundle['records'])==50
dp=DeepPot(cfg['checkpoint']['path']);names=dp.get_type_map()
assert names.index('Al')==12 and names.index('Ti')==21
assert dp.get_dim_fparam()==dp.get_dim_aparam()==0
r=bundle['records'][0]
cell=np.asarray(r['cell_A']);coords=np.asarray(r['fractional_positions'])@cell
types=[names.index(s) for s in r['symbols']]
e,f,v=dp.eval(coords.reshape(1,-1),cell.reshape(1,9),types)[:3]
atom,axis=divmod(int(np.argmax(np.abs(f))),3)
force=float(f.reshape(-1,3)[atom,axis]);virial=float(v.reshape(3,3)[0,0])
fd=[]
for h in (.003,.001,.0003):
    ef=[];ev=[]
    for sign in (-1,1):
        moved=coords.copy();moved[atom,axis]+=sign*h
        ef.append(float(dp.eval(moved.reshape(1,-1),cell.reshape(1,9),types)[0].ravel()[0]))
        strain=np.eye(3);strain[0,0]+=sign*h
        ev.append(float(dp.eval((coords@strain).reshape(1,-1),(cell@strain).reshape(1,9),types)[0].ravel()[0]))
    fd.append({'h':h,'force_eV_A':-(ef[1]-ef[0])/(2*h),'virial_xx_eV':-(ev[1]-ev[0])/(2*h)})
assert sum(abs(x['force_eV_A']-force)<.01 for x in fd)>=2,fd
assert sum(abs(x['virial_xx_eV']-virial)<.05 for x in fd)>=2,fd
save('verification.json',{'status':'passed','checkpoint_step':15000,'stored_train_step':14999,
    'checkpoint_sha256':cfg['checkpoint']['sha256'],'type_map':names,
    'force_eV_A':force,'virial_xx_eV':virial,'finite_differences':fd,
    'geometry_sha256':cfg['geometry_sha256'],'job_id':os.environ['SLURM_JOB_ID'],
    'inference_artifact':'original .pt checkpoint directly; no export or weight changes',
    'unix':time.time()})
del dp
import sys
sys.argv=['predict.py','all50-geometries.json','model_registry.json','predictions','--functional','PBE']
predict()
files=list(Path('predictions/PBE').glob('*.json'))
assert len(files)==50
save('outputs.json',{str(p):sha(p) for p in files+[Path('verification.json'),Path('model_registry.json'),Path('checkpoint-metadata.json')]})
print('PASS: checkpoint 15k and all 50 predictions verified',flush=True)
