"""Native CPU checkpoint evaluation; all predictions archived, no fitting."""
from pathlib import Path
import hashlib,json,os,time,sys,importlib.metadata
import numpy as np
import torch
from deepmd.infer import DeepPot
assert os.environ.get('SLURM_JOB_ID')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=json.loads(Path('remote-input.json').read_text());mode=sys.argv[1]
task=int(os.environ['SLURM_ARRAY_TASK_ID']);keys=list(cfg['models'])
if mode=='pilot':key=['5000','23400'][task];indices=cfg['pilot_indices'];suffix='pilot-'+key
else:key=keys[task//8];indices=list(range(task%8,cfg['total'],8));suffix='result-'+str(task)
model=cfg['models'][key];path=model['path']
assert sha(path)==model['sha256'] and sha('geometries.npz')==cfg['geometry_sha256']
assert not Path(suffix+'.json').exists() and not Path(suffix+'.npz').exists()
state=torch.load(path,map_location='cpu',weights_only=True);extra=state.get('model',state).get('_extra_state',{})
assert extra['train_infos']['step']==int(key)-1,extra.get('train_infos')
del state
start=time.monotonic();dp=DeepPot(path);startup=time.monotonic()-start
names=dp.get_type_map();assert dp.get_dim_fparam()==dp.get_dim_aparam()==0
data=np.load('geometries.npz',allow_pickle=False);out={};timings=[]
for i in indices:
    coords=data[f'coords_{i}'];cell=data[f'cell_{i}'];types=[names.index(['Al','Ti'][int(t)]) for t in data[f'types_{i}']]
    ts=time.monotonic();e,f,v=dp.eval(coords.reshape(1,-1),cell.reshape(1,9),types)[:3];timings.append(time.monotonic()-ts)
    assert all(np.isfinite(a).all() for a in [e,f,v])
    out[f'energy_{i}']=e.reshape(());out[f'forces_{i}']=f.reshape(-1,3);out[f'virial_{i}']=v.reshape(3,3)
    print(key,i,timings[-1],flush=True)
np.savez_compressed(suffix+'.npz',**out)
result={'status':'passed','checkpoint':key,'model_sha256':model['sha256'],'input_sha256':sha('remote-input.json'),
    'geometries_sha256':cfg['geometry_sha256'],'predictions_sha256':sha(suffix+'.npz'),
    'script_sha256':sha(__file__),'train_infos':extra['train_infos'],'indices':indices,
    'startup_seconds':startup,'inference_seconds':timings,'total_wall_seconds':time.monotonic()-start,
    'job_id':os.environ['SLURM_JOB_ID'],'task_id':task,'node':os.uname().nodename,
    'versions':{p:importlib.metadata.version(p) for p in ['numpy','torch','deepmd-kit']},'type_indices':{s:names.index(s) for s in ['Al','Ti']}}
Path(suffix+'.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps({k:result[k] for k in ['status','checkpoint','startup_seconds','total_wall_seconds','train_infos']}))
