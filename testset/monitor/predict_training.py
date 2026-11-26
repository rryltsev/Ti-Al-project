"""Compute-node-only inference on audited training geometries, without labels."""
from pathlib import Path
import hashlib, json, os, time, platform, importlib.metadata
import numpy as np
from deepmd.infer import DeepPot

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())

assert os.environ.get('SLURM_JOB_ID')
registry=read('pretrained_registry.json'); audit=read('input-audit.json')
assert sha(registry['path'])==registry['sha256']
assert sha('training-geometries.npz')==audit['geometries_sha256']
task=int(os.environ['SLURM_ARRAY_TASK_ID']); chunks=8
data=np.load('training-geometries.npz',allow_pickle=False)
count=len(data.files)//3
dp=DeepPot(registry['path']); names=dp.get_type_map()
assert dp.get_dim_fparam()==dp.get_dim_aparam()==0
energies={}
for i in range(task,count,chunks):
    coords=data[f'coords_{i}']; cell=data[f'cell_{i}']; types=data[f'types_{i}']
    atypes=[names.index(['Al','Ti'][int(t)]) for t in types]
    e,f,v=dp.eval(coords.reshape(1,-1),cell.reshape(1,9),atypes)[:3]
    assert all(np.isfinite(x).all() for x in [e,f,v])
    energies[str(i)]=float(e.ravel()[0])
    print(i,energies[str(i)],flush=True)
result={'status':'passed','energies_eV':energies,'model_sha256':registry['sha256'],
        'geometries_sha256':audit['geometries_sha256'],'script_sha256':sha(__file__),
        'job_id':os.environ['SLURM_JOB_ID'],'task':task,'chunks':chunks,'total_geometries':count,
        'versions':{p:importlib.metadata.version(p) for p in ['numpy','deepmd-kit','torch']},
        'python':platform.python_version(),'unix':time.time()}
out=Path(f'result-{task}.json')
with out.open('x') as stream: json.dump(result,stream,indent=2,allow_nan=False)
Path(f'result-{task}.sha256').write_text(sha(out))
