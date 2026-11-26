"""CPU inference of the unchanged article PBE model on its 850 train frames."""
from pathlib import Path
import hashlib, json, os, time, importlib.metadata
import numpy as np
from deepmd.infer import DeepPot
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=json.loads(Path('inference-input.json').read_text())
assert os.environ.get('SLURM_JOB_ID')
assert sha(cfg['geometries_path'])==cfg['geometries_sha256']
assert sha(cfg['model_path'])==cfg['model_sha256']
data=np.load(cfg['geometries_path'],allow_pickle=False)
task=int(os.environ['SLURM_ARRAY_TASK_ID'])
dp=DeepPot(cfg['model_path']); names=dp.get_type_map()
assert dp.get_dim_fparam()==dp.get_dim_aparam()==0
energies={}
for i in cfg['geometry_ids'][task::8]:
    coords=data[f'coords_{i}']; cell=data[f'cell_{i}']; types=data[f'types_{i}']
    atypes=[names.index(['Al','Ti'][int(t)]) for t in types]
    e,f,v=dp.eval(coords.reshape(1,-1),cell.reshape(1,9),atypes)[:3]
    assert all(np.isfinite(x).all() for x in [e,f,v])
    energies[str(i)]=float(e.ravel()[0])
    print(i,energies[str(i)],flush=True)
result={'status':'passed','energies_eV':energies,'model_sha256':cfg['model_sha256'],
        'geometries_sha256':cfg['geometries_sha256'],'script_sha256':sha(__file__),
        'input_sha256':sha('inference-input.json'),'job_id':os.environ['SLURM_JOB_ID'],
        'task':task,'chunks':8,'unix':time.time(),
        'versions':{p:importlib.metadata.version(p) for p in ['numpy','deepmd-kit','torch']}}
out=Path(f'result-{task}.json')
with out.open('x') as stream: json.dump(result,stream,indent=2,allow_nan=False)
Path(f'result-{task}.sha256').write_text(sha(out))
