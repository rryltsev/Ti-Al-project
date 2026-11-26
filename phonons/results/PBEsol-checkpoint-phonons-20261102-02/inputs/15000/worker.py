"""CPU Slurm inference on the three exact DFT-PBEsol force geometries."""
from pathlib import Path
import hashlib,json,os,time,importlib.metadata
import numpy as np
import torch
from ase.io import read
from deepmd.infer import DeepPot

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

assert os.environ.get('SLURM_JOB_ID') and not os.environ.get('CUDA_VISIBLE_DEVICES')
cfg=json.loads(Path('protocol.json').read_text())
assert sha(cfg['model']['path'])==cfg['model']['sha256']
for name,value in cfg['input_hashes'].items(): assert sha(name)==value
state=torch.load(cfg['model']['path'],map_location='cpu',weights_only=True)
train=state['model']['_extra_state']['train_infos']
assert train['step']==cfg['model']['step']-1
del state
dp=DeepPot(cfg['model']['path']);names=dp.get_type_map()
assert dp.get_dim_fparam()==dp.get_dim_aparam()==0
out=Path('results');out.mkdir(exist_ok=False)
records={}
for key in ['force000','force001','force002']:
    at=read(key+'/POSCAR',format='vasp');assert len(at)==96
    assert set(at.get_chemical_symbols())=={'Ti'}
    start=time.monotonic()
    e,f,v=dp.eval(at.positions.reshape(1,-1),at.cell.array.reshape(1,9),[names.index(s) for s in at.get_chemical_symbols()])[:3]
    f=f.reshape(96,3);e=float(e.reshape(-1)[0]);v=v.reshape(3,3)
    assert np.isfinite(f).all() and np.isfinite(e) and np.isfinite(v).all()
    dest=out/key;dest.mkdir();np.save(dest/'forces.npy',f)
    stress=-(v+v.T)/(2*at.get_volume())
    records[key]=dict(energy_eV=e,max_force_eV_A=float(np.linalg.norm(f,axis=1).max()),
                     force_sum_eV_A=f.sum(axis=0).tolist(),stress_eV_A3=stress.tolist(),
                     seconds=time.monotonic()-start,POSCAR_sha256=sha(key+'/POSCAR'))
    print(key,records[key],flush=True)
summary=dict(status='passed',model=cfg['model'],train_infos=train,structures=records,
             protocol_sha256=sha('protocol.json'),worker_sha256=sha(__file__),
             job_id=os.environ['SLURM_JOB_ID'],node=os.uname().nodename,
             versions={n:importlib.metadata.version(n) for n in ['torch','deepmd-kit','ase','numpy']})
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(out/'manifest.json').write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},indent=2))
print('COMPLETE',flush=True)
