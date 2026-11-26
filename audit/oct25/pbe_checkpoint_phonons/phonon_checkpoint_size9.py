"""Additional size convergence check, using already optimized geometries."""
from pathlib import Path
import os,json,hashlib,time,importlib.metadata
import numpy as np
from ase.io import read
from ase.phonons import Phonons
from ase.calculators.calculator import Calculator,all_changes
from deepmd.infer import DeepPot
ROOT=Path.cwd();cfg=json.loads((ROOT/'protocol.json').read_text());key=cfg['checkpoints'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert os.environ.get('SLURM_JOB_ID') and not os.environ.get('CUDA_VISIBLE_DEVICES')
m=cfg['models'][key];assert sha(m['path'])==m['sha256']
previous=json.loads((ROOT/key/'summary.json').read_text());assert previous['status']=='passed'
out=ROOT/('size9-'+key);out.mkdir(exist_ok=False);start=time.monotonic();dp=DeepPot(m['path']);ti=dp.get_type_map().index('Ti')
class Native(Calculator):
    implemented_properties=['forces']
    def calculate(self,atoms=None,properties=['forces'],system_changes=all_changes):
        super().calculate(atoms,properties,system_changes)
        e,f,v=dp.eval(atoms.positions.reshape(1,-1),atoms.cell.array.reshape(1,9),[ti]*len(atoms))[:3]
        self.results={'forces':f.reshape(-1,3)};assert np.isfinite(f).all()
calc=Native();common=read(ROOT/'common.cif');path=common.cell.bandpath('GKMGA',npoints=200)
arrays={};summary={'checkpoint':key,'model_sha256':m['sha256'],'cases':{},'source_geometries':{}}
for g in ['common','relaxed']:
    p=ROOT/key/('POSCAR-'+g);summary['source_geometries'][g]=sha(p)
    at=read(p,format='vasp');at.set_masses([47.867]*2);ts=time.monotonic()
    ph=Phonons(at,calc,supercell=(9,9,9),delta=.01,name=str(out/g));ph.run()
    ph.read(method='standard',symmetrize=0,acoustic=False);arrays[g+'-C-raw']=ph.C_N.copy()
    ph.read(method='Frederiksen',symmetrize=3,acoustic=True)
    arrays[g+'-C']=ph.C_N.copy();arrays[g+'-D']=ph.D_N.copy();arrays[g+'-R']=ph.compute_lattice_vectors()
    bands=ph.band_structure(path.kpts,verbose=False)*241.799;arrays[g+'-bands']=bands
    summary['cases'][g]={'wall_seconds':time.monotonic()-ts,'min_frequency_THz':float(bands.min()),'max_frequency_THz':float(bands.max()),'gamma_THz':bands[0].tolist()}
    (out/'progress.json').write_text(json.dumps(summary,indent=2));print('DONE',key,g,summary['cases'][g],flush=True)
np.savez_compressed(out/'results.npz',**arrays)
summary.update(status='passed',script_sha256=sha(__file__),protocol_sha256=sha(ROOT/'protocol.json'),results_sha256=sha(out/'results.npz'),
               wall_seconds=time.monotonic()-start,job_id=os.environ['SLURM_JOB_ID'],node=os.uname().nodename)
(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False));print('COMPLETE',key,flush=True)
