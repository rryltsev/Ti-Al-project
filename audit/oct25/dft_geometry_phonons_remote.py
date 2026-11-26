"""PBE checkpoint phonons at the exact archived DFT geometry, CPU only."""
from pathlib import Path
import os,json,hashlib,time,importlib.metadata
import numpy as np
from ase.io import read
from ase.phonons import Phonons
from ase.calculators.calculator import Calculator,all_changes
from ase.units import GPa
from deepmd.infer import DeepPot
import torch
ROOT=Path.cwd();cfg=json.loads((ROOT/'protocol.json').read_text())
key=cfg['checkpoints'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert os.environ.get('SLURM_JOB_ID') and not os.environ.get('CUDA_VISIBLE_DEVICES')
model=cfg['models'][key];assert sha(model['path'])==model['sha256']
assert sha(ROOT/'POSCAR-DFT')==cfg['geometry_sha256']
state=torch.load(model['path'],map_location='cpu',weights_only=True)
info=state['model']['_extra_state']['train_infos'];assert info['step']==int(key)-1
del state
out=ROOT/key;out.mkdir(exist_ok=False);start=time.monotonic();dp=DeepPot(model['path']);ti=dp.get_type_map().index('Ti');timings=[]
class Native(Calculator):
    implemented_properties=['energy','forces','stress']
    def calculate(self,atoms=None,properties=['forces'],system_changes=all_changes):
        super().calculate(atoms,properties,system_changes);t=time.monotonic()
        e,f,v=dp.eval(atoms.positions.reshape(1,-1),atoms.cell.array.reshape(1,9),[ti]*len(atoms))[:3]
        timings.append({'n_atoms':len(atoms),'seconds':time.monotonic()-t})
        st=-v.reshape(3,3)/atoms.get_volume();st=(st+st.T)/2
        self.results={'energy':float(e.reshape(-1)[0]),'forces':f.reshape(-1,3),'stress':st.flat[[0,4,8,5,2,1]]}
        assert all(np.isfinite(a).all() for a in self.results.values())
at=read(ROOT/'POSCAR-DFT',format='vasp');assert len(at)==2 and at.get_chemical_symbols()==['Ti','Ti']
at.set_masses([47.867]*2);initial_cell=at.cell.array.copy();initial_frac=at.get_scaled_positions().copy();at.calc=Native()
summary={'checkpoint':key,'model_sha256':model['sha256'],'train_infos':info,'geometry_sha256':sha(ROOT/'POSCAR-DFT'),
    'cell_A':at.cell.array.tolist(),'fractional_positions':initial_frac.tolist(),'cell_parameters':at.cell.cellpar().tolist(),
    'V_per_atom_A3':at.get_volume()/2,'energy_eV_atom':at.get_potential_energy()/2,
    'stress_GPa':(at.get_stress()/GPa).tolist(),'max_force_eV_A':float(np.linalg.norm(at.get_forces(),axis=1).max()),
    'relaxation_performed':False,'supercell':[9,9,9],'displacement_A':.01,'mass_amu':47.867}
assert summary['max_force_eV_A']<1e-4
(out/'progress.json').write_text(json.dumps(summary,indent=2));print('FIXED_GEOMETRY',key,summary,flush=True)
ph=Phonons(at,at.calc,supercell=(9,9,9),delta=.01,name=str(out/'forces'));ph.run()
path=at.cell.bandpath('GKMGA',npoints=200);qaxis,ticks,labels=path.get_linear_kpoint_axis()
arrays={'qpoints':path.kpts,'qaxis':qaxis,'ticks':ticks}
ph.read(method='standard',symmetrize=0,acoustic=False);arrays['C_raw']=ph.C_N.copy()
arrays['bands_raw_THz']=ph.band_structure(path.kpts,verbose=False)*241.799
ph.read(method='Frederiksen',symmetrize=3,acoustic=True)
arrays.update(C=ph.C_N.copy(),D=ph.D_N.copy(),R=ph.compute_lattice_vectors(),bands_THz=ph.band_structure(path.kpts,verbose=False)*241.799)
assert np.array_equal(at.cell.array,initial_cell) and np.array_equal(at.get_scaled_positions(),initial_frac)
assert len(list((out/'forces').glob('cache.*.json')))==13
np.savez_compressed(out/'results.npz',**arrays)
summary.update(status='passed',geometry_unchanged=True,script_sha256=sha(__file__),protocol_sha256=sha(ROOT/'protocol.json'),
    results_sha256=sha(out/'results.npz'),wall_seconds=time.monotonic()-start,job_id=os.environ['SLURM_JOB_ID'],node=os.uname().nodename,
    gamma_THz=arrays['bands_THz'][0].tolist(),min_band_THz=float(arrays['bands_THz'].min()),inference_timings=timings,
    versions={p:importlib.metadata.version(p) for p in ['numpy','ase','torch','deepmd-kit']})
(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False));print('COMPLETE',key,summary['wall_seconds'],flush=True)
