"""hcp Ti 15k: independent a,c relaxation and 9^3 phonons, native CPU only."""
from pathlib import Path
import os,json,hashlib,time,importlib.metadata
import numpy as np
from scipy.optimize import least_squares
from ase.io import read,write
from ase.phonons import Phonons
from ase.calculators.calculator import Calculator,all_changes
from ase.units import GPa
from deepmd.infer import DeepPot
import torch

assert os.environ.get('SLURM_JOB_ID') and not os.environ.get('CUDA_VISIBLE_DEVICES')
ROOT=Path.cwd();cfg=json.loads((ROOT/'protocol.json').read_text());key='15000'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
model=cfg['models'][key]
assert sha(model['path'])==model['sha256'] and sha(ROOT/'common.cif')==cfg['common_sha256']
state=torch.load(model['path'],map_location='cpu',weights_only=True)
info=state['model']['_extra_state']['train_infos'];assert info['step']==14999
del state
out=ROOT/key;out.mkdir(exist_ok=False);start=time.monotonic()
dp=DeepPot(model['path']);ti=dp.get_type_map().index('Ti');timings=[]
class Native(Calculator):
    implemented_properties=['energy','forces','stress']
    def calculate(self,atoms=None,properties=['energy'],system_changes=all_changes):
        super().calculate(atoms,properties,system_changes);ts=time.monotonic()
        e,f,v=dp.eval(atoms.positions.reshape(1,-1),atoms.cell.array.reshape(1,9),[ti]*len(atoms))[:3]
        timings.append({'n_atoms':len(atoms),'seconds':time.monotonic()-ts})
        st=-v.reshape(3,3)/atoms.get_volume();st=(st+st.T)/2
        self.results={'energy':float(e.reshape(-1)[0]),'forces':f.reshape(-1,3),'stress':st.flat[[0,4,8,5,2,1]]}
        assert all(np.isfinite(v).all() for v in self.results.values())
calc=Native();common=read(ROOT/'common.cif');assert len(common)==2
common.set_masses([47.867]*2);common.calc=calc
basecell=common.cell.array.copy();basefrac=common.get_scaled_positions()
def geometry(x):
    at=common.copy();cell=basecell.copy();cell[:2]*=np.exp(x[0]);cell[2]*=np.exp(x[1])
    at.set_cell(cell);at.set_scaled_positions(basefrac);at.calc=calc
    return at
def residual(x):
    s=geometry(x).get_stress()/GPa
    return np.array([(s[0]+s[1])/2,s[2]])
def jacobian(x):
    h=1e-4
    return np.column_stack([(residual(x+np.eye(2)[i]*h)-residual(x-np.eye(2)[i]*h))/(2*h) for i in range(2)])
def describe(at):
    cp=at.cell.cellpar()
    return {'a_A':float(cp[0]),'c_A':float(cp[2]),'c_over_a':float(cp[2]/cp[0]),
      'V_per_atom_A3':at.get_volume()/len(at),'cell_A':at.cell.array.tolist(),
      'fractional_positions':at.get_scaled_positions().tolist(),
      'energy_eV_atom':at.get_potential_energy()/len(at),'stress_GPa':(at.get_stress()/GPa).tolist(),
      'max_force_eV_A':float(np.linalg.norm(at.get_forces(),axis=1).max())}
def replication(at,desc):
    rep=at.repeat((4,4,4));rep.calc=calc;d=describe(rep)
    v={'energy_difference_meV_atom':1000*(d['energy_eV_atom']-desc['energy_eV_atom']),
       'stress_max_difference_GPa':float(np.max(abs(np.array(d['stress_GPa'])-desc['stress_GPa'])))}
    assert abs(v['energy_difference_meV_atom'])<.02 and v['stress_max_difference_GPa']<.005
    return v
common_desc=describe(common);equiv={'common':replication(common,common_desc)}
opt=least_squares(residual,np.zeros(2),jac=jacobian,bounds=(-.08,.08),xtol=1e-10,ftol=1e-10,gtol=1e-8,max_nfev=60)
at=geometry(opt.x);desc=describe(at)
assert opt.success and np.max(abs(np.array(desc['stress_GPa'])))<.005 and desc['max_force_eV_A']<1e-4
j=jacobian(opt.x);hess=at.get_volume()*GPa*np.diag([2.,1.])@j
eig=np.linalg.eigvalsh((hess+hess.T)/2);assert eig.min()>0
equiv['relaxed']=replication(at,desc)
write(out/'POSCAR-relaxed',at,format='vasp',direct=True)
summary={'checkpoint':key,'model_sha256':model['sha256'],'train_infos':info,'common':common_desc,'relaxed':desc,
 'small_cell_replication':equiv,'optimization':{'success':bool(opt.success),'message':str(opt.message),'nfev':opt.nfev,
 'log_strain':opt.x.tolist(),'hcp_strain_hessian_eigenvalues_eV':eig.tolist()},
 'supercell':[9,9,9],'displacement_A':.01,'mass_amu':47.867,'relaxed_POSCAR_sha256':sha(out/'POSCAR-relaxed')}
(out/'progress.json').write_text(json.dumps(summary,indent=2));print('RELAXED',key,desc,flush=True)
path=common.cell.bandpath('GKMGA',npoints=200);qaxis,ticks,labels=path.get_linear_kpoint_axis()
cell=at.cell.array.copy();frac=at.get_scaled_positions().copy()
ph=Phonons(at,calc,supercell=(9,9,9),delta=.01,name=str(out/'forces'));ph.run()
arrays={'qpoints':path.kpts,'qaxis':qaxis,'ticks':ticks}
ph.read(method='standard',symmetrize=0,acoustic=False);arrays['C_raw']=ph.C_N.copy()
arrays['bands_raw_THz']=ph.band_structure(path.kpts,verbose=False)*241.799
ph.read(method='Frederiksen',symmetrize=3,acoustic=True)
arrays.update(C=ph.C_N.copy(),D=ph.D_N.copy(),R=ph.compute_lattice_vectors(),bands_THz=ph.band_structure(path.kpts,verbose=False)*241.799)
assert np.array_equal(cell,at.cell.array) and np.array_equal(frac,at.get_scaled_positions())
assert len(list((out/'forces').glob('cache.*.json')))==13
np.savez_compressed(out/'results.npz',**arrays)
summary.update(status='passed',geometry_unchanged_during_phonons=True,script_sha256=sha(__file__),protocol_sha256=sha(ROOT/'protocol.json'),
 results_sha256=sha(out/'results.npz'),wall_seconds=time.monotonic()-start,job_id=os.environ['SLURM_JOB_ID'],node=os.uname().nodename,
 gamma_THz=arrays['bands_THz'][0].tolist(),min_band_THz=float(arrays['bands_THz'].min()),inference_timings=timings,
 versions={p:importlib.metadata.version(p) for p in ['numpy','scipy','ase','torch','deepmd-kit']})
(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False));print('COMPLETE',key,summary['wall_seconds'],flush=True)
