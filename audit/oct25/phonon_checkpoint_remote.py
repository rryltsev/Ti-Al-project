"""Harmonic hcp Ti checkpoint comparison, native CPU inference, no training."""
from pathlib import Path
import os, json, hashlib, time, importlib.metadata
import numpy as np
from scipy.optimize import least_squares
from ase import Atoms
from ase.io import read, write
from ase.phonons import Phonons
from ase.calculators.calculator import Calculator, all_changes
from ase.units import GPa
from deepmd.infer import DeepPot
import torch

assert os.environ.get('SLURM_JOB_ID') and not os.environ.get('CUDA_VISIBLE_DEVICES')
ROOT=Path.cwd(); cfg=json.loads((ROOT/'protocol.json').read_text())
key=cfg['checkpoints'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
model=cfg['models'][key]; sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(model['path'])==model['sha256']
assert sha('common.cif')==cfg['common_sha256']
out=ROOT/key;out.mkdir(exist_ok=False); start=time.monotonic()
state=torch.load(model['path'],map_location='cpu',weights_only=True)
info=state['model']['_extra_state']['train_infos']; assert info['step']==int(key)-1
del state
dp=DeepPot(model['path']); ti=dp.get_type_map().index('Ti');times=[]

class Native(Calculator):
    implemented_properties=['energy','forces','stress']
    def calculate(self,atoms=None,properties=['energy'],system_changes=all_changes):
        super().calculate(atoms,properties,system_changes)
        ts=time.monotonic()
        e,f,v=dp.eval(atoms.positions.reshape(1,-1),atoms.cell.array.reshape(1,9),[ti]*len(atoms))[:3]
        times.append({'n':len(atoms),'seconds':time.monotonic()-ts})
        stress=-v.reshape(3,3)/atoms.get_volume();stress=(stress+stress.T)/2
        self.results=dict(energy=float(e.reshape(-1)[0]),forces=f.reshape(-1,3),stress=stress.flat[[0,4,8,5,2,1]])
        assert all(np.isfinite(a).all() for a in self.results.values())

calc=Native(); common=read(ROOT/'common.cif');common.set_masses([47.867]*2);common.calc=calc
basecell=common.cell.array.copy();basefrac=common.get_scaled_positions()
def geometry(x):
    at=common.copy();cell=basecell.copy();cell[:2]*=np.exp(x[0]);cell[2]*=np.exp(x[1])
    at.set_cell(cell);at.set_scaled_positions(basefrac);at.calc=calc
    return at
def stress_residual(x):
    s=geometry(x).get_stress()/GPa
    return np.array([(s[0]+s[1])/2,s[2]])
def jacobian(x):
    h=1e-4;return np.column_stack([(stress_residual(x+np.eye(2)[i]*h)-stress_residual(x-np.eye(2)[i]*h))/(2*h) for i in range(2)])
def describe(at):
    cp=at.cell.cellpar()
    return {'a_A':float(cp[0]),'c_A':float(cp[2]),'c_over_a':float(cp[2]/cp[0]),'V_per_atom_A3':at.get_volume()/len(at),
            'energy_eV_atom':at.get_potential_energy()/len(at),'stress_GPa':(at.get_stress()/GPa).tolist(),
            'max_force_eV_A':float(np.max(np.linalg.norm(at.get_forces(),axis=1)))}

common_desc=describe(common)
# Check periodic small-cell inference against the same crystal replicated 4^3.
rep=common.repeat((4,4,4));rep.calc=calc;rep_desc=describe(rep)
equiv={'energy_difference_meV_atom':1000*(rep_desc['energy_eV_atom']-common_desc['energy_eV_atom']),
       'stress_max_difference_GPa':float(np.max(np.abs(np.array(rep_desc['stress_GPa'])-common_desc['stress_GPa'])))}
assert abs(equiv['energy_difference_meV_atom'])<0.02 and equiv['stress_max_difference_GPa']<0.005,equiv
opt=least_squares(stress_residual,np.zeros(2),jac=jacobian,bounds=(-.08,.08),xtol=1e-10,ftol=1e-10,gtol=1e-8,max_nfev=60)
relaxed=geometry(opt.x);relaxed_desc=describe(relaxed)
assert np.max(np.abs(relaxed_desc['stress_GPa']))<.005,relaxed_desc
assert relaxed_desc['max_force_eV_A']<1e-4,relaxed_desc
# Positive curvature within the hcp a,c strain subspace.
j=jacobian(opt.x); hess=relaxed.get_volume()*GPa*np.diag([2.,1.])@j
assert np.min(np.linalg.eigvalsh((hess+hess.T)/2))>0
write(out/'POSCAR-common',common,format='vasp',direct=True)
write(out/'POSCAR-relaxed',relaxed,format='vasp',direct=True)
path=common.cell.bandpath('GKMGA',npoints=200)
qaxis,ticks,labels=path.get_linear_kpoint_axis()
summary={'checkpoint':key,'model_sha256':model['sha256'],'train_infos':info,'common':common_desc,'relaxed':relaxed_desc,
         'small_cell_replication':equiv,'optimization':{'success':bool(opt.success),'message':str(opt.message),'nfev':opt.nfev,'log_strain':opt.x.tolist(),'hcp_strain_hessian_eigenvalues_eV':np.linalg.eigvalsh((hess+hess.T)/2).tolist()},
         'cases':{},'path_labels':labels,'path_ticks':ticks.tolist(),'path_qaxis':qaxis.tolist(),
         'versions':{p:importlib.metadata.version(p) for p in ['numpy','scipy','ase','torch','deepmd-kit']}}
(out/'progress.json').write_text(json.dumps(summary,indent=2));print('RELAXED',key,relaxed_desc,flush=True)
arrays={'qpoints':path.kpts,'qaxis':qaxis,'ticks':ticks}
for geom,at in [('common',common),('relaxed',relaxed)]:
    for variant,size,delta in [('base',(5,5,5),.01),('half-displacement',(5,5,5),.005),('large-cell',(7,7,7),.01)]:
        tag=geom+'-'+variant; ts=time.monotonic()
        ph=Phonons(at,calc,supercell=size,delta=delta,name=str(out/tag))
        ph.run()
        rawforces=[]
        for cpath in sorted((out/tag).glob('cache.*.json')):
            from ase.io.jsonio import read_json
            data=read_json(cpath);rawforces.append(data['forces'])
        assert len(rawforces)==13
        arrays[tag+'-forces']=np.array(rawforces)
        ph.read(method='standard',symmetrize=0,acoustic=False)
        arrays[tag+'-C-raw']=ph.C_N.copy()
        arrays[tag+'-bands-raw']=ph.band_structure(path.kpts,verbose=False)*241.799
        ph.read(method='Frederiksen',symmetrize=3,acoustic=True)
        arrays[tag+'-C']=ph.C_N.copy();arrays[tag+'-D']=ph.D_N.copy()
        arrays[tag+'-R']=ph.compute_lattice_vectors()
        bands=ph.band_structure(path.kpts,verbose=False)*241.799
        arrays[tag+'-bands']=bands
        gamma=ph.band_structure(np.zeros((1,3)),verbose=False)[0]*241.799
        summary['cases'][tag]={'supercell':list(size),'delta_A':delta,'natoms':int(2*np.prod(size)),
            'wall_seconds':time.monotonic()-ts,'gamma_THz':gamma.tolist(),'min_band_THz':float(bands.min()),'max_band_THz':float(bands.max()),
            'force_sum_max_eV_A':float(max(np.linalg.norm(f.sum(axis=0)) for f in rawforces)),
            'fc_correction_max_eV_A2':float(np.max(np.abs(ph.C_N-arrays[tag+'-C-raw'])))}
        print('DONE',key,tag,summary['cases'][tag],flush=True)
        (out/'progress.json').write_text(json.dumps(summary,indent=2))
np.savez_compressed(out/'results.npz',**arrays)
summary.update(status='passed',job_id=os.environ['SLURM_JOB_ID'],node=os.uname().nodename,
               wall_seconds=time.monotonic()-start,inference_calls=times,results_sha256=sha(out/'results.npz'),
               script_sha256=sha(__file__),protocol_sha256=sha(ROOT/'protocol.json'))
(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
print('COMPLETE',key,summary['wall_seconds'],flush=True)
