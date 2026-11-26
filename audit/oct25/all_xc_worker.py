"""CPU-only, immutable-input checkpoint audit worker: inference, EOS, liquids, Ti DOS."""
from pathlib import Path
import os,json,hashlib,time,importlib.metadata
import numpy as np
from scipy.optimize import brentq
from ase import Atoms
from ase.build import bulk
from ase.io import read,write
from ase.units import GPa
from ase.calculators.calculator import Calculator,all_changes
from ase.phonons import Phonons
from deepmd.infer import DeepPot
import torch
ROOT=Path.cwd();cfg=json.loads(Path('protocol.json').read_text());tasks=json.loads(Path('tasks.json').read_text())
task=tasks[int(os.environ['SLURM_ARRAY_TASK_ID'])];key=task['key'];model=cfg['models'][key];xc=model['functional']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert os.environ.get('SLURM_JOB_ID') and not os.environ.get('CUDA_VISIBLE_DEVICES')
assert sha(model['path'])==model['sha256'] and sha('tasks.json')==cfg['tasks_sha256']
state=torch.load(model['path'],map_location='cpu',weights_only=True)
info=state['model']['_extra_state']['train_infos'];assert info['step']==model['step']-1;del state
out=ROOT/'results'/str(task['id']);out.mkdir(parents=True,exist_ok=False)
start=time.monotonic();dp=DeepPot(model['path']);names=dp.get_type_map();timings=[]
assert dp.get_dim_fparam()==dp.get_dim_aparam()==0
def evaluate(at):
    t=time.monotonic();e,f,v=dp.eval(at.positions.reshape(1,-1),at.cell.array.reshape(1,9),[names.index(s) for s in at.get_chemical_symbols()])[:3]
    e=float(e.reshape(-1)[0]);f=f.reshape(-1,3);v=v.reshape(3,3)
    timings.append({'n':len(at),'s':time.monotonic()-t})
    assert np.isfinite(e) and np.isfinite(f).all() and np.isfinite(v).all()
    stress=-(v+v.T)/(2*at.get_volume())
    return {'energy':e,'forces':f,'virial':v,'stress':stress,
            'E_eV_atom':e/len(at),'P_GPa':float(-np.trace(stress)/(3*GPa)),
            'Fmax_eV_A':float(np.linalg.norm(f,axis=1).max())}
def from_arrays(data,i):
    return Atoms(symbols=[['Al','Ti'][int(t)] for t in data[f'types_{i}']],positions=data[f'coords_{i}'],cell=data[f'cell_{i}'],pbc=True)
def prediction_set(ds,indices,filename):
    assert sha(ds['geometry_file'])==ds['geometry_sha256']
    data=np.load(ds['geometry_file'],allow_pickle=False);arrays={};scalars=[]
    for n,i in enumerate(indices):
        at=from_arrays(data,i);d=evaluate(at)
        for k in ['energy','forces','virial']:arrays[f'{k}_{i}']=d[k]
        scalars.append({'i':i,**{k:d[k] for k in ['E_eV_atom','P_GPa','Fmax_eV_A']}})
        if n%50==0:print('INFERENCE',task['id'],key,n+1,len(indices),flush=True)
    np.savez_compressed(out/(filename+'.npz'),**arrays)
    (out/(filename+'.json')).write_text(json.dumps(scalars,indent=2))
def structure(phase,V,ratio=1.58):
    if phase=='Ti_hcp':
        a=(4*V/(np.sqrt(3)*ratio))**(1/3);return bulk('Ti','hcp',a=a,c=ratio*a)
    if phase=='Ti_bcc':return bulk('Ti','bcc',a=(2*V)**(1/3),cubic=True)
    return bulk('Al','fcc',a=(4*V)**(1/3),cubic=True)
def eos_point(phase,V):
    ratio=None
    if phase=='Ti_hcp':
        def dev(r):
            s=evaluate(structure(phase,V,r))['stress']/GPa
            return s[2,2]-(s[0,0]+s[1,1])/2
        ratio=brentq(dev,*cfg['eos']['hcp_c_over_a_bracket'],xtol=1e-8)
    at=structure(phase,V,ratio if ratio else 1.58);d=evaluate(at);s=d['stress']/GPa
    assert np.max(abs(s-np.eye(3)*np.trace(s)/3))<.005 and d['Fmax_eV_A']<1e-4
    return at,{'V_A3_atom':V,'E_eV_atom':d['E_eV_atom'],'P_GPa':d['P_GPa'],
               'c_over_a':ratio,'cell_A':at.cell.array.tolist(),'Fmax_eV_A':d['Fmax_eV_A']}
class Native(Calculator):
    implemented_properties=['energy','forces','stress']
    def calculate(self,atoms=None,properties=['forces'],system_changes=all_changes):
        super().calculate(atoms,properties,system_changes);d=evaluate(atoms)
        self.results={'energy':d['energy'],'forces':d['forces'],'stress':d['stress'].flat[[0,4,8,5,2,1]]}
def phonons(at,label):
    folder=out/label;folder.mkdir();at=at.copy();at.set_masses([47.867]*len(at));at.calc=Native()
    cell=at.cell.array.copy();frac=at.get_scaled_positions().copy()
    desc={'cell_A':cell.tolist(),'fractional_positions':frac.tolist(),'V_A3_atom':at.get_volume()/len(at),
          'E_eV_atom':at.get_potential_energy()/len(at),'stress_GPa':(at.get_stress()/GPa).tolist()}
    ph=Phonons(at,at.calc,supercell=(9,9,9),delta=.01,name=str(folder/'forces'));ph.run()
    path=at.cell.bandpath('GKMGA',npoints=200);qaxis,ticks,labels=path.get_linear_kpoint_axis()
    arrays={'qpoints':path.kpts,'qaxis':qaxis,'ticks':ticks}
    ph.read(method='standard',symmetrize=0,acoustic=False);arrays['C_raw']=ph.C_N.copy()
    arrays['bands_raw_THz']=ph.band_structure(path.kpts,verbose=False)*241.799
    ph.read(method='Frederiksen',symmetrize=3,acoustic=True)
    arrays.update(C=ph.C_N.copy(),D=ph.D_N.copy(),R=ph.compute_lattice_vectors(),bands_THz=ph.band_structure(path.kpts,verbose=False)*241.799)
    assert np.array_equal(cell,at.cell.array) and np.array_equal(frac,at.get_scaled_positions())
    np.savez_compressed(folder/'results.npz',**arrays)
    desc.update(min_band_THz=float(arrays['bands_THz'].min()),gamma_THz=arrays['bands_THz'][0].tolist(),
                geometry_unchanged=True,supercell=[9,9,9],delta_A=.01,
                force_files={p.name:sha(p) for p in (folder/'forces').glob('cache.*.json')})
    assert len(desc['force_files'])==13
    (folder/'summary.json').write_text(json.dumps(desc,indent=2));print('PHONONS',key,label,flush=True)
if task['kind']=='inference':
    prediction_set(cfg['datasets'][xc],task['indices'],'predictions')
else:
    eos={};relaxed={}
    for phase in ['Ti_hcp','Ti_bcc','Al_fcc']:
        ref=cfg['eos'][phase+'_reference_V_A3_atom'];vols=ref*np.array(cfg['eos']['volume_factors'])
        curve=[eos_point(phase,float(v))[1] for v in vols]
        v0=brentq(lambda v:eos_point(phase,v)[1]['P_GPa'],vols[0],vols[-1],xtol=1e-7)
        at,eq=eos_point(phase,v0);relaxed[phase]=at
        write(out/(phase+'-POSCAR'),at,format='vasp',direct=True)
        for h,label in [(.001,'B_GPa'),(.0005,'B_halfstep_GPa')]:
            lo=eos_point(phase,v0*(1-h))[1];hi=eos_point(phase,v0*(1+h))[1]
            eq[label]=-(hi['P_GPa']-lo['P_GPa'])/(2*h)
        assert eq['B_GPa']>0 and abs(eq['B_GPa']-eq['B_halfstep_GPa'])<.2
        vt=ref*.96;test=eos_point(phase,vt)[1];fd=[]
        for h in [.0005,.001,.002]:
            p=-(eos_point(phase,vt*(1+h))[1]['E_eV_atom']-eos_point(phase,vt*(1-h))[1]['E_eV_atom'])/(2*vt*h)/GPa
            fd.append({'relative_h':h,'pressure_GPa':p,'difference_GPa':p-test['P_GPa']})
        # Require agreement for two independent increments; all values retained.
        assert sum(abs(z['difference_GPa'])<.02 for z in fd)>=2,(phase,fd)
        a=evaluate(at);b=evaluate(at.repeat((3,3,3)))
        check={'delta_E_meV_atom':1000*(b['E_eV_atom']-a['E_eV_atom']),
               'delta_P_GPa':b['P_GPa']-a['P_GPa'],'FD_checks':fd}
        assert abs(check['delta_E_meV_atom'])<.02 and abs(check['delta_P_GPa'])<.005
        eos[phase]={'curve':curve,'equilibrium':eq,'verification':check}
        print('EOS',key,phase,v0,eq['B_GPa'],flush=True)
    (out/'eos.json').write_text(json.dumps(eos,indent=2))
    prediction_set(cfg['liquids'][xc],list(range(250)),'liquid')
    if task['fixed_phonons']:
        assert sha('POSCAR-DFT')==cfg['geometry_sha256'];phonons(read('POSCAR-DFT',format='vasp'),'fixed')
    if task['equilibrium_phonons']:phonons(relaxed['Ti_hcp'],'equilibrium')
summary={'status':'passed','task':task,'model_sha256':model['sha256'],'train_infos':info,
         'protocol_sha256':sha('protocol.json'),'script_sha256':sha(__file__),
         'wall_seconds':time.monotonic()-start,'job_id':os.environ['SLURM_JOB_ID'],'node':os.uname().nodename,
         'timings':timings,'versions':{p:importlib.metadata.version(p) for p in ['numpy','scipy','ase','torch','deepmd-kit']}}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print('COMPLETE',task['id'],key,summary['wall_seconds'],flush=True)
