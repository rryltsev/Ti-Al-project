"""Shared NPT code for CPU/GPU-validated full composition series."""
from pathlib import Path
import os,sys,json,time,hashlib,importlib.metadata
import numpy as np
from ase import units
from ase.io import read,write
from ase.io.trajectory import Trajectory
from ase.calculators.calculator import Calculator,all_changes
from ase.md.melchionna import MelchionnaNPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution,Stationary
from deepmd.infer import DeepPot
import torch
ROOT=Path.cwd();cfg=json.loads(Path('protocol.json').read_text());tasks=json.loads(Path('tasks.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert os.environ.get('SLURM_JOB_ID')
assert sha('tasks.json')==cfg['tasks_sha256']
torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','4')))
class Native(Calculator):
    implemented_properties=['energy','forces','stress']
    def __init__(self,step):
        super().__init__();m=cfg['models'][str(step)];assert sha(m['path'])==m['sha256']
        data=torch.load(m['path'],map_location='cpu',weights_only=True)
        assert data['model']['_extra_state']['train_infos']['step']==step-1
        self.dp=DeepPot(m['path']);self.names=self.dp.get_type_map();self.calls=0;self.seconds=0
    def calculate(self,atoms=None,properties=['forces'],system_changes=all_changes):
        super().calculate(atoms,properties,system_changes);t=time.monotonic()
        e,f,v=self.dp.eval(atoms.positions.reshape(1,-1),atoms.cell.array.reshape(1,9),[self.names.index(s) for s in atoms.get_chemical_symbols()])[:3]
        e=float(e.reshape(-1)[0]);f=f.reshape(-1,3);v=v.reshape(3,3);s=-(v+v.T)/(2*atoms.get_volume())
        assert np.isfinite(e) and np.isfinite(f).all() and np.isfinite(s).all()
        self.results={'energy':e,'forces':f,'stress':s.flat[[0,4,8,5,2,1]]}
        self.calls+=1;self.seconds+=time.monotonic()-t
def initial(state,calc):
    assert sha(state['initial_file'])==state['initial_sha256']
    a=read(state['initial_file']);a.calc=calc
    cell,Q=a.cell.standard_form('upper');a.set_cell(cell,scale_atoms=False);a.positions=a.positions@Q.T
    a.wrap();return a
def row(a,step):
    N=len(a);V=a.get_volume();E=a.get_potential_energy();K=a.get_kinetic_energy()
    P=-float(np.mean(a.get_stress(include_ideal_gas=True)[:3]));T=a.get_temperature()
    assert np.isfinite(V) and 10<V/N<40 and 200<T<6000,(step,V/N,T)
    return dict(step=int(step),time_ps=step*.001,Epot_eV_atom=E/N,Ekin_eV_atom=K/N,
      H_external_eV_atom=(E+K+cfg['P_bar']*units.bar*V)/N,H_instantaneous_eV_atom=(E+K+P*V)/N,
      H_legacy_eV_atom=(E+K-P*V)/N,T_K=T,P_GPa=P/units.GPa,V_A3_atom=V/N,
      rho_g_cm3=float(a.get_masses().sum()/V*1.66053906660))
def stage(a,state,steps,folder,label,seed):
    MaxwellBoltzmannDistribution(a,temperature_K=cfg['T_K'],force_temp=True,rng=np.random.RandomState(seed));Stationary(a)
    dyn=MelchionnaNPT(a,timestep=cfg['dt_fs']*units.fs,temperature_K=cfg['T_K'],externalstress=cfg['P_bar']*units.bar,
       ttime=cfg['ttime_fs']*units.fs,pfactor=(cfg['ttime_fs']*units.fs)**2*state['bulk_modulus_GPa']*units.GPa,logfile=None)
    dyn.set_fraction_traceless(0)
    fp=(folder/(label+'-thermo.jsonl')).open('x');tr=Trajectory(str(folder/(label+'.traj')),'w',a)
    def observe():
        r=row(a,dyn.nsteps);fp.write(json.dumps(r)+'\n');fp.flush()
        if dyn.nsteps%1000==0:print(label,dyn.nsteps,round(r['T_K'],1),round(r['rho_g_cm3'],4),flush=True)
    dyn.attach(observe,interval=cfg['thermo_interval']);dyn.attach(tr.write,interval=cfg['trajectory_interval'])
    t=time.monotonic();dyn.run(steps);seconds=time.monotonic()-t;fp.close();tr.close()
    write(folder/(label+'-final.traj'),a)
    return seconds
