"""Freeze the full 2000 K composition series; one allocation for both checkpoints."""
from pathlib import Path
import sys,json,hashlib,shutil
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'pbe_full_enthalpy_gpu'
OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists()
sys.path.insert(0,str(ROOT/'audit/oct20/deps'))
import numpy as np
from ase.io import read,write
from ase import units
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
save=lambda n,d:(OUT/n).write_text(json.dumps(d,indent=2))
old=json.loads((HERE/'pbe_control_md/protocol.json').read_text())
base=Path(old['states'][0]['initial_source']).parent
comps=[('Ti',0.),('Al',1.)]+[(f'Al{i/10:.1f}Ti{1-i/10:.1f}',i/10) for i in range(1,10)]
states=[];baseline={};oldseeds={s['composition']:s['seed'] for s in old['states']}
for i,(name,x) in enumerate(comps):
    p=base/(name+'_2000K.traj');ats=read(p,index=':');assert len(ats)==301
    at=ats[-1].copy();assert len(at)==200 and at.get_chemical_symbols().count('Al')==round(x*200)
    at.calc=None;write(OUT/(name+'.traj'),at)
    rows=[]
    for j,a in enumerate(ats):
        n=len(a);V=a.get_volume();E=a.get_potential_energy();K=a.get_kinetic_energy();P=-float(np.mean(a.get_stress(include_ideal_gas=True)[:3]))
        rows.append(dict(step=j*100,time_ps=j*.1,Epot_eV_atom=E/n,Ekin_eV_atom=K/n,
          H_external_eV_atom=(E+K+units.bar*V)/n,H_instantaneous_eV_atom=(E+K+P*V)/n,H_legacy_eV_atom=(E+K-P*V)/n,
          T_K=a.get_temperature(),P_GPa=P/units.GPa,V_A3_atom=V/n,rho_g_cm3=float(a.get_masses().sum()/V*1.66053906660)))
    baseline[name]={'source':str(p),'sha256':sha(p),'x_Al':x,'rows':rows}
    states.append(dict(composition=name,x_Al=x,initial_file=name+'.traj',initial_sha256=sha(OUT/(name+'.traj')),
      initial_source=str(p),initial_frame=300,bulk_modulus_GPa=110*(1-x)+76*x,seed=oldseeds.get(name,20270000+i*101)))
tasks=[dict(id=j*11+i,checkpoint=step,state=i) for j,step in enumerate([15000,20000]) for i in range(11)]
save('tasks.json',tasks);save('baseline_5k.json',baseline)
cfg={**old,'remote':'/home6/u3129/tial-testsets/pbe-full-enthalpy-gpu-20261025-01','states':states,'tasks_sha256':sha(OUT/'tasks.json'),
 'scope':'All eleven compositions at 2000 K for PBE 15k and 20k. One Slurm job, two A100 GPU workers, one checkpoint per GPU, sequential compositions within each worker.',
 'gpu_validation':'CPU/GPU predictions on all eleven initial structures before MD. No TF32, AMP, Triton or compile acceleration.',
 'allocation_wall_hours':24}
save('protocol.json',cfg)
common=(HERE/'control_md_worker.py').read_text().split("if len(sys.argv)>1 and sys.argv[1]=='preflight':")[0]
common=common.replace("assert os.environ.get('SLURM_JOB_ID') and not os.environ.get('CUDA_VISIBLE_DEVICES')","assert os.environ.get('SLURM_JOB_ID')")
common=common.replace('"""Bounded CPU NPT runs, matching the archived 5k integrator and thermodynamic state."""','"""Shared NPT code for CPU/GPU-validated full composition series."""')
(OUT/'md_common.py').write_text(common)
for n in ['gpu_enthalpy_worker.py','gpu_enthalpy_job.sbatch']:shutil.copyfile(HERE/n,OUT/n)
print('Prepared',len(tasks),'trajectories, 11 compositions; protocol',sha(OUT/'protocol.json'))
