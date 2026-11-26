"""Freeze a matched NPT checkpoint comparison and reprocess the archived 5k baseline."""
from pathlib import Path
import sys,json,hashlib,shutil
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'audit/oct20/deps'))
import numpy as np
from ase.io import read,write
from ase import units
OUT=HERE/'pbe_control_md'; OUT.mkdir(exist_ok=True)
assert not (OUT/'protocol.json').exists()
audit=json.loads((HERE/'all_xc_audit/protocol.json').read_text())
source=Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project/results-scripts-logs/models_check_wsl2')
base=source/'results/mixing_enthalpy/dpa4/neo/finetune/AlTi/2000K/pbe'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
save=lambda n,d:(OUT/n).write_text(json.dumps(d,indent=2),encoding='utf-8')
comps=[('Ti',0.),('Al',1.),('Al0.3Ti0.7',.3),('Al0.5Ti0.5',.5)]
states=[];baseline={}
for i,(name,x) in enumerate(comps):
    p=base/(name+'_2000K.traj');ats=read(p,index=':');assert len(ats)==301
    at=ats[-1].copy();assert len(at)==200 and at.get_chemical_symbols().count('Al')==round(x*200)
    at.calc=None;write(OUT/(name+'.traj'),at)
    rows=[]
    for j,a in enumerate(ats):
        n=len(a);V=a.get_volume();E=a.get_potential_energy();K=a.get_kinetic_energy()
        s=a.get_stress(include_ideal_gas=True);P=-float(np.mean(s[:3]))
        rows.append(dict(step=j*100,time_ps=j*.1,Epot_eV_atom=E/n,Ekin_eV_atom=K/n,
          H_external_eV_atom=(E+K+units.bar*V)/n,H_instantaneous_eV_atom=(E+K+P*V)/n,
          H_legacy_eV_atom=(E+K-P*V)/n,T_K=a.get_temperature(),P_GPa=P/units.GPa,
          V_A3_atom=V/n,rho_g_cm3=float(a.get_masses().sum()/V*1.66053906660)))
    baseline[name]={'source':str(p),'sha256':sha(p),'x_Al':x,'rows':rows}
    states.append(dict(composition=name,x_Al=x,initial_file=name+'.traj',initial_sha256=sha(OUT/(name+'.traj')),
                       initial_source=str(p),initial_frame=300,bulk_modulus_GPa=110*(1-x)+76*x,seed=20261025+i*101))
save('baseline_5k.json',baseline)
tasks=[dict(id=len(comps)*j+i,checkpoint=step,state=i) for j,step in enumerate([15000,20000]) for i in range(4)]
save('tasks.json',tasks)
cfg=dict(remote='/home6/u3129/tial-testsets/pbe-control-md-20261025-01',models={str(s):audit['models']['PBE-'+str(s)] for s in [5000,15000,20000]},
 states=states,tasks_sha256=sha(OUT/'tasks.json'),T_K=2000,P_bar=1,dt_fs=1,ttime_fs=25,
 equil_steps=30000,production_steps=30000,thermo_interval=10,trajectory_interval=100,n_atoms=200,
 barostat='ASE MelchionnaNPT; pfactor=(25 fs)^2 B; fraction_traceless=0',
 initialization='Final archived 5k liquid snapshot, new seeded velocities; 30 ps equilibration for each model. Production velocities reinitialized as in the original script.',
 enthalpy='Primary H=(Epot+Ekin+P_external*V)/N; also store instantaneous positive-compression P and legacy sign for traceability.',
 comparison='Archived 5k trajectories reprocessed identically. Same random velocity seeds across checkpoints per composition; trajectories are not statistically paired samples.',
 uncertainty='Five and ten contiguous production blocks; report conservative SEM and half-trajectory drift. One trajectory per composition/model; extend if inconclusive.',
 scope='Eight new NPT runs. Existing article results unchanged. Checkpoint decision after completed MD and uncertainty assessment.')
save('protocol.json',cfg)
for f in ['md_runner.py','trajectory_analyzer.py','bulk_modulus.py']:
    shutil.copyfile(source/'utils'/f,OUT/('original_'+f))
print('Prepared 8 runs; archived 5k baseline:',{k:len(v['rows']) for k,v in baseline.items()})
