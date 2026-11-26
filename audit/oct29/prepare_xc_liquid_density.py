"""Freeze the two final-XC liquid density extensions, without overwriting prior work."""
from pathlib import Path
import csv, hashlib, json, shutil, sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'audit/oct20/deps'))
import numpy as np
from ase.io import write
from ase.io.trajectory import Trajectory

OUT = Path(__file__).parent/'xc_liquid_density_extended'
OUT.mkdir(exist_ok=True)
assert not (OUT/'protocol.json').exists()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
save = lambda n,d: (OUT/n).write_text(json.dumps(d, indent=2)+'\n', encoding='utf-8')
project = Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project')
base = project/'results-scripts-logs/models_check_wsl2/results'
models_all = json.loads((ROOT/'audit/oct25/all_xc_audit/protocol.json').read_text())['models']
models = {str(step): models_all[f'{xc}-{step}'] for xc,step in [('PBEsol',23100),('r2SCAN',23200)]}
states = []; old = []
for T in [1873,2500,2400,2300,2200,2100,2000]:
    for key,xc,step in [('sol','PBEsol',23100),('scan','r2SCAN',23200)]:
        source = (base/f'mixing_enthalpy/dpa4/neo/finetune/AlTi/1873K/{key}/Ti_1873K.traj' if T==1873 else
                  base/f'density/dpa4/neo/finetune/Ti/{key}/liquid/T_{T}K/trajectory.traj')
        with Trajectory(str(source)) as tr:
            assert len(tr)==301, (source,len(tr))
            a = tr[-1]; assert len(a)==200 and set(a.get_chemical_symbols())=={'Ti'}
            rho = [f.get_masses().sum()/f.get_volume()*1.66053906660 for f in tr]
            e = float(a.get_potential_energy())
            old.append(dict(functional=xc,key=key,T_K=T,source=str(source),source_sha256=sha(source),
                            rho_mean_all_frames=float(np.mean(rho)),rho_mean_excluding_t0=float(np.mean(rho[1:])),
                            production_ps=30.,cadence_fs=100,frames=301))
        sid=f'{xc}-{T}'; initial_name=sid+'.traj'
        a.calc=None; write(OUT/initial_name,a)
        states.append(dict(id=sid,functional=xc,key=key,checkpoint=step,phase='liquid',T_K=T,N=200,
                           initial_file=initial_name,initial_sha256=sha(OUT/initial_name),
                           initial_source=str(source),source_sha256=sha(source),archived_last_energy_eV=e,
                           bulk_modulus_GPa=110.,seed=202610291+len(states)*43,
                           equil_steps=10000,production_steps=90000))
for key in ['sol','scan']:
    path=project/f'MD-results/density_DPA4-neo/density_scan_{key}.csv'
    rows=list(csv.DictReader(path.open(),delimiter=';'))
    for r in rows:
        if r['phase']=='liquid':
            p=next(p for p in old if p['key']==key and p['T_K']==float(r['T_target_K']))
            assert abs(p['rho_mean_all_frames']-float(r['density_g_cm3']))<.000051,(p,r)
save('archived_liquid_points.json',old)
tasks=[dict(id=s['id'],state=i,checkpoint=s['checkpoint']) for i,s in enumerate(states)]
save('tasks.json',tasks)
cfg=dict(remote='/home6/u3129/tial-testsets/ti-liquid-density-xc-20261029-01',models=models,states=states,
         tasks_sha256=sha(OUT/'tasks.json'),P_bar=1.,dt_fs=1.,ttime_fs=25.,thermo_interval=10,
         trajectory_interval=100,allocation_wall_hours=20,
         priority='Both 1873 K points first, then paired descending temperatures 2500 to 2000 K',
         protocol_notes=['Final article checkpoints: PBEsol 23100 and r2SCAN 23200, verified by SHA256.',
         'Each point restarts from the final archived same-functional, same-temperature liquid cell.',
         '10 ps discarded re-equilibration plus a new 90 ps production window (three times the old 30 ps).',
         'Velocities and thermostat/barostat variables restart at each stage; this is not an exact MD continuation.',
         'Old 30 ps sparse trajectories remain separate. Primary estimates use new dense 90 ps logs, with no pooling across cadences.',
         'No fitted smoothing or selection of windows by smoothness. Plot means and block SEM; inspect pressure, drift, mobility and phase.',
         'One Slurm allocation on one A100; no automatic replacement job.'])
save('protocol.json',cfg)
shutil.copyfile(ROOT/'audit/oct26/pbe15k_density_1873/md_common.py',OUT/'md_common.py')
shutil.copyfile(Path(__file__).parent/'xc_liquid_density_workflow.py',OUT/'workflow.py')
job=(ROOT/'audit/oct29/pbe15k_ti_density_temperature/job.sbatch').read_text()
(OUT/'job.sbatch').write_text(job,newline='\n')
save('preparation-verification.json',dict(states=len(states),total_steps=sum(s['equil_steps']+s['production_steps'] for s in states),
     primary_production_ps_per_point=90,archive_production_ps_per_point=30,archived_csv_reproduced=True))
print('Prepared',len(states),'points, 1.4 million MD steps, 90 ps new production per point')
