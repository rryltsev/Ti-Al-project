from pathlib import Path
import csv,json,hashlib,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent;DATA=ROOT/'deliverables/2026-11-03/figure_data'
sources={}
def load(rel):
 p=ROOT/rel;sources[rel]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text(encoding='utf-8-sig'))
def csvout(rel,fields,rows):
 p=DATA/rel;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='',encoding='utf-8-sig') as f:w=csv.writer(f);w.writerow(fields);w.writerows(rows)
def npz(rel):
 p=ROOT/rel;sources[rel]=hashlib.sha256(p.read_bytes()).hexdigest();return np.load(p)
a=np.loadtxt(ROOT/'manuscript/supporting_data/lcurve_pbe.out');fields=['optimization_steps','steps_thousands','energy_train_meV_atom','energy_validation_meV_atom','force_train_eV_A','force_validation_eV_A']
y=np.column_stack([a[:,0],a[:,0]/1000,a[:,2]*1000,a[:,1]*1000,a[:,4],a[:,3]])
csvout('FigS01_learning/raw.csv',fields,y)
csvout('FigS01_learning/smoothed.csv',fields,np.column_stack([y[5:-5,:2]]+[np.convolve(y[:,i],np.ones(11)/11,mode='valid') for i in range(2,6)]))
d=load('audit/oct25/pbe_checkpoint_sweep/analysis.json');rows=[]
for step,splits in d['groups'].items():
 for split,v in splits.items():rows.append([int(step),int(step)/1000,split,v['energy_rmse_meV_atom'],v['force_rmse_eV_A'],d['subgroups'][step][split]['xTi_ge_0.9']['energy_bias_meV_atom']])
csvout('FigS02_checkpoints/metrics.csv',['optimization_steps','steps_thousands','split','energy_RMSE_meV_atom','force_RMSE_eV_A','Ti_rich_energy_bias_meV_atom'],rows)
d=load('audit/oct25/pbe_checkpoint_sweep/per_configuration.json');rows=[]
for step,g in d.items():
 for r in g:rows.append([int(step),r['split'],r['id'],r['x_Ti'],r['volume_A3_atom'],r['energy_error_meV_atom']])
fields=['optimization_steps','split','structure_id','x_Ti','volume_A3_atom','energy_residual_meV_atom'];csvout('FigS02_checkpoints/energy_residuals.csv',fields,rows)
csvout('FigS03_fixed_composition/points.csv',fields,[r for r in rows if r[0] in [5000,23400] and (abs(r[3]-205/216)<1e-12 or r[3]==1) and r[1] in ['train','test20']])
d=load('audit/oct25/pbe_train_comparison/all_residuals.json')
csvout('FigS04_train_test/points.csv',['split','structure_id','x_Ti','volume_A3_atom','fine_tuned_residual_meV_atom','pretrained_residual_meV_atom'],[[split,r['id'],r['x_Ti'],r['volume_A3_atom'],r['fine_tuned'],r['pretrained']] for split,g in d.items() for r in g])
z=npz('audit/oct25/pbe_checkpoint_phonons/5000/results.npz');q=z['qaxis']/z['qaxis'][-1]
csvout('FigS05_phonons/q_ticks.csv',['q_normalized'],[[x] for x in z['ticks']/z['qaxis'][-1]])
for step in ['5000','20000','23400']:
 z=npz(f'audit/oct25/pbe_checkpoint_phonons/size9-{step}/results.npz')
 for geom in ['common','relaxed']:csvout(f'FigS05_phonons/{step}_{geom}.csv',['q_normalized']+[f'branch_{i+1}_THz' for i in range(6)],np.column_stack([q,z[geom+'-bands']]))
p=DATA/'FigS05_phonons/experimental_source.csv';shutil.copyfile(ROOT/'audit/oct25/pbe_checkpoint_phonons/reference-data.csv',p)
def dos_csv(rel,dest,keys=None):
 z=npz(rel);keys=keys or z.files;axis='frequency_THz';csvout(dest,[axis]+[k+'_modes_atom_THz' for k in keys if k!=axis],np.column_stack([z[axis]]+[z[k] for k in keys if k!=axis]))
dos_csv('audit/oct25/pbe_checkpoint_phonons/dos.npz','FigS06_DOS/curves.csv')
dos_csv('audit/oct25/pbe_checkpoint_phonons/dft_dos_comparison/normalized_dos_sigma_0p05.npz','FigS07_DOS_DFT/curves.csv')
keys=['frequency_THz','DFT']+[f'PBE-{step}/{g}/MP40' for g in ['fixed','equilibrium'] for step in [5000,10000,15000,20000,23400]]
dos_csv('audit/oct25/all_xc_audit/analysis/dos_curves.npz','FigS08_five_DOS/curves.csv',keys)
d=load('audit/oct25/pbe_phonons_five_checkpoints/analysis.json');rows=[];pr=d['width_checks']['0.05']
for step in ['5000','10000','15000','20000','23400']:
 v=pr[step+'/MP40'];rows.append([int(step),int(step)/1000,d['errors'][step]['validation']['energy_rmse_meV_atom'],d['errors'][step]['test20']['energy_rmse_meV_atom'],v['L2_relative'],v['TV'],v['W1_THz'],v['L2_relative']/pr['5000/MP40']['L2_relative'],v['TV']/pr['5000/MP40']['TV'],v['W1_THz']/pr['5000/MP40']['W1_THz']])
csvout('FigS09_tradeoff/points.csv',['optimization_steps','steps_thousands','validation_energy_RMSE_meV_atom','test20_energy_RMSE_meV_atom','DOS_L2_relative','DOS_total_variation','DOS_W1_THz','L2_ratio_to_5k','TV_ratio_to_5k','W1_ratio_to_5k'],rows)
d=load('audit/oct25/pbe_full_enthalpy_gpu/analysis/statistics.json');w=d['windows'];p=w['production'];rows=[]
for step in ['5000','15000','20000']:
 for name,v in p['groups'][step].items():
  h=p['mixing'][step][name]['H'];rows.append([int(step),v['x_Al'],v['rho']['mean'],v['rho']['sem'],h['mean'],h['sem']])
csvout('FigS10_density_enthalpy/curves.csv',['optimization_steps','x_Al','density_g_cm3','density_SEM_g_cm3','mixing_enthalpy_kJ_mol','enthalpy_SEM_kJ_mol'],rows)
rows=[]
for step in ['15000','20000']:
 for win,g in w.items():
  for name,v in g['mixing'][step].items():rows.append([int(step),win,v['x_Al'] if 'x_Al' in v else p['groups'][step][name]['x_Al'],v['H']['mean']-p['mixing'][step][name]['H']['mean']])
csvout('FigS11_window_sensitivity/curves.csv',['optimization_steps','window','x_Al','change_from_production_kJ_mol'],rows)
(OUT/'si_source_hashes.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
print('Exported all eleven SI figures.')
