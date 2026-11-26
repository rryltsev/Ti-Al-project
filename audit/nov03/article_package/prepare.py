from pathlib import Path
import json,hashlib,shutil,zipfile,csv,re,runpy
import numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent;MAN=ROOT/'manuscript'
DEL=ROOT/'deliverables/2026-11-03';DATA=DEL/'figure_data';DATA.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sources={}
def load(rel):
 p=ROOT/rel;sources[rel]=sha(p);return read(p)
def writecsv(rel,fields,rows):
 p=DATA/rel;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.writer(f);w.writerow(fields);w.writerows(rows)

with zipfile.ZipFile(OUT/'before_revision.zip','a' if (OUT/'before_revision.zip').exists() else 'w',zipfile.ZIP_DEFLATED) as z:
 for p in MAN.rglob('*'):
  if p.is_file() and 'build' not in p.relative_to(MAN).parts and p.relative_to(MAN).as_posix() not in z.namelist():z.write(p,p.relative_to(MAN).as_posix())
m=load('testset/monitor/results/latest.json');b=load('testset/monitor/data/latest.json')
assert m['analysis_mode']=='monitor-raw' and not m['pending_predictions']
assert m['models']['PBE']['checkpoint_step']==20000 and len(m['common_ids'])==49
(OUT/'test49_input_snapshot.json').write_text(json.dumps(b),encoding='utf-8')
(MAN/'supporting_data/test49_metrics.json').write_text(json.dumps(m,indent=2)+'\n',encoding='utf-8')
src=(ROOT/'audit/oct29/manuscript_revision/prepare_parity.py').read_text(encoding='utf-8')
src=src.replace('test30','test49').replace('30-structure','49-structure')
(OUT/'prepare_parity.py').write_text(src,encoding='utf-8')
runpy.run_path(str(OUT/'prepare_parity.py'),run_name='__main__')
records={r['metadata']['structure_id']:r for r in b['records']}
energies=[];forces=[];stresses=[]
for xc in ['PBE','PBEsol','r2SCAN']:
 for model,cache in [('Fine-tuned','PBE-20000' if xc=='PBE' else xc),('Pretrained','pretrained')]:
  for sid in m['common_ids']:
   r=records[sid];p=load(f'testset/monitor/predictions/{cache}/{sid}.json');ref=r['labels'][xc];n=len(r['symbols']);family=r['metadata']['family']
   energies.append([xc,model,sid,family,n,ref['energy_eV']/n,p['energy_eV']/n])
   for i,(a,c) in enumerate(zip(ref['forces_eV_A'],p['forces_eV_A'])):
    for j,axis in enumerate('xyz'):forces.append([xc,model,sid,family,i,r['symbols'][i],axis,a[j],c[j]])
   s0=np.asarray(ref['stress_eV_A3'])*160.21766208;s1=-np.asarray(p['virial_eV'])/ref['volume_A3']*160.21766208
   for i,j in zip(*np.triu_indices(3)):stresses.append([xc,model,sid,'xyz'[i]+'xyz'[j],s0[i,j],s1[i,j]])
writecsv('Fig01_parity/energy.csv',['functional','model','structure_id','family','n_atoms','DFT_energy_eV_atom','model_energy_eV_atom'],energies)
writecsv('Fig01_parity/force.csv',['functional','model','structure_id','family','atom_index_zero_based','element','component','DFT_force_eV_A','model_force_eV_A'],forces)
writecsv('Fig01_parity/stress.csv',['functional','model','structure_id','component','DFT_stress_GPa','model_stress_GPa'],stresses)
metricrows=[];texrows=[]
for xc,label in [('PBE','PBE'),('PBEsol','PBEsol'),('r2SCAN',r'\rscan{}')]:
 for kind,model in [('metrics','Fine-tuned'),('pretrained_metrics','Pretrained')]:
  a=m[kind][xc]['all'];v=[a[q][s] for q in ['energy_meV_atom','force_eV_A','stress_GPa'] for s in ['mae','rmse']]
  metricrows.append([xc,model]+v);texrows.append(label+' & '+model+' & '+' & '.join(f'{x:.{n}f}' for x,n in zip(v,[2,2,4,4,3,3]))+r' \\')
writecsv('Fig01_parity/metrics.csv',['functional','model','energy_MAE_meV_atom','energy_RMSE_meV_atom','force_MAE_eV_A','force_RMSE_eV_A','stress_MAE_GPa','stress_RMSE_GPa'],metricrows)
table=(MAN/'sections/testset_metrics_table.tex').read_text()
table=re.sub(r'\\caption\{.*?\}\n\\label',lambda _:r'\caption{Errors on 49 common held-out geometries from the 50-configuration test design. Energy errors use raw energies per atom; forces pool Cartesian components and stresses pool six independent tensor entries. The pretrained rows compare one unchanged model with each functional. No energy offsets are fitted.}'+'\n'+r'\label',table,count=1)
table=table[:table.index('PBE & Fine-tuned')]+ '\n'.join(texrows)+'\n'+table[table.index(r'\end{tabular}'):]
(MAN/'sections/testset_metrics_table.tex').write_text(table,encoding='utf-8')
a={xc:m['metrics'][xc]['all'] for xc in m['metrics']};pt={xc:m['pretrained_metrics'][xc]['all'] for xc in m['metrics']}
trip=lambda key,stat,fmt: ', '.join(format(a[xc][key][stat],fmt) for xc in ['PBE','PBEsol','r2SCAN'])
intervals=lambda key:', '.join(f"{a[xc][key]['rmse_ci95'][0]:.4f}--{a[xc][key]['rmse_ci95'][1]:.4f}" for xc in ['PBE','PBEsol','r2SCAN'])
para=f'''The fine-tuned energy MAEs are {trip('energy_meV_atom','mae','.2f')}~meV/atom for PBE, PBEsol, and \\rscan{{}}, respectively; the corresponding RMSEs are {trip('energy_meV_atom','rmse','.2f')}~meV/atom.
Descriptive 95\\% structure-bootstrap intervals for these RMSEs are {intervals('energy_meV_atom')}~meV/atom.
PBE has a mean energy residual of $+{a['PBE']['energy_meV_atom']['bias']:.2f}$~meV/atom and a largest absolute residual of {a['PBE']['energy_meV_atom']['max_abs']:.2f}~meV/atom.
Its energy MAE and RMSE are both smaller than the pretrained values of {pt['PBE']['energy_meV_atom']['mae']:.2f} and {pt['PBE']['energy_meV_atom']['rmse']:.2f}~meV/atom.

Fine-tuned force-component RMSEs are {trip('force_eV_A','rmse','.4f')}~eV/\\AA, compared with {', '.join(format(pt[xc]['force_eV_A']['rmse'],'.4f') for xc in a)}~eV/\\AA{{}} for the pretrained model against PBE, PBEsol, and \\rscan{{}}, respectively.
The central values decrease by approximately {', '.join(str(round(100*(1-a[xc]['force_eV_A']['rmse']/pt[xc]['force_eV_A']['rmse'])))+r'\%' for xc in a)} after fine-tuning.
The force-RMSE bootstrap intervals are {intervals('force_eV_A')}~eV/\\AA; their overlap cautions against a strong ranking of the three fits.
Stress RMSEs are {trip('stress_GPa','rmse','.3f')}~GPa.
These pooled Cartesian-component errors differ from the force-vector quantities printed in the training logs.

Raw pretrained energy RMSEs are approximately {pt['PBEsol']['energy_meV_atom']['rmse']:.0f}~meV/atom against PBEsol and {pt['r2SCAN']['energy_meV_atom']['rmse']:.0f}~meV/atom against \\rscan{{}}.
They include differences in elemental energy references and exchange-correlation descriptions, and therefore do not measure configuration-dependent errors alone.
Force and stress comparisons provide complementary evidence because constant elemental energy shifts leave these derivatives unchanged.

'''
p=MAN/'sections/results.tex';s=p.read_text(encoding='utf-8');s=s.replace('30 completed common geometries','49 common geometries').replace(r'20\,532 components',r'34\,446 components')
s=s[:s.index('The fine-tuned energy MAEs')]+para+s[s.index(r'\subsection{Lattice dynamics'):];p.write_text(s,encoding='utf-8')
p=MAN/'sections/conclusions.tex';s=p.read_text(encoding='utf-8');s=s.replace('1.51, 1.77, and 1.60', '1.47, 1.77, and 1.58').replace('0.063--0.073','0.062--0.071');p.write_text(s,encoding='utf-8')
p=MAN/'main.tex';p.write_text(p.read_text().replace('1 November 2026','3 November 2026'),encoding='utf-8')

# Fig.2: frozen exact plotting tables, not a new phonon calculation.
for name in ['bands.csv','DOS.csv','experiment.csv']:
 dest=DATA/'Fig02_phonons'/name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(MAN/'supporting_data/phonons_matched_20261101'/name,dest)
# Fig.3: all curves and reference values shown in the active density figure.
project=Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project')
rows=[]
for model in ['pbe','sol','scan']:
 path=project/f'MD-results/density_DPA4-neo/density_scan_{model}.csv';sources[str(path)]=sha(path)
 with path.open() as f:
  for r in csv.DictReader(f,delimiter=';'):
   T=float(r['T_target_K']);phase='alpha' if T<=1100 else 'beta' if T<=1900 else 'liquid'
   if model!='pbe' and phase=='liquid':continue
   rows.append([model,'archived',phase,T,float(r['density_g_cm3']),''])
for p in load('manuscript/supporting_data/pbe15k_density_temperature.json')['points']:
 if p['composition']=='Ti' and p['kind']=='homogeneous_NPT':rows.append(['pbe','updated',p['phase_initial'],p['T_K'],p['rho']['mean'],p['rho'].get('sem','')])
for p in load('manuscript/supporting_data/xc_liquid_density_extended.json')['points']:rows.append([p['key'],'extended','liquid',p['T_K'],p['rho']['mean'],p['rho']['sem']])
writecsv('Fig03_density_T/models.csv',['functional','series','starting_phase','temperature_K','density_g_cm3','SEM_g_cm3'],rows)
dref=load('audit/density_reference/references.json');lr=dref['liquid'];T=np.linspace(*lr['plotted_range_K'],200);rho=(lr['c1_kg_m3']+lr['c2_kg_m3_K']*(T-lr['Tm_K']))/1000;u=lr['expanded_relative_uncertainty']
writecsv('Fig03_density_T/liquid_correlation.csv',['temperature_K','density_g_cm3','lower_g_cm3','upper_g_cm3'],zip(T,rho,rho*(1-u),rho*(1+u)))
writecsv('Fig03_density_T/solid_reference.csv',['phase','temperature_K','density_g_cm3'],[[phase,*r] for phase in ['alpha','beta'] for r in dref['solid'][phase]])
exp=load('audit/density_reference/reiplinger2022_points.json')['points'];writecsv('Fig03_density_T/liquid_experiment.csv',['temperature_K','density_g_cm3','lower_g_cm3','upper_g_cm3'],[[r['T_K'],r['rho_g_cm3'],r['lower_g_cm3'],r['upper_g_cm3']] for r in exp])
# Fig.4: exact plotted structural-order trajectories.
for model,g in {'pbe':[(1750,'001'),(1775,'001'),(1800,'001')],'sol':[(1730,'001'),(1750,'001'),(1770,'001')],'scan':[(1850,'001'),(1850,'002'),(1860,'001')]}.items():
 for T,run in g:
  path=ROOT/(f'audit/oct26/pbe15k_melting_1775/analysis/15k_{T}.npz' if model=='pbe' and T==1775 else f'audit/oct26/pbe15k_melting/attempt2/analysis/15k_{T}.npz' if model=='pbe' else f'audit/melting/{model}_{T}_{run}.npz');sources[str(path)]=sha(path);z=np.load(path)
  t=z['time'] if model=='pbe' else z['step']*z['dt_ps'];f=z['fraction'][:,1] if model=='pbe' else z['solid_fraction'][:,1]
  writecsv(f'Fig04_melting/{model}_{T}K_{run}.csv',['time_ps','solid_like_fraction'],zip(t,f))
# Fig.5 and Fig.6.
thermo=load('audit/oct20/thermo_reprocessed.json');refs=load('audit/oct20/reference_values.json');density=load('audit/oct26/pbe15k_density_1873/analysis/statistics.json')
rows=[]
for model in ['pbe','sol','scan']:
 g=sorted(density['results']['15000'].values(),key=lambda r:r['x_Al']) if model=='pbe' else sorted([r for r in thermo if r['T']==1873 and r['dft']==model],key=lambda r:r['x_Al'])
 for r in g:rows.append([model,r['x_Al'],r['rho']['mean'] if model=='pbe' else r['rho'],r['Vex_cm3mol']['mean'] if model=='pbe' else r['Vex_cm3mol']])
writecsv('Fig05_density_x/models.csv',['functional','x_Al','density_g_cm3','excess_volume_cm3_mol'],rows)
rows=[]
for x,values in refs['density_table7'].items():
 for rho in values:
  x=float(x);v=((1-x)*47.867+x*26.9815385)/rho-(1-x)*47.867/4.184-x*26.9815385/2.08;rows.append([x,rho,v])
writecsv('Fig05_density_x/experiment.csv',['x_Al','density_g_cm3','excess_volume_cm3_mol'],rows)
mix=load('manuscript/supporting_data/revision_oct29_sources.json')['mixing']
writecsv('Fig06_enthalpy/models.csv',['functional','x_Al','mixing_enthalpy_kJ_mol','SEM_kJ_mol'],[[model,*r] for model,g in mix.items() for r in zip(g['x'],g['H'],g['SEM'])])
writecsv('Fig06_enthalpy/experiment.csv',['x_Al','mixing_enthalpy_kJ_mol'],refs['mixing_digitized_kJmol'].items())
# Fig.7: selected points and the exact sampling of fitted curves.
profile=load('audit/nov01/viscosity_article_combined/figure_profile.json')
for kind,xkey in [('composition','c_Ti'),('temperature','T')]:
 writecsv(f'Fig07_viscosity/{kind}.csv',['functional','x_Al' if kind=='composition' else 'temperature_K','viscosity_mPa_s','SE_mPa_s'],[[model,1-r[xkey] if kind=='composition' else r[xkey],r['eta_mPa_s'],r['eta_std_mPa_s']] for model,rs in profile[kind].items() for r in rs])
rows=[]
for model,rs in profile['temperature'].items():
 t=np.array([r['T'] for r in rs]);y=np.array([r['eta_mPa_s'] for r in rs]);sd=np.array([r['eta_std_mPa_s'] for r in rs])/y;X=np.column_stack([np.ones(len(t)),1000/t]);a,B=np.linalg.lstsq(X/sd[:,None],np.log(y)/sd,rcond=None)[0];tt=np.linspace(t.min(),t.max(),400)
 rows.extend([[model,T,float(np.exp(a+B*1000/T))] for T in tt])
writecsv('Fig07_viscosity/Arrhenius_curves.csv',['functional','temperature_K','viscosity_mPa_s'],rows)
writecsv('Fig07_viscosity/Takeda.csv',['x_Al','viscosity_mPa_s','measurement_Tmin_K','measurement_Tmax_K'],[[1-float(c),10**(a+b/1873),lo,hi] for c,(a,b,lo,hi) in profile['references']['takeda_table3'].items()])
t=np.linspace(1850,2120,200);eta=.61*np.exp(27000/(8.31*t));writecsv('Fig07_viscosity/Ishikawa_curve.csv',['temperature_K','viscosity_mPa_s','lower_mPa_s','upper_mPa_s'],zip(t,eta,eta*.897,eta*1.103))
(OUT/'source_hashes.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
print('Updated parity, metrics and main-text numbers; exported all seven main figures.')
