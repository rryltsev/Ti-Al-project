"""Freeze completed inputs for the 29 October revision and regenerate its figures."""
from pathlib import Path
import csv, hashlib, json, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
MAN=ROOT/'manuscript';FIG=MAN/'figures';DATA=MAN/'supporting_data'
PROJECT=Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project')
SOURCES={}
def track(p):
    p=Path(p);SOURCES[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def read(p):return json.loads(track(p).read_text(encoding='utf-8-sig'))
def copy(p,q):shutil.copyfile(track(p),q)
def save(fig,name):
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.png'),dpi=220,bbox_inches='tight');plt.close(fig)
STYLE={'pbe':('#2878B5','PBE'),'sol':('#39835B','PBEsol'),'scan':('#D55E00',r'r$^2$SCAN')}
plt.rcParams.update({'font.size':8,'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':7,
 'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})

metrics_source=ROOT/'testset/monitor/results/n30-9be7c27ae9e8c28b/metrics.json'
m=read(metrics_source)
assert len(m['common_ids'])==30 and m['models']['PBE']['checkpoint_step']==20000
copy(metrics_source,DATA/'test30_metrics.json')
copy(ROOT/'testset/monitor/data/latest.json',OUT/'test30_input_snapshot.json')
# Same checked plotting routine, using the newly frozen IDs and original prediction cache.
src=track(ROOT/'audit/oct26/manuscript_revision/prepare_parity.py').read_text(encoding='utf-8')
src=src.replace('23-structure','30-structure').replace('test23','test30').replace('PBE-15000','PBE-20000')
src=src.replace("'PBE 15k'","'PBE'").replace("MON/'data/latest.json'","OUT/'test30_input_snapshot.json'")
src=src.replace("figures/test30_parity.pdf","figures/testset_parity.pdf")
(OUT/'prepare_parity.py').write_text(src,encoding='utf-8')

rows=[]
for xc,label in [('PBE','PBE'),('PBEsol','PBEsol'),('r2SCAN',r'\rscan{}')]:
    for kind,model in [('metrics','Fine-tuned'),('pretrained_metrics','Pretrained')]:
        a=m[kind][xc]['all'];v=[a[q][p] for q in ['energy_meV_atom','force_eV_A','stress_GPa'] for p in ['mae','rmse']]
        rows.append(label+' & '+model+' & '+' & '.join(f'{x:.{n}f}' for x,n in zip(v,[2,2,4,4,3,3]))+r' \\')
table=r'''\begin{table*}[t]
\caption{Errors on 30 completed common geometries from the 50-configuration test design: 20 liquid, four hcp, three bcc, and three $\gamma$-TiAl structures. Energy errors use raw energies per atom; forces pool Cartesian components and stresses pool six independent tensor entries. The pretrained rows compare one unchanged model with each functional. No energy offsets are fitted.}
\label{tab:fit}
\begin{ruledtabular}
\begin{tabular}{llrrrrrr}
 & & \multicolumn{2}{c}{Energy (meV/atom)} & \multicolumn{2}{c}{Force (eV/\AA)} & \multicolumn{2}{c}{Stress (GPa)} \\
Reference & Model & MAE & RMSE & MAE & RMSE & MAE & RMSE \\
\hline
'''+ '\n'.join(rows)+r'''
\end{tabular}
\end{ruledtabular}
\end{table*}
'''
(MAN/'sections/testset_metrics_table.tex').write_text(table,encoding='utf-8')

# Three dispersions in one column, with experimental points only.
ph=np.load(track(ROOT/'audit/oct25/pbe_phonons_15k_equilibrium/15000/results.npz'))
old=read(ROOT/'audit/results/numerical_summary.json')
phroot=PROJECT/'results-scripts-logs/models_check_wsl2/results/phonons/dpa4/neo/finetune/alpha_Ti'
with track(PROJECT/'MD-results/phonons_DPA4-neo/finetune/alpha_Ti/pbe/phonon_bands_data_alpha_Ti_dpa4-neo-finetune_pbe.csv').open() as fp:
    ex=np.array([(float(r['Exp_x']),float(r['Exp_y'])) for r in csv.DictReader(fp) if r['Exp_x'] and r['Exp_y']])
fig,axs=plt.subplots(3,1,figsize=(3.35,5.8),layout='constrained')
for j,model in enumerate(STYLE):
    color,label=STYLE[model]
    if model=='pbe':q=ph['qaxis']/ph['qaxis'][-1];ticks=ph['ticks']/ph['qaxis'][-1];bands=ph['bands_THz']
    else:
        a=np.loadtxt(track(phroot/model/'phonon_bands.dat'));q=a[:,0]/a[-1,0];bands=a[:,1:]
        ticks=old['phonons']['alpha_Ti/'+model]['q_special_normalized']
    ax=axs[j];ax.plot(q,bands,c=color,lw=.8)
    ax.scatter(ex[:,0],ex[:,1],s=7,c='black',alpha=.85,zorder=3,label='Experiment')
    for t in ticks:ax.axvline(t,c='.82',lw=.5,zorder=0)
    ax.set(xticks=ticks,xticklabels=[r'$\Gamma$','K','M',r'$\Gamma$','A'],ylabel='Frequency (THz)',
           xlim=(0,1),ylim=(-.15,10),yticks=[0,2,4,6,8,10])
    ax.text(.03,.93,f'({chr(97+j)}) '+label,transform=ax.transAxes,va='top',fontsize=9)
    if j==0:ax.legend(frameon=False,loc='upper right',fontsize=7)
save(fig,'phonons_alpha_ti')

# Rebuild previously checked density/coexistence figures with functional-only labels.
density=read(ROOT/'audit/oct26/pbe15k_density_1873/analysis/statistics.json')
enthalpy=read(ROOT/'audit/oct25/pbe_full_enthalpy_gpu/analysis/statistics.json')
rho2000=enthalpy['windows']['production']['groups']['15000']['Ti']['rho']
rho1873=next(r['rho'] for r in density['results']['15000'].values() if r['x_Al']==0)
previous=track(ROOT/'audit/oct27/manuscript_revision/prepare_assets.py').read_text(encoding='utf-8')
meltingcode=previous.split('# All PBE melting trajectories in the article now use 15k.')[1].split('# Viscosity:')[0]
meltingcode=meltingcode.replace("+(' 15k' if model=='pbe' else '')",'')
exec(meltingcode)
densitycode=previous.split('# Elemental density:')[1].split('\nfor src,dest')[0]
densitycode='# Elemental density:'+densitycode
densitycode=densitycode.replace("+(' 5k' if model=='pbe' else '')",'').replace("label='PBE 15k'","label=None")
densitycode=densitycode.replace("mfc='white' if model=='pbe' else col","mfc=col")
exec(densitycode)

thermo=read(ROOT/'audit/oct20/thermo_reprocessed.json');refs=read(ROOT/'audit/oct20/reference_values.json')
fig,axs=plt.subplots(1,2,figsize=(7,2.9),layout='constrained')
for model in STYLE:
    color,label=STYLE[model]
    if model=='pbe':
        g=sorted(density['results']['15000'].values(),key=lambda r:r['x_Al'])
        x=[r['x_Al'] for r in g];rho=[r['rho']['mean'] for r in g];vex=[r['Vex_cm3mol']['mean'] for r in g]
    else:
        g=sorted([r for r in thermo if r['T']==1873 and r['dft']==model],key=lambda r:r['x_Al'])
        x=[r['x_Al'] for r in g];rho=[r['rho'] for r in g];vex=[r['Vex_cm3mol'] for r in g]
    axs[0].plot(x,rho,'o-',c=color,ms=3,lw=1,label=label);axs[1].plot(x,vex,'o-',c=color,ms=3,lw=1,label=label)
xx=[];yy=[];vv=[]
for x,values in refs['density_table7'].items():
    x=float(x)
    for rho in values:
        xx.append(x);yy.append(rho);vv.append(((1-x)*47.867+x*26.9815385)/rho-(1-x)*47.867/4.184-x*26.9815385/2.08)
axs[0].scatter(xx,yy,c='black',marker='s',s=14,zorder=5,label='Published densities')
axs[1].scatter(xx,vv,c='black',marker='s',s=14,zorder=5,label='From published densities')
axs[1].axhline(0,c='.6',lw=.7)
for ax in axs:ax.set(xlabel=r'$x_{\mathrm{Al}}$',xlim=(-.02,1.02));ax.legend(frameon=False,fontsize=7,loc='upper right')
axs[0].set(title='(a) Density, 1873 K',ylabel=r'$\rho$ (g cm$^{-3}$)')
axs[1].set(title='(b) Excess volume, 1873 K',ylabel=r'$V_m^{\mathrm{ex}}$ (cm$^3$ mol$^{-1}$)',ylim=(-1.14,.60))
save(fig,'liquid_density')

# Enthalpy: PBE/PBEsol numerical values unchanged, r2SCAN is the 60 ps series.
scan=read(ROOT/'audit/oct26/r2scan_enthalpy_extend30k/analysis/comparison.json')
copy(ROOT/'audit/oct26/r2scan_enthalpy_extend30k/analysis/comparison.json',DATA/'r2scan_enthalpy_60ps.json')
mix={};fig,ax=plt.subplots(figsize=(3.4,2.8),layout='constrained')
for model in STYLE:
    color,label=STYLE[model]
    if model=='pbe':
        g=sorted(enthalpy['windows']['production']['mixing']['15000'].values(),key=lambda r:r['x_Al'])
        x=[r['x_Al'] for r in g];y=[r['H']['mean'] for r in g];e=[r['H']['sem'] for r in g]
    elif model=='scan':
        g=sorted(scan['comparison'],key=lambda r:r['x_Al'])
        x=[0]+[r['x_Al'] for r in g]+[1];y=[0]+[r['deltaH_60_kJ_mol'] for r in g]+[0];e=[0]+[r['SEM_60_kJ_mol'] for r in g]+[0]
    else:
        g=sorted([r for r in thermo if r['T']==2000 and r['dft']==model],key=lambda r:r['x_Al'])
        x=[r['x_Al'] for r in g];y=[r['dH_external_pressure'] for r in g];e=[r['dH_external_sem_5ps'] for r in g]
    mix[model]={'x':x,'H':y,'SEM':e,'MAD':float(np.mean([abs(np.interp(float(a),x,y)-b) for a,b in refs['mixing_digitized_kJmol'].items()]))}
    ax.errorbar(x,y,yerr=e,fmt='o-',color=color,ms=3,lw=1,capsize=2,label=label)
ax.scatter([float(x) for x in refs['mixing_digitized_kJmol']],list(refs['mixing_digitized_kJmol'].values()),
    c='black',marker='s',s=14,label='Calorimetry (reproduced)',zorder=5)
ax.set(xlabel=r'$x_{\mathrm{Al}}$',ylabel=r'$\Delta H_{\mathrm{mix}}$ (kJ mol$^{-1}$)',xlim=(-.02,1.02),ylim=(-36,2),title='2000 K')
ax.legend(frameon=False,loc='upper center');save(fig,'mixing_enthalpy')

# Viscosity: only the requested N=15 alloy estimates plus the two pure-Ti 15k points.
visc_old=read(ROOT/'audit/oct25/viscosity/viscosity_merged.json')
pure=read(ROOT/'audit/oct27/viscosity_5k_vs_15k/comparison.json')
batch=read(ROOT/'audit/oct27/pbe15k_viscosity_iterative/collected/analysis/batch_0015/summary.json')
assert batch['n_replicas']==15 and batch['checkpoint_sha256']=='39506ecf42a49193717af8351521678b8665f662faf8fc374fec121f366feaaf'
copy(ROOT/'audit/oct27/pbe15k_viscosity_iterative/collected/analysis/batch_0015/summary.json',DATA/'pbe15k_viscosity_N15.json')
copy(ROOT/'audit/oct27/pbe15k_viscosity_iterative/protocol.json',DATA/'pbe15k_viscosity_protocol.json')
visc=[r for r in visc_old if r['dft']=='scan']+[dict(r['checkpoint_15k'],x_Al=0.) for r in pure['comparison']]
for r in batch['points']:
    visc.append(dict(r['windows']['40ps'],dft='pbe',T=1873,c_Ti=1-r['x_Al'],x_Al=r['x_Al'],n_replicas=15,production_ps=40))
fig,axs=plt.subplots(1,2,figsize=(7,2.95),layout='constrained')
for model in ['pbe','scan']:
    color,label=STYLE[model]
    for j in [0,1]:
        g=sorted([r for r in visc if r['dft']==model and (r['T']==1873 if j==0 else r['c_Ti']==1)],
            key=lambda r:1-r['c_Ti'] if j==0 else r['T'])
        axs[j].errorbar([1-r['c_Ti'] if j==0 else r['T'] for r in g],[r['eta_mPa_s'] for r in g],
            yerr=[r['eta_std_mPa_s'] for r in g],fmt='o-',c=color,ms=3.5,lw=1,capsize=2,label=label)
for c,(a,b,lo,hi) in refs['takeda_table3'].items():
    axs[0].scatter(1-float(c),10**(a+b/1873),s=23,marker='s',facecolors='black' if hi>=1873 else 'white',edgecolors='black',zorder=5)
axs[0].plot([],[],'ks',ms=4,label='Takeda fit');axs[0].plot([],[],'ks',mfc='white',ms=4,label='Fit extrapolated by 3 K')
t=np.linspace(1850,2120,150);eta=.61*np.exp(27000/(8.31*t))
axs[1].plot(t,eta,'k--',lw=1,label='Ishikawa fit*');axs[1].fill_between(t,eta*.897,eta*1.103,color='.7',alpha=.25)
axs[0].set(title='(a) Composition dependence, 1873 K',xlabel=r'$x_{\mathrm{Al}}$',ylabel=r'$\eta$ (mPa s)',xlim=(-.02,1.02),ylim=(0,8.3))
axs[1].set(title='(b) Pure Ti',xlabel='Temperature (K)',ylabel=r'$\eta$ (mPa s)',xlim=(1840,2130),ylim=(2.1,4.25))
axs[0].legend(fontsize=7,frameon=False,loc='lower left');axs[1].legend(fontsize=7,frameon=False,loc='upper right');save(fig,'viscosity')
(DATA/'viscosity_current_oct29.json').write_text(json.dumps(visc,indent=2)+'\n',encoding='utf-8')
summary={'date':'2026-10-29','test_n':30,'test_design_n':50,'test_PBE_step':20000,'viscosity_PBE_step':15000,
    'viscosity_alloy_n_replicas':15,'mixing':mix,'viscosity':visc,
    'sources_sha256':SOURCES,'benchmark_source':str(metrics_source)}
(OUT/'numerical_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
(DATA/'revision_oct29_sources.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'mixing_MAD':{k:v['MAD'] for k,v in mix.items()},'viscosity_PBE':[r for r in visc if r['dft']=='pbe']},indent=2))
