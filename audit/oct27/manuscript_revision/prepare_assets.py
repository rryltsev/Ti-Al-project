"""Update only properties with completed PBE 15k calculations. No MD is run."""
from pathlib import Path
import csv, hashlib, json, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
MAN=ROOT/'manuscript'; FIG=MAN/'figures'; DATA=MAN/'supporting_data'
PROJECT=Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project')
SOURCES={}
def track(p):
    p=Path(p); SOURCES[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest(); return p
def read(p): return json.loads(track(p).read_text(encoding='utf-8'))
def save(fig,name):
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.png'),bbox_inches='tight',dpi=220)
    plt.close(fig)
STYLE={'pbe':('#2878B5','PBE'),'sol':('#39835B','PBEsol'),'scan':('#D55E00',r'r$^2$SCAN')}
plt.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':8,
    'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})

density=read(ROOT/'audit/oct26/pbe15k_density_1873/analysis/statistics.json')
enthalpy=read(ROOT/'audit/oct25/pbe_full_enthalpy_gpu/analysis/statistics.json')
rho2000=enthalpy['windows']['production']['groups']['15000']['Ti']['rho']
rho1873=next(r['rho'] for r in density['results']['15000'].values() if r['x_Al']==0)
refs=read(ROOT/'audit/oct20/reference_values.json')
visc=read(ROOT/'audit/oct25/viscosity/viscosity_merged.json')
comparison=read(ROOT/'audit/oct27/viscosity_5k_vs_15k/comparison.json')
updates={r['T']:r['checkpoint_15k'] for r in comparison['comparison']}
for r in comparison['comparison']: track(r['source_15k'])
current=[]
for old in visc:
    r=dict(old)
    r['checkpoint']='5k' if r['dft']=='pbe' else 'final'
    if r['dft']=='pbe' and r['c_Ti']==1 and r['T'] in updates:
        r=dict(updates[r['T']],checkpoint='15k')
    current.append(r)
(DATA/'viscosity_current_oct27.json').write_text(json.dumps(current,indent=2)+'\n',encoding='utf-8')

# Reuse the verified density figure made in the calculation chat.
shutil.copyfile(track(FIG/'liquid_density_pbe15k.pdf'),FIG/'liquid_density.pdf')
shutil.copyfile(track(FIG/'liquid_density_pbe15k.png'),OUT/'liquid_density.png')

# All PBE melting trajectories in the article now use 15k.
fig,axs=plt.subplots(1,3,figsize=(7,2.65),sharey=True,layout='constrained')
chosen={'pbe':[(1750,'001'),(1775,'001'),(1800,'001')],
        'sol':[(1730,'001'),(1750,'001'),(1770,'001')],
        'scan':[(1850,'001'),(1850,'002'),(1860,'001')]}
melting={}
for j,model in enumerate(chosen):
    for i,(T,run) in enumerate(chosen[model]):
        if model=='pbe':
            p=ROOT/(f'audit/oct26/pbe15k_melting_1775/analysis/15k_{T}.npz' if T==1775
                    else f'audit/oct26/pbe15k_melting/attempt2/analysis/15k_{T}.npz')
            a=np.load(track(p)); t=a['time']; f=a['fraction'][:,1]
        else:
            a=np.load(track(ROOT/f'audit/melting/{model}_{T}_{run}.npz'))
            t=a['step']*a['dt_ps']; f=a['solid_fraction'][:,1]
        axs[j].plot(t,f,color=['#2878B5','#9467BD','#D55E00'][i],
                    ls='--' if run=='002' else '-',lw=1.1,label=f'{T} K'+(' (2)' if run=='002' else ''))
        melting[f'{model}_{T}_{run}']={'time_ps':t.tolist(),'solid_fraction':f.tolist()}
    axs[j].set(xlabel='Time (ps)',title=f'({chr(97+j)}) '+STYLE[model][1]+(' 15k' if model=='pbe' else ''),
               xlim=(0,100 if model=='pbe' else 50),ylim=(-.03,1.04))
    axs[j].legend(fontsize=7,frameon=False,loc='center right')
axs[0].set_ylabel('Solid-like fraction');save(fig,'melting_ti')

# Viscosity: filled squares identify the two 15k state points. Never connect mixed checkpoints.
fig,axs=plt.subplots(1,2,figsize=(7,3.1),layout='constrained')
for model,checkpoint in [('pbe','5k'),('pbe','15k'),('scan','final')]:
    col,label=STYLE[model]; label+=(' '+checkpoint if model=='pbe' else '')
    for j in [0,1]:
        g=sorted([r for r in current if r['dft']==model and r['checkpoint']==checkpoint and
                  (r['T']==1873 if j==0 else r['c_Ti']==1)],key=lambda r:1-r['c_Ti'] if j==0 else r['T'])
        x=[1-r['c_Ti'] if j==0 else r['T'] for r in g]
        marker='s' if checkpoint=='15k' else 'o'
        # The retained 5k temperatures straddle a replaced point: show them without a connecting line.
        ls='none' if model=='pbe' and j==1 and checkpoint=='5k' else '-'
        axs[j].errorbar(x,[r['eta_mPa_s'] for r in g],yerr=[r['eta_std_mPa_s'] for r in g],
            marker=marker,ls=ls,c=col,mfc='white' if checkpoint=='5k' else col,ms=3.7,capsize=2,lw=1,label=label)
for c,(a,b,lo,hi) in refs['takeda_table3'].items():
    axs[0].scatter(1-float(c),10**(a+b/1873),s=23,marker='s',
                   facecolors='black' if hi>=1873 else 'white',edgecolors='black',zorder=5)
axs[0].plot([],[],'ks',ms=4,label='Takeda fit')
axs[0].plot([],[],'ks',mfc='white',ms=4,label='Fit extrapolated by 3 K')
t=np.linspace(1850,2120,150);eta=.61*np.exp(27000/(8.31*t))
axs[1].plot(t,eta,'k--',lw=1,label='Ishikawa fit*')
axs[1].fill_between(t,eta*.897,eta*1.103,color='.7',alpha=.25)
axs[0].set(title='(a) Composition dependence, 1873 K',xlabel=r'$x_{\mathrm{Al}}$',ylabel=r'$\eta$ (mPa s)',xlim=(-.02,1.02),ylim=(0,8.3))
axs[1].set(title='(b) Pure Ti',xlabel='Temperature (K)',ylabel=r'$\eta$ (mPa s)',xlim=(1840,2130),ylim=(2.1,4.25))
axs[0].legend(fontsize=7,frameon=False,loc='lower left')
axs[1].legend(fontsize=7,frameon=False,loc='upper right');save(fig,'viscosity')

# Elemental density: replace the available 2000 K point and add the new 1873 K endpoint.
drefs=read(ROOT/'audit/density_reference/references.json');lref=drefs['liquid']
points=read(ROOT/'audit/density_reference/reiplinger2022_points.json')['points']
fig,axs=plt.subplots(1,2,figsize=(7,3.3),layout='constrained')
tirows={}
for model in STYLE:
    col,label=STYLE[model]
    with track(PROJECT/f'MD-results/density_DPA4-neo/density_scan_{model}.csv').open() as fp:
        rows=list(csv.DictReader(fp,delimiter=';'))
    tirows[model]=rows
    for lo,hi,marker,ls in [(300,1100,'o','-'),(1200,1900,'^','--'),(2000,2500,'s','-')]:
        g=[r for r in rows if lo<=float(r['T_target_K'])<=hi and not (model=='pbe' and float(r['T_target_K'])==2000)]
        ax=axs[1] if lo==2000 else axs[0]
        ax.plot([float(r['T_target_K']) for r in g],[float(r['density_g_cm3']) for r in g],
                c=col,marker=marker,ls=ls,ms=3,lw=1,mfc='white' if model=='pbe' else col,
                label=label+(' 5k' if model=='pbe' else '') if lo in [300,2000] else None)
axs[1].errorbar([1873,2000],[rho1873['mean'],rho2000['mean']],yerr=[rho1873['sem'],rho2000['sem']],
    color=STYLE['pbe'][0],fmt='s-',ms=4,lw=1,capsize=2,label='PBE 15k')
for phase in ['alpha','beta']:
    a=np.array(drefs['solid'][phase]);axs[0].plot(a[:,0],a[:,1],c='black',marker='D',ms=3,ls=':',lw=.9,
        label="Zinov'ev (1989)" if phase=='alpha' else None)
tt=np.linspace(*lref['plotted_range_K'],200)
rr=(lref['c1_kg_m3']+lref['c2_kg_m3_K']*(tt-lref['Tm_K']))/1000
u=lref['expanded_relative_uncertainty']
axs[1].fill_between(tt,rr*(1-u),rr*(1+u),color='.85',zorder=0,label=r'$\pm2\%$ (95% coverage)')
axs[1].plot(tt,rr,c='black',lw=1.2,label='Ntonti et al. (2024)')
axs[1].errorbar([p['T_K'] for p in points],[p['rho_g_cm3'] for p in points],
    yerr=[[p['rho_g_cm3']-p['lower_g_cm3'] for p in points],[p['upper_g_cm3']-p['rho_g_cm3'] for p in points]],
    fmt='o',ms=2.6,mfc='white',mec='black',mew=.6,ecolor='.3',elinewidth=.55,capsize=1.3,zorder=4,
    label='Reiplinger & Brillo (2022)')
for T,label in [(1155,r'$\alpha$--$\beta$: 1155 K'),(1943,r'$T_m$: 1943 K')]:
    axs[0].axvline(T,color='.45',ls='-.',lw=.8,zorder=0)
    axs[0].text(T-25,4.82,label,rotation=90,ha='right',va='top',fontsize=7,color='.3')
axs[0].set(title=r'(a) $\alpha$- and $\beta$-Ti starting cells',xlim=(220,2010),ylim=(4.22,4.85))
axs[1].set(title='(b) Separately prepared liquid',xlim=(1835,2530),ylim=(3.87,4.85))
axs[0].legend(frameon=True,facecolor='white',edgecolor='none',framealpha=1,ncol=2,fontsize=7,loc='lower left',borderpad=.3)
axs[1].legend(frameon=False,fontsize=6.7,loc='upper right',labelspacing=.27,handlelength=1.5)
for ax in axs:ax.set(xlabel='Temperature (K)',ylabel=r'Density (g cm$^{-3}$)')
save(fig,'density_ti')

for src,dest in [
    ('audit/oct26/pbe15k_density_1873/analysis/statistics.json','pbe15k_density1873_statistics.json'),
    ('audit/oct26/pbe15k_melting/attempt2/analysis/summary.json','pbe15k_melting_summary.json'),
    ('audit/oct26/pbe15k_melting_1775/analysis/summary.json','pbe15k_melting1775_summary.json'),
    ('audit/oct26/pbe15k_melting_1775/analysis/assessment.json','pbe15k_melting1775_assessment.json'),
    ('audit/oct27/viscosity_5k_vs_15k/comparison.json','pbe15k_viscosity_comparison.json'),
    ('audit/oct27/viscosity_5k_vs_15k/source_manifest.json','pbe15k_viscosity_source_manifest.json')]:
    shutil.copyfile(track(ROOT/src),DATA/dest)
for p in ['audit/oct26/pbe15k_density_1873/protocol.json','audit/oct26/pbe15k_melting/attempt2/protocol.json',
          'audit/oct26/pbe15k_melting_1775/protocol.json','testset/monitor/model_registry.json',
          'testset/monitor/results/n23-1a6fdc00d4d0b30a/metrics.json']:
    if (ROOT/p).exists(): track(ROOT/p)
latest=read(ROOT/'testset/monitor/data/latest.json')
assert len(latest['common_ids'])==23, 'New benchmark: regenerate metrics and parity plot'
summary={'date':'2026-10-27','PBE_step':15000,'rho_Ti_1873':{k:rho1873[k] for k in ['mean','sem']},
    'rho_Ti_2000':{k:rho2000[k] for k in ['mean','sem']},'density_reference_metrics':density['reference_metrics'],
    'melting_bracket_K':[1750,1775],'benchmark_common_ids':latest['common_ids'],
    'viscosity_replaced_states':[[1873,'Ti'],[2000,'Ti']],
    'remaining_PBE_5k':['Ti thermal density: solid 300--1900 K, liquid 2100--2500 K',
        'Ti3Al phonons','STACIE alloy viscosities at 1873 K; Ti at 1941 and 2100 K; historical TDM']}
(OUT/'numerical_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
(DATA/'revision_oct27_sources.json').write_text(json.dumps({'summary':summary,'source_sha256':SOURCES},indent=2)+'\n',encoding='utf-8')
(OUT/'melting_plotted_data.json').write_text(json.dumps(melting,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))
