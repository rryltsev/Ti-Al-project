from pathlib import Path
import sys,json,os
OUT=Path(__file__).parent
sys.path.insert(0,str(OUT/'deps'))
os.environ['MPLBACKEND']='Agg'
import numpy as np
import matplotlib.pyplot as plt
rows=json.loads((OUT/'thermo_reprocessed.json').read_text())
st=json.loads((OUT/'viscosity_merged.json').read_text())
refs=json.loads((OUT/'reference_values.json').read_text())
styles={'pbe':('#2878B5','PBE'),'scan':('#D55E00',r'r$^2$SCAN'),'sol':('#39835B','PBEsol')}
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.grid':True,'grid.alpha':.15,'savefig.dpi':180})
fig,axes=plt.subplots(1,3,figsize=(15.5,4.8),layout='constrained')
for m,(col,label) in styles.items():
    g=sorted((r for r in rows if r['T']==2000 and r['dft']==m),key=lambda r:r['x_Al'])
    axes[0].errorbar([r['x_Al'] for r in g],[r['dH_external_pressure'] for r in g],yerr=[r['dH_external_sem_5ps'] for r in g],fmt='o-',ms=4,c=col,label=label,capsize=2)
    g=sorted((r for r in rows if r['T']==1873 and r['dft']==m),key=lambda r:r['x_Al'])
    axes[1].plot([r['x_Al'] for r in g],[r['rho'] for r in g],'o-',ms=4,c=col,label=label)
    axes[2].plot([r['x_Al'] for r in g],[r['Vex_cm3mol'] for r in g],'o-',ms=4,c=col,label=label)
mix=refs['mixing_digitized_kJmol'];axes[0].scatter([float(x) for x in mix],list(mix.values()),marker='s',s=30,c='black',label='Exp., reproduced in Witusiewicz (2008)',zorder=5)
rho=refs['density_table7'];axes[1].scatter([float(x) for x,vs in rho.items() for v in vs],[v for vs in rho.values() for v in vs],marker='s',s=22,c='black',label='Brillo et al. (2025), Table 7',zorder=5)
# Excess volumes recomputed consistently from the printed density column.
xs=[];vs=[];vt=47.867/4.184;va=26.9815385/2.08
for x,vals in rho.items():
    x=float(x)
    for val in vals:
        xs.append(x);vs.append(((1-x)*47.867+x*26.9815385)/val-((1-x)*vt+x*va))
axes[2].scatter(xs,vs,marker='s',s=22,c='black',label='From published densities',zorder=5)
for ax,title,yl in zip(axes,['Mixing enthalpy, 2000 K','Density, 1873 K','Excess molar volume, 1873 K'],[r'$\Delta H_{mix}$ (kJ mol$^{-1}$)',r'$\rho$ (g cm$^{-3}$)',r'$V^E$ (cm$^3$ mol$^{-1}$)']):
    ax.set(xlabel=r'$x_{Al}$',ylabel=yl,title=title,xlim=(-.02,1.02));ax.legend(fontsize=8,loc='best')
fig.suptitle('Ti-Al liquid thermodynamics: audit of 20 October 2026',fontsize=15)
fig.savefig(OUT/'thermodynamics_overview.png');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11.8,4.8),layout='constrained')
for m in ['pbe','scan']:
    col,label=styles[m];g=[r for r in st if r['T']==1873 and r['dft']==m]
    axes[0].errorbar([r['c_Ti'] for r in g],[r['eta_mPa_s'] for r in g],yerr=[r['eta_std_mPa_s'] for r in g],fmt='o-',c=col,label=label,capsize=3,ms=4)
    g=[r for r in st if r['c_Ti']==1 and r['dft']==m]
    axes[1].errorbar([r['T'] for r in g],[r['eta_mPa_s'] for r in g],yerr=[r['eta_std_mPa_s'] for r in g],fmt='o-',c=col,label=label,capsize=3,ms=5)
for c,(a,b,lo,hi) in refs['takeda_table3'].items():
    c=float(c);v=10**(a+b/1873);axes[0].scatter(c,v,s=45,marker='s',edgecolors='black',facecolors='black' if hi>=1873 else 'white',zorder=5)
axes[0].plot([],[],'ks',label='Takeda (2025) fits at 1873 K')
axes[0].plot([],[],marker='s',mfc='white',mec='black',ls='',label='3 K extrapolation of fits')
Ts=np.linspace(1850,2120,150);eta=.61*np.exp(27000/(8.31*Ts))
axes[1].plot(Ts,eta,'k--',label='Ishikawa fit, from supplied review')
axes[1].fill_between(Ts,eta*.897,eta*1.103,color='grey',alpha=.15,label='Reported 10.3% reference band')
axes[0].set(xlabel=r'$c_{Ti}$',ylabel=r'$\eta$ (mPa s)',title='Composition dependence, 1873 K',xlim=(0,1.02),ylim=(0,8.5))
axes[1].set(xlabel='Temperature (K)',ylabel=r'$\eta$ (mPa s)',title='Pure Ti',ylim=(2.1,4.2))
for ax in axes:ax.legend(fontsize=8,loc='best')
fig.suptitle('STACIE viscosity: supplied central values and estimator uncertainties',fontsize=14)
fig.savefig(OUT/'viscosity_overview.png');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11.8,4.7),layout='constrained')
for m in ['pbe','scan']:
    col,label=styles[m];g=[r for r in st if r['dft']==m and 'tdm_mPa_s' in r]
    axes[0].scatter([r['eta_mPa_s'] for r in g],[r['tdm_mPa_s'] for r in g],c=col,label=label)
    g=[r for r in st if r['dft']==m and r['T']==1873]
    axes[1].plot([r['c_Ti'] for r in g],[r['neff'] for r in g],'o-',c=col,label=label)
axes[0].plot([0,8],[0,8],'k--',lw=1);axes[0].set(xlabel='STACIE (mPa s)',ylabel='TDM (mPa s)',title='25 matched case summaries')
axes[1].axhline(60,c='black',ls='--',label='Recommended guide: 20 parameters\nper fit parameter = 60 frequency points')
# Correct the wording: N_eff concerns frequency points, not independent trajectories.
axes[1].lines[-1].set_label(r'Guide: $N_{eff}>20P=60$, $P=3$')
axes[1].set(xlabel=r'$c_{Ti}$',ylabel=r'$N_{eff}$ (frequency points)',title='STACIE fit resolution, 1873 K')
for ax in axes:ax.legend(fontsize=9)
fig.savefig(OUT/'viscosity_diagnostics.png');plt.close(fig)
print('Saved three audit figures')
