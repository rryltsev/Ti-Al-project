"""Inventory, existing SACF checks and provenance; independent of main fit audit."""
from analyze import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out={'sacf':{},'copies':{}}
fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,T in zip(axs,[1773,1943]):
    root=BASE/f'Ti-viscosity-r2SCAN/scan_{T}K_wsl2/scan'
    fs=sorted((root/'03_stress_autocorr/c_Ti_1.00').glob('visc*/sacf_*.data'),key=lambda f:int(f.parent.name[4:]))
    curves=[];hs=[]
    for f in fs:
        a=np.loadtxt(f,skiprows=4);assert a.shape==(5000,6) and np.array_equal(a[:,1],np.arange(5000))
        v=volume(f.parent/'mol.data');c=a[:,3:].mean(axis=1)*1e10*v*1e-30/(1.380649e-23*T)
        integral=np.r_[0,np.cumsum((c[1:]+c[:-1])*.5*1e-15)]*1000;curves.append(integral);hs.append(sha(f))
    cs=np.array(curves);mu=cs.mean(axis=0);se=cs.std(axis=0,ddof=1)/np.sqrt(len(cs));ts=np.arange(5000)*.001
    old=read(root/'04_viscosity_analysis/c_Ti_1.00/summary.json')
    out['sacf'][str(T)]={'n':len(cs),'unique_sacf':len(set(hs)),'existing_TDM':old,'integrals':{str(t):{'eta':float(mu[min(round(t*1000),4999)]),'replica_se':float(se[min(round(t*1000),4999)])} for t in [.25,.5,1.,2.,3.,4.,4.999]},'note':'Direct integration of stored three-channel ACF. Not STACIE; SE assumes independent replicas; cutoff is diagnostic.'}
    ax.plot(ts,mu,color='#315e92');ax.fill_between(ts,mu-se,mu+se,color='#315e92',alpha=.2);ax.axhline(old['viscosity_mPa_s'],color='#d76b3f',ls='--',label='Stored TDM fit')
    ax.set(xlabel='Integration limit (ps)',ylabel='Viscosity integral (mPa s)',title=f'{T} K: {len(cs)} stored ACFs');ax.legend();ax.grid(alpha=.2)
for T in [1873,2000]:
    src=BASE/f'Ti-viscosity-r2SCAN/scan_{T}K_wsl2/scan/03_stress_autocorr/c_Ti_1.00'
    other=BASE/f'models_check_wsl2/results/viscosity/dpa4/neo/finetune/AlTi/{T}K/scan/03_stress_autocorr/c_Ti_1.00'
    fs=list(src.glob('visc*/pressure_*.data'))
    out['copies'][str(T)]={'n':len(fs),'all_match_other_archive':all((other/f.parent.name/f.name).exists() and sha(f)==sha(other/f.parent.name/f.name) for f in fs)}
published=read(WORK/'manuscript/supporting_data/viscosity_temperature_current.json')['temperature']['scan']
out['published_arrhenius']=arrhenius([x['T'] for x in published],[x['eta_mPa_s'] for x in published],[x['eta_std_mPa_s'] for x in published])
fig.savefig(OUT/'stored_acf_integrals.png',dpi=180);fig.savefig(OUT/'stored_acf_integrals.svg');plt.close(fig)
save('supplement.json',out);print(json.dumps(out,indent=2),flush=True)
