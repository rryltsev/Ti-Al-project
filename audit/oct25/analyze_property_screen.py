"""Paired checkpoint changes, block uncertainties, provenance checks and figures."""
from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'audit/oct20/deps'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ase.units import GPa
OUT=HERE/'pbe_property_screen';DEST=OUT/'comparison';DEST.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text())
cfg=read(OUT/'protocol.json');rows=read(OUT/'rows.json')
keys=['5000','15000'];dirs={k:OUT/k for k in keys}
if (OUT/'retry15k/15000/summary.json').exists():dirs['15000']=OUT/'retry15k/15000'
eos={k:read(dirs[k]/'eos.json') for k in keys}
liq={k:read(dirs[k]/'liquid.json') for k in keys}
pred={k:np.load(dirs[k]/'predictions.npz',allow_pickle=False) for k in keys}
cached=np.load(OUT/'archived_predictions.npz',allow_pickle=False)
assert all(read(dirs[k]/'summary.json')['status']=='passed' for k in keys)
def stat(values):
    a=np.array(values);assert len(a)==50
    se={str(n):float(np.std(a.reshape(n,-1).mean(axis=1),ddof=1)/np.sqrt(n)) for n in [5,10]}
    return {'mean':float(a.mean()),'std':float(a.std(ddof=1)),'sem_blocks':se,'sem_max':max(se.values()),
            'first_half_mean':float(a[:25].mean()),'second_half_mean':float(a[25:].mean())}
results=[];perframe=[]
for x in cfg['liquid']['compositions_x_Al']:
    selected=[r for r in rows if r['x_Al']==x];de=[];dp=[];df=[];f5=[];archive_e=[];archive_f=[];archive_p=[]
    for r in selected:
        i=r['i'];a,b=liq['5000'][i],liq['15000'][i]
        assert a['i']==b['i']==i
        de.append(1000*(b['E_eV_atom']-a['E_eV_atom']));dp.append(b['P_GPa']-a['P_GPa'])
        fd=pred['15000'][f'forces_{i}']-pred['5000'][f'forces_{i}'];df.append(fd)
        f5.append(pred['5000'][f'forces_{i}'])
        archive_e.append(1000*(a['E_eV_atom']-float(cached[f'energy_{i}'])/r['n_atoms']))
        archive_f.append(pred['5000'][f'forces_{i}']-cached[f'forces_{i}'])
        st=cached[f'stress_{i}'];pc=-np.mean(st[:3])/GPa if st.shape==(6,) else -np.trace(st)/(3*GPa)
        archive_p.append(a['P_GPa']-float(pc))
        perframe.append({'i':i,'x_Al':x,'time_ps':r['time_ps_from_first_frame'],
                         'delta_E_meV_atom':de[-1],'delta_P_GPa':dp[-1],
                         'delta_F_component_RMS_eV_A':float(np.sqrt(np.mean(fd**2)))})
    drms=float(np.sqrt(np.mean(np.array(df)**2)));frms=float(np.sqrt(np.mean(np.array(f5)**2)))
    results.append({'x_Al':x,'x_Ti':1-x,'delta_E_meV_atom':stat(de),'delta_P_GPa':stat(dp),
                    'delta_F_component_RMS_eV_A':drms,'F5_component_RMS_eV_A':frms,'relative_delta_F_RMS':drms/frms,
                    'mean_V_A3_atom':float(np.mean([r['V_A3_atom'] for r in selected])),
                    'mean_T_K':float(np.mean([r['temperature_K'] for r in selected])),
                    'archive_comparison_5k':{'delta_E_meV_atom_mean':float(np.mean(archive_e)),
                      'delta_E_meV_atom_max_abs':float(np.max(abs(np.array(archive_e)))),
                      'delta_F_RMS_eV_A':float(np.sqrt(np.mean(np.array(archive_f)**2))),
                      'delta_P_GPa_max_abs':float(np.max(abs(np.array(archive_p))))}})
for r in results:
    x=r['x_Al'];d=r['delta_E_meV_atom'];ti=results[0]['delta_E_meV_atom'];al=results[-1]['delta_E_meV_atom']
    value=d['mean']-(1-x)*ti['mean']-x*al['mean']
    se=np.sqrt(d['sem_max']**2+(1-x)**2*ti['sem_max']**2+x*x*al['sem_max']**2)
    r['endpoint_corrected_delta_E_meV_atom']={'mean':value if 0<x<1 else 0.,'sem':float(se) if 0<x<1 else 0.}
crystals=[]
for phase in ['Ti_hcp','Ti_bcc','Al_fcc']:
    a=eos['5000'][phase]['equilibrium'];b=eos['15000'][phase]['equilibrium']
    crystals.append({'phase':phase,'5000':a,'15000':b,'delta_V_percent':100*(b['V_A3_atom']/a['V_A3_atom']-1),
                     'delta_E_meV_atom':1000*(b['E_eV_atom']-a['E_eV_atom']),
                     'delta_B_GPa':b['B_GPa']-a['B_GPa']})
gaps={k:1000*(eos[k]['Ti_bcc']['equilibrium']['E_eV_atom']-eos[k]['Ti_hcp']['equilibrium']['E_eV_atom']) for k in keys}
summary={'definition':'15k minus 5k; model differences, not errors relative to DFT',
         'liquid_ensemble':'Archived PBE 2000 K NPT trajectory; no 15k re-equilibration',
         'uncertainty':'SEM from 5 or 10 contiguous blocks of 50 spaced frames; use larger SEM. Each state has one trajectory only.',
         'crystals':crystals,'bcc_minus_hcp_meV_atom':gaps,'liquids':results,
         'computing':{k:{f:read(dirs[k]/'summary.json')[f] for f in ['wall_seconds','job_id','node']} for k in keys}}
(DEST/'analysis.json').write_text(json.dumps(summary,indent=2))
(DEST/'per_frame.json').write_text(json.dumps(perframe,indent=2))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':200})
colors={'5000':'#2473aa','15000':'#d26b24'}
fig,axs=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
for ax,phase,title in zip(axs,['Ti_hcp','Ti_bcc','Al_fcc'],[r'$\alpha$-Ti (relaxed $c/a$)',r'$\beta$-Ti (bcc symmetry)','Al (fcc symmetry)']):
    for k in keys:
        curve=eos[k][phase]['curve'];refphase='Ti_hcp' if phase.startswith('Ti') else phase
        e0=eos[k][refphase]['equilibrium']['E_eV_atom']
        ax.plot([r['V_A3_atom'] for r in curve],[1000*(r['E_eV_atom']-e0) for r in curve],'o-',ms=3,c=colors[k],label=k)
        eq=eos[k][phase]['equilibrium'];ax.plot(eq['V_A3_atom'],1000*(eq['E_eV_atom']-e0),'*',ms=10,c=colors[k])
    ylabel=r'$E-E_{\alpha,0}$ (meV/atom)' if phase.startswith('Ti') else r'$E-E_{\mathrm{fcc},0}$ (meV/atom)'
    ax.set(title=title,xlabel=r'Volume ($\mathrm{\AA^3}$/atom)',ylabel=ylabel);ax.legend(title='Checkpoint');ax.grid(alpha=.2)
fig.savefig(DEST/'eos_5k_15k.png');fig.savefig(DEST/'eos_5k_15k.pdf');plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(9,6.4),layout='constrained');xx=[r['x_Al'] for r in results]
panels=[('delta_E_meV_atom',r'$\langle E_{15k}-E_{5k}\rangle$ (meV/atom)','mean','sem_max'),
        ('endpoint_corrected_delta_E_meV_atom','Endpoint-corrected energy change\n(meV/atom)','mean','sem'),
        ('delta_P_GPa',r'$\langle P_{15k}-P_{5k}\rangle$ (GPa)','mean','sem_max')]
for ax,(field,label,m,s) in zip(axs.flat,panels):
    ax.errorbar(xx,[r[field][m] for r in results],yerr=[r[field][s] for r in results],fmt='o-',capsize=3,color=colors['15000'])
    ax.axhline(0,c='gray',lw=.7);ax.set(xlabel=r'$x_{\mathrm{Al}}$',ylabel=label);ax.grid(alpha=.2)
ax=axs.flat[3];ax.plot(xx,[1000*r['delta_F_component_RMS_eV_A'] for r in results],'o-',color=colors['15000'])
ax.set(xlabel=r'$x_{\mathrm{Al}}$',ylabel='Force-component difference RMS\n'+r'(meV/$\mathrm{\AA}$)');ax.grid(alpha=.2)
fig.suptitle('15k minus 5k on identical liquid snapshots, 2000 K\nEnergy/pressure error bars: block SEM; no new 15k MD',fontsize=11)
fig.savefig(DEST/'liquid_changes_5k_15k.png');fig.savefig(DEST/'liquid_changes_5k_15k.pdf');plt.close(fig)
lines=['PBE 5k -> 15k: CPU property screening, 2026-10-25',
       'All changes below are model differences; they are not DFT errors.',
       'EOS equilibrium: pressure root, hcp shape relaxed; cubic symmetry retained for bcc/fcc.',
       'Phase: V5 / V15 (A^3/atom), B5 / B15 (GPa), delta V (%)']
for r in crystals:lines.append(f"{r['phase']}: {r['5000']['V_A3_atom']:.6f} / {r['15000']['V_A3_atom']:.6f}, {r['5000']['B_GPa']:.3f} / {r['15000']['B_GPa']:.3f}, {r['delta_V_percent']:+.4f}%")
lines.append('bcc-hcp equilibrium energy gaps (meV/atom): '+json.dumps(gaps))
lines+=['Liquid 2000 K: 50 spaced frames per state, same geometries for both models.',
         'x_Al: delta E, endpoint-corrected delta E (meV/atom), delta P (GPa), force difference RMS (eV/A), relative RMS (%)']
for r in results:
    d=r['endpoint_corrected_delta_E_meV_atom'];p=r['delta_P_GPa']
    lines.append(f"{r['x_Al']:.1f}: {r['delta_E_meV_atom']['mean']:+.4f}, {d['mean']:+.4f} +/- {d['sem']:.4f}, {p['mean']:+.5f} +/- {p['sem_max']:.5f}, {r['delta_F_component_RMS_eV_A']:.5f}, {100*r['relative_delta_F_RMS']:.2f}")
lines+=['Uncertainties are block SEM, not confidence intervals, and do not capture independent-trajectory variability.',
         'The endpoint-corrected liquid energy shift is not a recalculated equilibrium mixing enthalpy.',
         'Pressure is configurational, positive in compression; kinetic contribution cancels in the paired comparison.',
         'The bcc-hcp static gap is not a melting-temperature prediction.',
         'Archive consistency (native 5k minus stored predictions):']
lines += [f"x_Al={r['x_Al']}: "+json.dumps(r['archive_comparison_5k']) for r in results]
(DEST/'REPORT.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
