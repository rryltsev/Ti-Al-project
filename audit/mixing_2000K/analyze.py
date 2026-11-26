"""Audit archived 2000 K trajectories. No MD, smoothing, or invented samples.

Run with Python 3.12 and NumPy, SciPy, Matplotlib, ASE installed.
Time is relative to production start; earlier equilibration was not recorded.
"""
from pathlib import Path
import csv
import hashlib
import json
import os
import sys

OUT = Path(__file__).resolve().parent
WORK = OUT.parents[1]
sys.path.insert(0, str(WORK / 'audit/oct20/deps'))
os.environ['MPLBACKEND'] = 'Agg'
os.environ['MPLCONFIGDIR'] = str(OUT / '.mplconfig')
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from ase import units
from ase.io.trajectory import Trajectory

ROOT = Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project')
PIPE = ROOT / 'results-scripts-logs/models_check_wsl2'
BASE = PIPE / 'results/mixing_enthalpy/dpa4/neo/finetune/AlTi/2000K'
MODELS = ['pbe', 'sol', 'scan']
STYLE = {'pbe': ('#2878B5', 'PBE'), 'sol': ('#39835B', 'PBEsol'),
         'scan': ('#D55E00', r'r$^2$SCAN')}
CONV = 96.48533212331
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})


def save(fig, name):
    for ext in ['png', 'pdf']:
        fig.savefig(OUT / f'{name}.{ext}', dpi=190, bbox_inches='tight')
    plt.close(fig)


def write_csv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def block(a, size):
    assert len(a) % size == 0
    return np.asarray(a).reshape(-1, size).mean(axis=1)


def sem(a):
    return float(np.std(a, ddof=1) / np.sqrt(len(a)))


def bh(p):
    p = np.asarray(p)
    order = np.argsort(p)
    q = np.empty_like(p)
    q[order] = np.minimum(1, np.minimum.accumulate(
        (p[order] * len(p) / np.arange(1, len(p)+1))[::-1])[::-1])
    return q


sources = []
def track(p):
    sources.append({'path': str(p), 'bytes': p.stat().st_size,
                    'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    return p


for source in [PIPE / 'utils/md_runner.py', PIPE / 'scripts/mixing_enthalpy.py',
               PIPE / 'utils/trajectory_analyzer.py']:
    track(source)
series, rows, tests, caches, block_rows = {}, [], [], [], []
previous = json.loads(track(WORK / 'audit/oct20/thermo_reprocessed.json').read_text())
for model in MODELS:
    summary = json.loads(track(BASE / model / 'summary.json').read_text())
    for path in sorted((BASE / model).glob('*.traj')):
        comp = path.stem.rsplit('_', 1)[0]
        with Trajectory(str(track(path))) as tr:
            assert len(tr) == 301, (path, len(tr))
            x = float(np.mean(tr[0].numbers == 13))
            a = []
            for atoms in tr:
                assert len(atoms) == 200
                ep = atoms.get_potential_energy()/len(atoms)
                et = atoms.get_total_energy()/len(atoms)
                v = atoms.get_volume()/len(atoms)
                pressure = -atoms.get_stress(include_ideal_gas=True)[:3].mean()/units.GPa
                rho = atoms.get_masses().sum()/atoms.get_volume()*1.66053906660
                a.append([ep, et, v, atoms.get_temperature(), pressure, rho,
                          (et + units.bar*v)*CONV,
                          (et - v*pressure*6.242e-3)*96.485])
        a = np.array(a)
        key = (model, x)
        series[key] = a
        old = next(r for r in previous if r['T'] == 2000 and r['dft'] == model and r['x_Al'] == x)
        assert abs(a[:,6].mean()-old['H_external_pressure']) < 1e-8
        assert abs(a[:,7].mean()-summary[comp]['H_kJ_per_mol']) <= .00501
        cache = BASE / model / 'cache' / f'{comp}_relaxed.traj'
        with Trajectory(str(track(cache))) as tr:
            caches.append({'model': model, 'composition': comp, 'frames': len(tr),
                           'T_K': tr[0].get_temperature()})
            assert len(tr) == 1
        h = a[1:,6]
        r = {'model': model, 'composition': comp, 'x_Al': x, 'n_frames': len(a),
             'H_kJ_mol': float(a[:,6].mean()), 'H_sem_5ps': sem(block(h,50)),
             'T_K': float(a[:,3].mean()), 'rho_g_cm3': float(a[:,5].mean()),
             'H_second_minus_first_15ps': float(h[150:].mean()-h[:150].mean()),
             'H_legacy': summary[comp]['H_kJ_per_mol']}
        rows.append(r)
        for obs, index in [('H',6), ('T',3), ('rho',5)]:
            values = a[1:,index]
            b = block(values,50)
            welch = stats.ttest_ind(b[:3], b[3:], equal_var=False)
            trend = stats.linregress(np.arange(6)*5+2.55, b)
            tests.append({'model':model, 'x_Al':x, 'observable':obs,
                          'half_difference':float(b[3:].mean()-b[:3].mean()),
                          'half_p':float(welch.pvalue), 'slope_per_ps':float(trend.slope),
                          'slope_p':float(trend.pvalue)})
        for size in [1,5,10,20,30,50,100]:
            block_rows.append({'model':model,'x_Al':x,'block_ps':size*.1,
                               'n_blocks':300//size,'H_sem':sem(block(h,size))})
    print('Read and validated', model, flush=True)

for field in ['half','slope']:
    for t, q in zip(tests, bh([t[field+'_p'] for t in tests])):
        t[field+'_q_BH'] = float(q)

cutrows, covariances = [], {}
for model in MODELS:
    g = sorted([r for r in rows if r['model']==model], key=lambda r:r['x_Al'])
    al, ti = g[-1],g[0]
    for r in g:
        x = r['x_Al']
        r['dH_mix_kJ_mol'] = r['H_kJ_mol']-x*al['H_kJ_mol']-(1-x)*ti['H_kJ_mol']
        r['dH_sem_5ps'] = 0. if x in [0,1] else float(np.sqrt(
            r['H_sem_5ps']**2+x*x*al['H_sem_5ps']**2+(1-x)**2*ti['H_sem_5ps']**2))
        r['dH_legacy'] = r['H_legacy']-x*al['H_legacy']-(1-x)*ti['H_legacy']
        # Overlapping cutoffs are a sensitivity check, not independent replicas.
        for cut in [0,1,2,5,10,15]:
            sl = slice(int(cut*10)+1, None)
            avg = lambda xx: series[model,xx][sl,6].mean()
            value = avg(x)-x*avg(1.)-(1-x)*avg(0.)
            cutrows.append({'model':model,'x_Al':x,'discard_production_ps':cut,
                            'retained_ps':30-cut,'dH_mix_kJ_mol':float(value)})
        for tag, sl in [('first15',slice(1,151)),('last15',slice(151,301))]:
            means = lambda xx: series[model,xx][sl,6].mean()
            r['dH_'+tag] = float(means(x)-x*means(1.)-(1-x)*means(0.))
        r['dH_half_difference'] = r['dH_last15']-r['dH_first15']
    # Common endpoints correlate errors between different compositions.
    cov = np.zeros((11,11))
    for i, ri in enumerate(g):
        for j, rj in enumerate(g):
            x,y=ri['x_Al'],rj['x_Al']
            if x in [0,1] or y in [0,1]: continue
            cov[i,j] = x*y*al['H_sem_5ps']**2+(1-x)*(1-y)*ti['H_sem_5ps']**2
            if i==j: cov[i,j] += ri['H_sem_5ps']**2
    covariances[model] = {'x_Al':[r['x_Al'] for r in g], 'covariance_kJ2_mol2':cov.tolist()}

write_csv('enthalpy_2000K.csv',rows)
write_csv('production_cutoff_sensitivity.csv',cutrows)
write_csv('stationarity_diagnostics.csv',tests)
write_csv('block_size_sensitivity.csv',block_rows)
write_csv('cache_inventory.csv',caches)
(OUT/'endpoint_covariance.json').write_text(json.dumps(covariances,indent=2))
(OUT/'sources.json').write_text(json.dumps(sources,indent=2))
np.savez_compressed(OUT/'production_timeseries.npz', time_ps=np.arange(301)*.1,
                    columns=np.array(['Epot_eV_atom','Etot_eV_atom','V_A3_atom','T_K',
                                      'P_GPa','rho_g_cm3','H_external_kJ_mol','H_legacy_kJ_mol']),
                    **{f'{m}_x{x:.1f}':a for (m,x),a in series.items()})

# Main curve: all archived frames, same estimator as the existing manuscript.
fig, ax = plt.subplots(figsize=(7.6,4.9),layout='constrained')
for model in MODELS:
    g=sorted([r for r in rows if r['model']==model],key=lambda r:r['x_Al'])
    c,label=STYLE[model]
    ax.errorbar([r['x_Al'] for r in g],[r['dH_mix_kJ_mol'] for r in g],
                yerr=[r['dH_sem_5ps'] for r in g],fmt='o-',ms=4,c=c,capsize=3,label=label)
ax.set(xlabel=r'Al atomic fraction, $x_{Al}$',ylabel=r'$\Delta H_{mix}$ (kJ mol$^{-1}$)',
       title='Ti–Al, 2000 K: archived 30 ps production',xlim=(0,1),ylim=(-35,1))
fig.text(.10,-.045,'Error bars: block SEM (6 × 5 ps), including pure-liquid references\nEquilibration samples were not saved; averaging has not been extended.',fontsize=8)
ax.legend(frameon=False)
ax.grid(alpha=.15)
save(fig,'mixing_enthalpy_2000K_recomputed')

fig,axs=plt.subplots(1,3,figsize=(12,3.8),sharey=True,layout='constrained')
for ax,model in zip(axs,MODELS):
    g=sorted([r for r in rows if r['model']==model],key=lambda r:r['x_Al'])
    x=[r['x_Al'] for r in g]
    ax.plot(x,[r['dH_legacy'] for r in g],'x:',c='.5',label='Original summary formula')
    ax.errorbar(x,[r['dH_mix_kJ_mol'] for r in g],yerr=[r['dH_sem_5ps'] for r in g],
                fmt='o-',c=STYLE[model][0],ms=3,capsize=2,label='External-pressure enthalpy')
    ax.plot(x,[r['dH_first15'] for r in g],'--',c='#9467BD',lw=1,label='First 15 ps')
    ax.plot(x,[r['dH_last15'] for r in g],'-.',c='#A66A23',lw=1,label='Last 15 ps')
    ax.set(title=STYLE[model][1],xlabel=r'$x_{Al}$',xlim=(0,1))
axs[0].set_ylabel(r'$\Delta H_{mix}$ (kJ mol$^{-1}$)')
axs[0].legend(fontsize=7,frameon=False,loc='lower left')
save(fig,'formula_and_sampling_comparison')

fig,axs=plt.subplots(1,3,figsize=(12,3.5),sharey=True,layout='constrained')
for ax,model in zip(axs,MODELS):
    for x in np.arange(1,10)/10:
        g=[r for r in cutrows if r['model']==model and r['x_Al']==x]
        ax.plot([r['discard_production_ps'] for r in g],
                [r['dH_mix_kJ_mol']-g[0]['dH_mix_kJ_mol'] for r in g],
                'o-',ms=2,label=f'{x:.1f}')
    ax.axhline(0,c='.5',lw=.7)
    ax.set(title=STYLE[model][1],xlabel='Discarded production (ps)',xticks=[0,1,2,5,10,15])
axs[0].set_ylabel(r'Change in $\Delta H_{mix}$ (kJ mol$^{-1}$)')
axs[-1].legend(title=r'$x_{Al}$',fontsize=7,ncol=3,frameon=False)
save(fig,'production_cutoff_sensitivity')

for model in MODELS:
    fig,axs=plt.subplots(11,3,figsize=(11,19),layout='constrained')
    for i,x in enumerate(np.arange(11)/10):
        a=series[model,float(x)]
        for j,(col,label) in enumerate([(6,'H − mean (kJ/mol)'),(3,'T (K)'),(5,'Density (g/cm³)')]):
            values=a[:,col].copy()
            if col==6: values-=values.mean()
            ax=axs[i,j]
            ax.plot(np.arange(301)*.1,values,c=STYLE[model][0],alpha=.35,lw=.6)
            ax.plot(np.arange(6)*5+2.55,block(values[1:],50),'o-',c='black',lw=1,ms=2.5)
            ax.axvline(15,c='.5',ls=':',lw=.7)
            ax.set_xlim(0,30)
            if i==0: ax.set_title(label)
            if j==0: ax.set_ylabel(f'x(Al) = {x:.1f}')
            if i==10: ax.set_xlabel('Production time (ps)')
    fig.suptitle(STYLE[model][1]+' at 2000 K: production only; black points = 5 ps averages')
    save(fig,f'production_diagnostics_{model}')

metrics={}
for model in MODELS:
    g=[r for r in rows if r['model']==model and 0<r['x_Al']<1]
    minimum=min(g,key=lambda r:r['dH_mix_kJ_mol'])
    metrics[model]={'minimum_kJ_mol':minimum['dH_mix_kJ_mol'], 'minimum_x_Al':minimum['x_Al'],
                    'minimum_SEM':minimum['dH_sem_5ps'],
                    'SEM_range':[min(r['dH_sem_5ps'] for r in g),max(r['dH_sem_5ps'] for r in g)],
                    'max_half_difference':max(abs(r['dH_half_difference']) for r in g),
                    'max_formula_correction':max(abs(r['dH_mix_kJ_mol']-r['dH_legacy']) for r in g)}
metrics['unadjusted_half_p_lt_005']=[t for t in tests if t['half_p']<.05]
metrics['BH_half_q_lt_005']=[t for t in tests if t['half_q_BH']<.05]
metrics['BH_slope_q_lt_005']=[t for t in tests if t['slope_q_BH']<.05]
(OUT/'metrics.json').write_text(json.dumps(metrics,indent=2))
print(json.dumps(metrics,indent=2),flush=True)
