"""Validate the October 25 STACIE export, merge by state, and rebuild viscosity only.

Run with Python 3.12 and Matplotlib. No raw trajectories are reprocessed.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import shutil
import sys

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
OUT = HERE / 'viscosity'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE / 'viscosity_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SOURCE = Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project\results-scripts-logs\stacie_pipeline_25.10.2026\results')
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
key = lambda r: (r['T'], r['dft'], round(r['c_Ti'], 6))
old_path = WORK / 'audit/oct22/viscosity_merged.json'
old = {key(r): r for r in read(old_path)}
merged = dict(old)
incoming = []
for t, groups in read(SOURCE/'summary.json')['temperatures'].items():
    for model, group in groups.items():
        for r in group['points']:
            assert r['T'] == int(t) and r['dft'] == model
            incoming.append(r)
assert len(incoming) == len({key(r) for r in incoming}) == 20
csv_rows = {key(dict(r, T=int(r['T']), c_Ti=float(r['c_Ti']))): r
            for r in csv.DictReader((SOURCE/'summary.csv').open())}
assert set(csv_rows) == {key(r) for r in incoming}
sources = [{'path':str(p), 'sha256':sha(p)} for p in
           [SOURCE/'summary.json', SOURCE/'summary.csv', old_path,
            WORK/'audit/oct20/reference_values.json']]
changes = []
for r in incoming:
    k = key(r)
    detail = SOURCE/f"AlTi/{r['T']}/{r['dft']}/c_Ti_{r['c_Ti']:.2f}/result.json"
    assert read(detail) == r
    for f in ['eta_mPa_s','eta_std_mPa_s','eta_pa_s','eta_std_pa_s','n_steps','n_replicas']:
        assert math.isclose(float(csv_rows[k][f]), r[f], rel_tol=1e-12)
    assert math.isclose(r['eta_pa_s']*1000, r['eta_mPa_s'])
    assert math.isclose(r['eta_std_pa_s']*1000, r['eta_std_mPa_s'])
    assert r['eta_mPa_s'] > 0 and r['eta_std_mPa_s'] > 0
    assert r['n_steps'] == 20001 and r['timestep_ps'] == .001
    sources.append({'path': str(detail), 'sha256':sha(detail)})
    before = old[k]
    changed = any(before.get(f) != v for f,v in r.items())
    if changed:
        assert r['T'] == 1873 and r['n_replicas'] == 50
        changes.append(dict(T=r['T'],model=r['dft'],x_Ti=r['c_Ti'],x_Al=round(1-r['c_Ti'],6),
                            old=before['eta_mPa_s'],new=r['eta_mPa_s'],
                            old_std=before['eta_std_mPa_s'],new_std=r['eta_std_mPa_s'],
                            delta_percent=100*(r['eta_mPa_s']/before['eta_mPa_s']-1),
                            std_ratio=r['eta_std_mPa_s']/before['eta_std_mPa_s'],
                            old_n=before['n_replicas'],new_n=r['n_replicas']))
        merged[k] = dict(r, source=str(SOURCE/'summary.json'),result_file=str(detail),
                         updated_on='2026-10-25',previous_estimate=before,
                         tdm_comparison_status='not matched to updated trajectory set')
    # Snapshot each supplied point; unchanged states retain their earlier provenance.
    dest = OUT/'input'/detail.relative_to(SOURCE)
    dest.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(detail,dest)
for name in ['summary.json','summary.csv']:
    shutil.copy2(SOURCE/name,OUT/'input'/name)
rows = sorted(merged.values(),key=key)
assert len(rows) == 28 and len(changes) == 14
assert sum(r['T']==1873 for r in rows) == 22
for k in old:
    if k[0] != 1873:
        assert old[k] == merged[k]
(OUT/'viscosity_merged.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
refs = read(WORK/'audit/oct20/reference_values.json')
comparison = []
for c,(a,b,lo,hi) in refs['takeda_table3'].items():
    c = float(c)
    exp = 10**(a+b/1873)
    record = dict(x_Al=round(1-c,6),experiment=exp,extrapolated=1873>hi)
    for m in ['pbe','scan']:
        record[m] = merged[(1873,m,c)]['eta_mPa_s']
        record[m+'_relative_error_percent'] = 100*(record[m]/exp-1)
    record['closer'] = min(['pbe','scan'],key=lambda m:abs(record[m]-exp))
    comparison.append(record)
metrics = {}
for m in ['pbe','scan']:
    metrics[m] = {}
    for name, dataset in [('old',old),('new',merged)]:
        deviations = [dataset[(1873,m,round(1-r['x_Al'],6))]['eta_mPa_s']-r['experiment']
                      for r in comparison]
        metrics[m][name] = dict(mae=float(np.mean(np.abs(deviations))),
                               rmse=float(np.sqrt(np.mean(np.square(deviations)))),
                               mape=float(np.mean([abs(d)/r['experiment']*100 for d,r in zip(deviations,comparison)])))
retained = [dict(T=k[0],model=k[1],x_Ti=k[2],x_Al=round(1-k[2],6),
                 n_replicas=r['n_replicas']) for k,r in old.items()
            if k[0]==1873 and k not in {key(r) for r in incoming}]
report = dict(sources=sources,n_incoming=len(incoming),n_merged=len(rows),changes=changes,
              pending_updates=[],retained_1873=retained,alloy_experiment_comparison=comparison,
              experimental_metrics=metrics,
              limitations=['Estimator standard deviations are not total uncertainty.',
                'Old and new ensembles may overlap; no independent-sample significance test is made.',
                'No raw pressure trajectories, MD logs, or run seeds occur in the supplied results directory.',
                'Time duration remains 20 ps. Increasing replica count does not validate time or preparation convergence.',
                'Experimental fit uncertainties are unavailable; model ranking concerns central values.',
                'neff is a spectral diagnostic, not a number of independent MD trajectories.'])
(OUT/'viscosity_update_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
with (OUT/'changes.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(changes[0]));w.writeheader();w.writerows(changes)
with (OUT/'viscosity_1873K.csv').open('w',newline='',encoding='utf-8') as f:
    fields=['model','x_Al','x_Ti','eta_mPa_s','std_mPa_s','n_replicas','duration_ps','source']
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
    for r in rows:
        if r['T']==1873:
            w.writerow(dict(model=r['dft'],x_Al=round(1-r['c_Ti'],6),x_Ti=r['c_Ti'],
                            eta_mPa_s=r['eta_mPa_s'],std_mPa_s=r['eta_std_mPa_s'],
                            n_replicas=r['n_replicas'],duration_ps=20,source=r['source']))

plt.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':10,
    'legend.fontsize':7,'xtick.labelsize':8,'ytick.labelsize':8,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':300})
style={'pbe':('#2878B5','PBE','o'),'scan':('#D55E00',r'r$^2$SCAN','D')}
def series(ax,dataset,m,temperature=False,**kwargs):
    g=sorted((r for r in dataset.values() if (r['c_Ti']==1 if temperature else r['T']==1873)),
             key=lambda r:r['T'] if temperature else 1-r['c_Ti'])
    g=[r for r in g if r['dft']==m]
    ax.errorbar([r['T'] if temperature else 1-r['c_Ti'] for r in g],
                [r['eta_mPa_s'] for r in g],yerr=[r['eta_std_mPa_s'] for r in g],
                capsize=2,ms=3.5,lw=1,**kwargs)
def experiments(ax):
    for r in comparison:
        ax.scatter(r['x_Al'],r['experiment'],s=23,marker='s',
                   facecolors='white' if r['extrapolated'] else 'black',edgecolors='black',zorder=5)
    ax.plot([],[],'ks',ms=4,label='Takeda fit')
    ax.plot([],[],'ks',mfc='white',ms=4,label='Fit extrapolated by 3 K')
def save(fig,stem):
    for ext in ['pdf','png','svg']:
        fig.savefig(OUT/f'{stem}.{ext}',bbox_inches='tight')
    plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(7,2.9),layout='constrained')
for m,(col,label,marker) in style.items():
    series(axs[0],merged,m,fmt=marker+'-',c=col,label=label)
    series(axs[1],merged,m,temperature=True,fmt=marker+'-',c=col,label=label)
experiments(axs[0])
t=np.linspace(1850,2120,150);eta=.61*np.exp(27000/(8.31*t))
axs[1].plot(t,eta,'k--',lw=1,label='Ishikawa fit*')
axs[1].fill_between(t,eta*.897,eta*1.103,color='.7',alpha=.25)
axs[0].set(title='(a) Composition dependence, 1873 K',xlabel=r'$x_{\mathrm{Al}}$',
           ylabel=r'$\eta$ (mPa s)',xlim=(-.02,1.02),ylim=(0,8.3))
axs[1].set(title='(b) Pure Ti',xlabel='Temperature (K)',ylabel=r'$\eta$ (mPa s)',
           xlim=(1840,2130),ylim=(2.1,4.25))
axs[0].legend(frameon=False,loc='lower left');axs[1].legend(frameon=False,loc='upper right')
save(fig,'viscosity')

fig,axs=plt.subplots(1,2,figsize=(7,3.1),layout='constrained',sharey=True)
for ax,(m,(col,label,marker)) in zip(axs,style.items()):
    series(ax,old,m,fmt='o--',c='.6',mfc='white',label='22 Oct')
    series(ax,merged,m,fmt=marker+'-',c=col,label='25 Oct')
    experiments(ax)
    ax.set(title=label+' at 1873 K',xlabel=r'$x_{\mathrm{Al}}$',xlim=(-.02,1.02),ylim=(0,8.3))
    ax.legend(frameon=False,loc='lower left')
axs[0].set_ylabel(r'$\eta$ (mPa s)')
save(fig,'viscosity_old_vs_new')

# Preserve the previous manuscript assets once before replacing them.
backups=OUT/'previous';backups.mkdir(exist_ok=True)
for p in [WORK/'manuscript/figures/viscosity.pdf',WORK/'audit/results/viscosity.png']:
    if not (backups/p.name).exists(): shutil.copy2(p,backups/p.name)
shutil.copy2(OUT/'viscosity.pdf',WORK/'manuscript/figures/viscosity.pdf')
shutil.copy2(OUT/'viscosity.png',WORK/'audit/results/viscosity.png')
summary_path=WORK/'audit/results/numerical_summary.json'
summary=read(summary_path);summary['viscosity_current']=rows;summary['viscosity_update_pending']=[]
summary['viscosity_oct25_comparison']=report['experimental_metrics']
summary_path.write_text(json.dumps(summary,indent=2),encoding='utf-8')
source_path=WORK/'audit/results/figure_sources.json'
source_index=read(source_path)
source_index.update({r['path']:r['sha256'] for r in sources})
source_index[str(OUT/'viscosity_merged.json')]=sha(OUT/'viscosity_merged.json')
source_path.write_text(json.dumps(source_index,indent=2),encoding='utf-8')
print(json.dumps(dict(n_changed=len(changes),retained=retained,comparison=comparison,metrics=metrics,
                     std_ratio_median=float(np.median([r['std_ratio'] for r in changes]))),indent=2))
