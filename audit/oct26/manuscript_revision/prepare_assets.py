"""Freeze the 26 October manuscript inputs and build only the updated figures."""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
MAN = ROOT / 'manuscript'
FIG = MAN / 'figures'
SI = FIG / 'supporting'
DATA = MAN / 'supporting_data'
PROJECT = Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project')
sources = {}

def track(path):
    p = Path(path)
    sources[str(p.resolve())] = hashlib.sha256(p.read_bytes()).hexdigest()
    return p

def read(path):
    return json.loads(track(path).read_text(encoding='utf-8-sig'))

def copy(src, dest):
    shutil.copy2(track(src), dest)

def save(fig, dest):
    fig.savefig(dest.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(OUT / (dest.stem + '.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)

plt.rcParams.update({'font.size':9, 'axes.labelsize':9, 'axes.titlesize':10,
    'legend.fontsize':7, 'axes.spines.top':False, 'axes.spines.right':False,
    'pdf.fonttype':42})
metrics_dir = ROOT / 'testset/monitor/results/n23-1a6fdc00d4d0b30a'
m = read(metrics_dir / 'metrics.json')
assert len(m['common_ids']) == 23 and m['models']['PBE']['checkpoint_step'] == 15000
copy(metrics_dir / 'parity.pdf', FIG / 'test23_parity.pdf')
copy(metrics_dir / 'metrics.json', DATA / 'test23_metrics.json')
copy(ROOT / 'testset/monitor/runs/pbe15k-switch-20261026-01/comparison-5k-vs15k.json', DATA / 'test23_5k_vs_15k.json')
rows = []
for xc, label in [('PBE','PBE'), ('PBEsol','PBEsol'), ('r2SCAN',r'\rscan{}')]:
    for kind, model in [('metrics','Fine-tuned'), ('pretrained_metrics','Pretrained')]:
        if xc == 'PBE' and kind == 'metrics': model += ' (15k)'
        a = m[kind][xc]['all']
        v = [a[q][p] for q in ['energy_meV_atom','force_eV_A','stress_GPa'] for p in ['mae','rmse']]
        rows.append(label+' & '+model+' & '+' & '.join(f'{x:.{n}f}' for x,n in zip(v,[2,2,4,4,3,3]))+r' \\')
table = r'''\begin{table*}[t]
\caption{Preliminary errors on the same 23 held-out configurations (20 liquid-pool and three solid-solution-pool structures). Energy errors use raw energies per atom; forces pool Cartesian components and stresses pool six independent tensor entries. The pretrained rows compare one unchanged model with each functional. No energy offsets are fitted.}
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
(MAN/'sections/test23_metrics_table.tex').write_text(table,encoding='utf-8')

# PBE15k hcp bands at its equilibrium cell. Other functional curves are preserved.
ph = np.load(track(ROOT/'audit/oct25/pbe_phonons_15k_equilibrium/15000/results.npz'))
ph_meta = read(ROOT/'audit/oct25/pbe_phonons_15k_equilibrium/15000/summary.json')
assert ph_meta['model_sha256'] == m['models']['PBE']['checkpoint_sha256']
dos = np.load(track(ROOT/'audit/oct25/all_xc_audit/analysis/dos_curves.npz'))
ph_metrics = read(ROOT/'audit/oct25/all_xc_audit/analysis/phonons.json')
old_summary = read(ROOT/'audit/results/numerical_summary.json')
phroot = PROJECT/'results-scripts-logs/models_check_wsl2/results/phonons/dpa4/neo/finetune/alpha_Ti'
expfile = PROJECT/'MD-results/phonons_DPA4-neo/finetune/alpha_Ti/pbe/phonon_bands_data_alpha_Ti_dpa4-neo-finetune_pbe.csv'
with track(expfile).open() as f:
    ex = np.array([(float(r['Exp_x']),float(r['Exp_y'])) for r in csv.DictReader(f) if r['Exp_x'] and r['Exp_y']])
style = {'pbe':('#2878B5','PBE 15k'), 'sol':('#39835B','PBEsol'), 'scan':('#D55E00',r'r$^2$SCAN')}
fig,axs = plt.subplots(2,2,figsize=(7,5.5),layout='constrained')
for j,model in enumerate(style):
    col,label = style[model]
    if model == 'pbe':
        q=ph['qaxis']/ph['qaxis'][-1]; ticks=ph['ticks']/ph['qaxis'][-1]; bands=ph['bands_THz']
        freq=dos['frequency_THz']; y=dos['PBE-15000/equilibrium/MP50']
    else:
        a=np.loadtxt(track(phroot/model/'phonon_bands.dat'))
        q=a[:,0]/a[-1,0]; bands=a[:,1:]; ticks=old_summary['phonons']['alpha_Ti/'+model]['q_special_normalized']
        d=np.loadtxt(track(phroot/model/'phonon_dos.dat')); freq=d[:,0]; y=d[:,1]*3/np.trapezoid(d[:,1],freq)
    ax=axs.flat[j]; ax.plot(q,bands,color=col,lw=1)
    ax.scatter(ex[:,0],ex[:,1],s=10,c='black',alpha=.8,label='Experiment (digitized)',zorder=3)
    for t in ticks: ax.axvline(t,color='.8',lw=.5,zorder=0)
    ax.set(xticks=ticks,xticklabels=[r'$\Gamma$','K','M',r'$\Gamma$','A'],ylabel='Frequency (THz)',
           xlim=(0,1),ylim=(-.2,10),title=f'({chr(97+j)}) {label}: '+r'$\alpha$-Ti')
    if j==0: ax.legend(frameon=False,loc='upper right')
    axs[1,1].plot(freq,y,c=col,lw=1.1,label=label)
dft=np.loadtxt(track(ROOT/'audit/oct25/pbe_phonons_dft_geometry/DFT-total_dos.dat'))
axs[1,1].plot(dft[:,0],dft[:,1]*3/np.trapezoid(dft[:,1],dft[:,0]),'k--',lw=1.1,label='PBE DFT reference')
axs[1,1].set(xlabel='Frequency (THz)',ylabel=r'DOS (atom$^{-1}$ THz$^{-1}$)',title=r'(d) $\alpha$-Ti density of states',xlim=(0,11))
axs[1,1].legend(frameon=False)
save(fig,FIG/'phonons_alpha_ti')

# Learning curves: change the selected marker, preserve the logged quantities.
a=np.loadtxt(track(DATA/'lcurve_pbe.out'))
fig,axs=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
for ax,cols,factor,title,ylabel in [(axs[0],(2,1),1000,'(a) Energy learning curves','Logged energy MAE (meV/atom)'),
        (axs[1],(4,3),1,'(b) Force learning curves','Logged force MAE (eV/Å)')]:
    for col,color,label in zip(cols,['#246db5','#d17a13'],['Training','Validation']):
        y=a[:,col]*factor; ax.plot(a[:,0]/1000,y,c=color,lw=.6,alpha=.22)
        ax.plot(a[5:-5,0]/1000,np.convolve(y,np.ones(11)/11,mode='valid'),c=color,lw=1.7,label=label)
    ax.axvline(15,c='.3',ls='--',lw=1,label='Selected 15k')
    ax.set(xlabel='Optimization steps (thousands)',ylabel=ylabel,title=title,xlim=(0,24));ax.grid(alpha=.15)
axs[0].set_ylim(0,12);axs[1].set_ylim(.08,.135);axs[0].legend(frameon=False)
save(fig,SI/'learning_curves')

# Full five-checkpoint DOS comparison at fixed and independently optimized cells.
colors=['#888888','#b58a00','#2878B5','#ae5479','#39835B']
steps=[5000,10000,15000,20000,23400]
fig,axs=plt.subplots(1,2,figsize=(10,3.5),sharey=True,layout='constrained')
for ax,geom,title in zip(axs,['fixed','equilibrium'],['(a) Identical DFT geometry','(b) Each equilibrium geometry']):
    ax.plot(dos['frequency_THz'],dos['DFT'],c='black',lw=1.6,label='PBE DFT')
    for step,col in zip(steps,colors):
        ax.plot(dos['frequency_THz'],dos[f'PBE-{step}/{geom}/MP40'],c=col,lw=1.6 if step==15000 else .9,
                label=f'{step/1000:g}k'+(' (selected)' if step==15000 else ''))
    ax.set(xlabel='Frequency (THz)',title=title,xlim=(0,10.5));ax.legend(frameon=False,ncol=2)
axs[0].set_ylabel(r'DOS (atom$^{-1}$ THz$^{-1}$)')
save(fig,SI/'dos_five_checkpoints')

copy(ROOT/'audit/oct25/pbe_phonons_five_checkpoints/checkpoint_tradeoff.png',SI/'checkpoint_tradeoff.png')
for src,name in [
    ('audit/oct25/pbe_phonons_five_checkpoints/analysis.json','five_checkpoint_dos.json'),
    ('audit/oct25/all_xc_audit/analysis/phonons.json','all_checkpoint_phonons.json'),
    ('audit/oct25/pbe_phonons_15k_equilibrium/15000/summary.json','pbe15k_equilibrium_phonons.json'),
    ('audit/oct25/pbe_full_enthalpy_gpu/analysis/statistics.json','pbe_checkpoint_enthalpy_statistics.json')]:
    copy(ROOT/src,DATA/name)

thermo=read(ROOT/'audit/oct20/thermo_reprocessed.json')
refs=read(ROOT/'audit/oct20/reference_values.json')
stat=read(ROOT/'audit/oct25/pbe_full_enthalpy_gpu/analysis/statistics.json')
new=sorted(stat['windows']['production']['mixing']['15000'].values(),key=lambda x:x['x_Al'])
mixref=refs['mixing_digitized_kJmol']; xp=np.array([r['x_Al'] for r in new]); yp=np.array([r['H']['mean'] for r in new])
mad=np.mean([abs(np.interp(float(x),xp,yp)-y) for x,y in mixref.items()])
import runpy
runpy.run_path(str(ROOT/'audit/oct26/remove_enthalpy_temperature/update_figure.py'),run_name='__main__')
for name in ['density_enthalpy_5k_15k_20k','equilibration_window_sensitivity','checkpoint_differences_95CI']:
    copy(ROOT/f'audit/oct25/pbe_full_enthalpy_gpu/analysis/{name}.pdf',SI/(name+'.pdf'))

rows=[]
for r in new:
    rows.append(f"{r['x_Al']:.1f} & {r['H']['mean']:.3f} & {r['H']['sem']:.3f}"+r' \\')
(DATA/'pbe15k_enthalpy_rows.tex').write_text('\n'.join(rows)+'\n',encoding='utf-8')
summary={'common_structures':23,'pbe15k_phonons':{'gamma_THz':ph['bands_THz'][0].tolist(),'maximum_THz':float(ph['bands_THz'].max())},
    'pbe15k_mixing_MAD_kJmol':float(mad),'pbe15k_enthalpy_minimum':min(new,key=lambda r:r['H']['mean']),
    'model_identity':m['models'],'frozen_test_directory':str(metrics_dir),'source_hashes':sources}
(DATA/'revision_oct26_sources.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
(OUT/'numerical_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ['source_hashes','model_identity']},indent=2))
