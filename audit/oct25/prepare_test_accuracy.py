"""Pin, verify and typeset the current 20-configuration test results for the article.

No DFT/inference jobs and no energy alignment. Monitor inputs are read-only.
Update ARCHIVE deliberately when adopting a larger common benchmark subset.
"""
from pathlib import Path
import hashlib, json, shutil, sys, os
HERE=Path(__file__).resolve().parent
WORK=HERE.parents[1]
MON=WORK/'testset/monitor'
ARCHIVE=MON/'results/n20-47a8b37210fc502b'
sys.path[:0]=[str(WORK/'audit/oct20/deps'),str(MON)]
os.environ['MPLBACKEND']='Agg'
import numpy as np
import matplotlib.pyplot as plt
import analyze

sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
report=read(ARCHIVE/'metrics.json')
assert report['analysis_mode']=='monitor-raw' and report['alignment'] is None
ids=report['common_ids']
assert len(ids)==20 and not report['pending_predictions']
snapshot=HERE/'test20_bundle.json'
if snapshot.exists():
    bundle=read(snapshot)
else:
    current=read(MON/'data/latest.json')
    byid={r['metadata']['structure_id']:r for r in current['records']}
    bundle={'common_ids':ids,'records':[byid[sid] for sid in ids],
            'manifest_sha256':current['manifest_sha256']}
    snapshot.write_text(json.dumps(bundle,indent=2),encoding='utf-8')
assert bundle['common_ids']==ids and bundle['manifest_sha256']==report['manifest_sha256']
cache=read(MON/'prediction-manifest.json')
registry=read(MON/'model_registry.json')
pretrained=read(MON/'pretrained_registry.json')
predictions={}; pending=[]; provenance={str(ARCHIVE/'metrics.json'):sha(ARCHIVE/'metrics.json')}
for i,r in enumerate(bundle['records']):
    assert r['metadata']['family']=='liquid' and len(r['symbols'])==216
    assert [r['labels'][xc]['label_sha256'] for xc in analyze.XC]==report['input_identity']['labels'][i]
    p=WORK/'testset/TiAl-test50-v1'/r['metadata']['poscar']
    assert sha(p)==r['metadata']['poscar_sha256']
    for xc in analyze.XC:
        label=r['labels'][xc]
        assert label['scf_converged'] and label['ionic_calculations']==1
for xc in analyze.XC:
    assert registry['models'][xc]['sha256']==report['models'][xc]['sha256']
    predictions[xc]=analyze.load_predictions(bundle,xc,registry['models'][xc]['sha256'],cache,pending)
assert pretrained['sha256']==report['pretrained_model']['sha256']
baseline=analyze.load_predictions(bundle,'pretrained',pretrained['sha256'],cache,pending)
assert not pending
values={}; stats={}
for role,preds,reportkey in [('Fine-tuned',predictions,'metrics'),('Pretrained',{xc:baseline for xc in analyze.XC},'pretrained_metrics')]:
    values[role]={};stats[role]={}
    for xc in analyze.XC:
        vals,groups=analyze.compare(bundle['records'],preds,xc)
        values[role][xc]=vals;stats[role][xc]=groups['all']
        for prop in ['energy_meV_atom','force_eV_A','stress_GPa','force_Al_eV_A','force_Ti_eV_A']:
            for metric in ['mae','rmse','bias','mae_ci95','rmse_ci95']:
                assert np.allclose(groups['all'][prop][metric],report[reportkey][xc]['all'][prop][metric],rtol=1e-9,atol=1e-10),(role,xc,prop,metric)
for role in list(analyze.XC)+['pretrained']:
    for sid in ids:
        p=MON/'predictions'/role/(sid+'.json');provenance[str(p)]=sha(p)
for p in [MON/'model_registry.json',MON/'pretrained_registry.json',MON/'prediction-manifest.json',MON/'analyze.py',snapshot]:
    provenance[str(p)]=sha(p)
shutil.copy2(ARCHIVE/'metrics.json',HERE/'test20_metrics.json')

# The same six parity panels, reformatted for readable journal column widths.
plt.rcParams.update({'font.size':8,'axes.labelsize':7,'axes.titlesize':9,
                     'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,3,figsize=(7.05,5.2),layout='constrained')
colors={'Fine-tuned':'#276FBF','Pretrained':'#D97917'}
for j,xc in enumerate(analyze.XC):
    for role in ['Pretrained','Fine-tuned']:
        vv=values[role][xc]; col=colors[role]
        axs[0,j].scatter([v['e_ref'] for v in vv],[v['e_pred'] for v in vv],s=23 if role=='Pretrained' else 11,
                         facecolors='none' if role=='Pretrained' else col,edgecolors=col,lw=.7,zorder=3)
        axs[1,j].scatter(np.concatenate([v['f_ref'].ravel() for v in vv]),np.concatenate([v['f_pred'].ravel() for v in vv]),
                         s=1.5,alpha=.23,color=col,rasterized=True)
    for i,prop in enumerate(['energy_meV_atom','force_eV_A']):
        ax=axs[i,j];lo=min(*ax.get_xlim(),*ax.get_ylim());hi=max(*ax.get_xlim(),*ax.get_ylim())
        ax.plot([lo,hi],[lo,hi],'--',color='.5',lw=.6,zorder=0)
        ax.set(xlim=(lo,hi),ylim=(lo,hi),aspect='equal',title=f'({chr(97+3*i+j)}) '+(r'r$^2$SCAN' if xc=='r2SCAN' else xc))
        for k,role in enumerate(['Fine-tuned','Pretrained']):
            m=stats[role][xc][prop];unit='meV/atom' if i==0 else r'eV/$\AA$'
            right=(i==0 and j==2)
            ax.text(.97 if right else .035,(.36 if right else .965)-k*.18,
                    f"{role}: RMSE {m['rmse']:.3g}\nMAE {m['mae']:.3g} {unit}",transform=ax.transAxes,
                    ha='right' if right else 'left',va='top',fontsize=6.3,color=colors[role],
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=.7))
    axs[0,j].set(xlabel='DFT energy (eV/atom)',ylabel='Model energy (eV/atom)')
    axs[1,j].set(xlabel=r'DFT force component (eV/$\AA$)',ylabel=r'Model force component (eV/$\AA$)')
fig.suptitle('Preliminary: 20 liquid-pool configurations; raw energies',fontsize=9)
fig.savefig(WORK/'manuscript/figures/test20_parity.pdf',bbox_inches='tight')
fig.savefig(HERE/'test20_parity.png',dpi=220,bbox_inches='tight')
plt.close(fig)

table=[r'\begin{table*}[t]',
r'\caption{Preliminary errors on the same 20 liquid-pool configurations. Energy errors use raw energies per atom, force errors pool Cartesian components, and stress errors pool the six independent tensor entries. The pretrained rows use one unchanged model compared with each functional; their absolute energy errors include differences in energy references. No energy shifts were fitted.}',
r'\label{tab:fit}',r'\begin{ruledtabular}',r'\begin{tabular}{llrrrrrr}',
r' & & \multicolumn{2}{c}{Energy (meV/atom)} & \multicolumn{2}{c}{Force (eV/\AA)} & \multicolumn{2}{c}{Stress (GPa)} \\',
r'Reference & Model & MAE & RMSE & MAE & RMSE & MAE & RMSE \\',r'\hline']
for xc in analyze.XC:
    for role in ['Fine-tuned','Pretrained']:
        m=stats[role][xc];ee=m['energy_meV_atom'];ff=m['force_eV_A'];ss=m['stress_GPa']
        name=r'\rscan{}' if xc=='r2SCAN' else xc
        table.append(f"{name} & {role} & {ee['mae']:.2f} & {ee['rmse']:.2f} & {ff['mae']:.4f} & {ff['rmse']:.4f} & {ss['mae']:.3f} & {ss['rmse']:.3f} "+r'\\')
table += [r'\end{tabular}',r'\end{ruledtabular}',r'\end{table*}']
(WORK/'manuscript/sections/test20_metrics_table.tex').write_text('\n'.join(table)+'\n',encoding='utf-8')
(HERE/'source_manifest.json').write_text(json.dumps({'fingerprint':report['fingerprint'],'ids':ids,'sha256':provenance,'figure_note':'Same data and metrics as archived parity plot; journal typography only.'},indent=2),encoding='utf-8')
print('Verified 80 cached predictions, 60 labels, all E/F/stress metrics and bootstrap intervals; wrote parity figure and table.')
