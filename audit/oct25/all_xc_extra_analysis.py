"""Secondary analysis on additional ready tests; never mix unequal XC coverage silently."""
from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];BASE=HERE/'all_xc_audit';OUT=BASE/'extra_tests';DEST=BASE/'analysis'
sys.path.insert(0,str(ROOT/'audit/oct20/deps'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
read=lambda p:json.loads(Path(p).read_text())
cfg=read(BASE/'protocol.json');tasks=read(OUT/'tasks.json');common=read(DEST/'per_configuration_metrics.json')
results={};records={};missing=[]
def metric(rr):
    if not rr:return None
    e=np.array([r['error_E_meV_atom'] for r in rr]);d={'n':len(rr),'E_RMSE_meV_atom':float(np.sqrt(np.mean(e*e))),
      'E_MAE_meV_atom':float(np.mean(abs(e))),'E_bias_meV_atom':float(e.mean())}
    for p,u in [('F','eV_A'),('S','GPa')]:d[p+'_RMSE_'+u]=float(np.sqrt(sum(r[p+'_sse'] for r in rr)/sum(r[p+'_n'] for r in rr)))
    return d
for task in tasks:
    key=task['key'];xc=cfg['models'][key]['functional'];p=OUT/'results'/str(task['id'])
    if not (p/'summary.json').exists():missing.append(key);continue
    assert read(p/'summary.json')['model_sha256']==cfg['models'][key]['sha256']
    rows=read(OUT/(xc+'-rows.json'));pred=np.load(p/'predictions.npz');ref=np.load(OUT/(xc+'-labels.npz'));rr=[]
    for r in rows:
        i=r['i'];n=r['n_atoms'];V=n*r['volume_A3_atom']
        e=1000*float(pred[f'energy_{i}']-ref[f'energy_{i}'])/n
        f=pred[f'forces_{i}']-ref[f'forces_{i}'];s=-(pred[f'virial_{i}']-ref[f'virial_{i}'])/V*160.21766208
        s=(s+s.T)/2;sv=s.flat[[0,4,8,5,2,1]]
        rr.append({**r,'error_E_meV_atom':e,'F_sse':float(np.sum(f*f)),'F_sae':float(np.sum(abs(f))),'F_n':f.size,
                   'S_sse':float(np.sum(sv*sv)),'S_sae':float(np.sum(abs(sv))),'S_n':6})
    co=[r for r in common[key] if r['split']=='test'];full=co+rr;records[key]=rr
    results[key]={'all_available':metric(full),'all_available_liquids':metric([r for r in full if r['family']=='liquid']),
       'extra_crystals':metric([r for r in rr if r['family']!='liquid']),
       'by_family':{f:metric([r for r in rr if r['family']==f]) for f in sorted(set(r['family'] for r in rr))}}
(DEST/'extra_test_metrics.json').write_text(json.dumps({'groups':results,'records':records,'missing':missing,
 'limitation':'Unequal functional coverage: these are secondary diagnostics, not directly comparable cross-functional benchmark averages.'},indent=2))
fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained');colors=['#2874a6','#20916f','#8e54ad','#d38c20','#bd4545']
for j,xc in enumerate(['PBE','PBEsol']):
    keys=[k for k,m in cfg['models'].items() if m['functional']==xc]
    for color,key in zip(colors,keys):
        if key not in records:continue
        rr=[r for r in records[key] if r['family']=='POSCAR_alpha-Ti-hcp_orthogonal'];rr=sorted(rr,key=lambda r:r['volume_A3_atom'])
        axs[0,j].plot([r['volume_A3_atom'] for r in rr],[r['error_E_meV_atom'] for r in rr],'o-',color=color,label=f"{cfg['models'][key]['step']/1000:g}k")
        axs[1,j].plot([r['volume_A3_atom'] for r in rr],[np.sqrt(r['S_sse']/6) for r in rr],'o-',color=color)
    axs[0,j].set(title=xc+' / extra hcp Ti tests',xlabel='Volume (Å³/atom)',ylabel='Energy residual (meV/atom)');axs[0,j].legend(fontsize=8)
    axs[0,j].axhline(0,color='gray',lw=.6)
    axs[1,j].set(xlabel='Volume (Å³/atom)',ylabel='Stress-component RMSE (GPa)')
    for ax in axs[:,j]:ax.grid(alpha=.2)
fig.suptitle('Additional validated crystals; r²SCAN crystal references are not yet available',fontsize=11)
fig.savefig(DEST/'extra_hcp_test_residuals.png',dpi=190);fig.savefig(DEST/'extra_hcp_test_residuals.pdf');plt.close(fig)
print(json.dumps({'missing':missing,'groups':results},indent=2))
