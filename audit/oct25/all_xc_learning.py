"""Audit logged learning curves and actual shared training settings."""
from pathlib import Path
import sys,json,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'all_xc_audit';DEST=OUT/'analysis';DEST.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'audit/oct20/deps'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
cfg=json.loads((OUT/'protocol.json').read_text());configs=json.loads((OUT/'configurations.json').read_text())
results={};settings={}
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(3,3,figsize=(12.5,9),layout='constrained')
for j,xc in enumerate(['PBE','PBEsol','r2SCAN']):
    inp=configs[xc]['input'];tr=inp['training']
    settings[xc]={'learning_rate':inp['learning_rate'],'loss':inp['loss'],'optimizer':inp.get('optimizer'),
      'training':{k:v for k,v in tr.items() if k not in ['training_data','validation_data','stat_file']},
      'training_data':{k:v for k,v in tr.get('training_data',{}).items() if k!='systems'},
      'validation_data':{k:v for k,v in tr.get('validation_data',{}).items() if k!='systems'},
      'validating':inp.get('validating'),'model_fields':{k:v for k,v in inp['model'].items() if k not in ['type_map','descriptor','fitting_net']}}
    header=(OUT/(xc+'-lcurve.out')).read_text().splitlines()[0].lstrip('# ').split()
    data=np.loadtxt(OUT/(xc+'-lcurve.out'));col={k:data[:,i] for i,k in enumerate(header)}
    steps=col['step'];assert np.all(np.diff(steps)>0)
    steps_saved=[m['step'] for m in cfg['models'].values() if m['functional']==xc]
    result={'header':header,'rows':len(data),'first_step':int(steps[0]),'last_step':int(steps[-1]),'checkpoints':{}}
    for step in steps_saved:
        window=(steps>=max(1,step-1000))&(steps<=step);idx=int(np.argmin(abs(steps-step)))
        result['checkpoints'][str(step)]={'exact_log_step':int(steps[idx]),'window_steps':[int(steps[window][0]),int(steps[window][-1])],
           'exact':{k:float(col[k][idx]) for k in header[1:]},
           'trailing_1000_median':{k:float(np.nanmedian(col[k][window])) for k in header[1:]}}
    for i,prop in enumerate(['e','f','v']):
        scale=1000 if prop=='e' else 1
        for suffix,color,label in [('trn','#2675a8','Training log'),('val','#c75b2f','Validation log')]:
            a=scale*col[f'mae_{prop}_{suffix}'];smooth=np.array([np.median(a[max(0,n-10):n+1]) for n in range(len(a))])
            axs[i,j].plot(steps/1000,a,color=color,alpha=.15,lw=.6)
            axs[i,j].plot(steps/1000,smooth,color=color,lw=1.3,label=label)
        for s in steps_saved:axs[i,j].axvline(s/1000,color='gray',lw=.5,alpha=.35)
        label={'e':'Energy MAE (meV/atom)','f':'Force-norm MAE (eV/Å)','v':'Virial MAE (eV/atom)'}[prop]
        axs[i,j].set(xlabel='Optimization steps (thousands)',ylabel=label,title=xc if i==0 else None)
        axs[i,j].set_yscale('log');axs[i,j].grid(alpha=.15)
        if i==0:axs[i,j].legend(fontsize=8)
    results[xc]=result
fig.suptitle('Logged learning curves: faint raw values; trailing 11-point medians\nForce loss uses vector norms; these logs are not full-set component RMSE',fontsize=11)
fig.savefig(DEST/'learning_curves.png',dpi=190);fig.savefig(DEST/'learning_curves.pdf');plt.close(fig)
(DEST/'learning_curves.json').write_text(json.dumps(results,indent=2))
(DEST/'training_settings.json').write_text(json.dumps(settings,indent=2))
print(json.dumps({xc:{'steps':r['last_step'],'force_val_log_medians':{k:v['trailing_1000_median']['mae_f_val'] for k,v in r['checkpoints'].items()}} for xc,r in results.items()},indent=2))
