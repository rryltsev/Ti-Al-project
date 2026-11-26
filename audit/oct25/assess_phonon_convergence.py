"""Assess convergence of the comparisons separately from absolute frequencies."""
from pathlib import Path
import json,sys,csv
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent/'pbe_checkpoint_phonons'
sys.path.insert(0,str(ROOT/'audit/oct20/deps'))
import numpy as np
keys=['5000','20000','23400'];a={k:np.load(OUT/k/'results.npz') for k in keys};b={k:np.load(OUT/('size9-'+k)/'results.npz') for k in keys}
def difference(d):return {'rms_THz':float(np.sqrt(np.mean(d*d))),'max_abs_THz':float(abs(d).max())}
out={'note':'9^3 is the largest computed cell; absolute-frequency convergence beyond it is not established. Relative checkpoint and geometry effects are assessed separately.',
     'checkpoint_difference_9_vs_7':{},'geometry_effect_9_vs_7':{},'experimental_nearest_branch_by_size':{}}
ex=np.array([[float(r['Exp_x']),float(r['Exp_y'])] for r in csv.DictReader((OUT/'reference-data.csv').open()) if r['Exp_x'] and r['Exp_y']])
x=a['5000']['qaxis'];x=x/x[-1]
for k in keys:
    out['geometry_effect_9_vs_7'][k]=difference((b[k]['relaxed-bands']-b[k]['common-bands'])-(a[k]['relaxed-large-cell-bands']-a[k]['common-large-cell-bands']))
    for g in ['common','relaxed']:
        if k!='5000':out['checkpoint_difference_9_vs_7'][k+'/'+g]=difference((b[k][g+'-bands']-b['5000'][g+'-bands'])-(a[k][g+'-large-cell-bands']-a['5000'][g+'-large-cell-bands']))
        for n,bs in [(5,a[k][g+'-base-bands']),(7,a[k][g+'-large-cell-bands']),(9,b[k][g+'-bands'])]:
            freq=np.column_stack([np.interp(ex[:,0],x,bs[:,i]) for i in range(6)])
            residual=np.min(abs(freq-ex[:,1,None]),axis=1)
            out['experimental_nearest_branch_by_size'][f'{k}/{g}/{n}']={'n':len(ex),'rms_THz':float(np.sqrt(np.mean(residual**2)))}
(OUT/'convergence-of-comparisons.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
