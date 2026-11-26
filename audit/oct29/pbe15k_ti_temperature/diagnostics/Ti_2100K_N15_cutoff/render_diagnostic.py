"""Post-hoc diagnostic only: original production analysis and physics remain unchanged."""
from pathlib import Path
import os,sys,json,hashlib,time,itertools
HERE=Path(__file__).resolve().parent
WORK=HERE.parents[4]
sys.path.insert(0,str(WORK/'audit/oct20/deps'))
sys.path.insert(0,str(HERE/'source'))
os.environ['MPLCONFIGDIR']=str(HERE/'mplcache')
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import numpy as np
import attrs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stacie import compute_spectrum,estimate_acint,LorentzModel,CV2LCriterion
from stacie.estimate import fit_model_spectrum
CAMPAIGN=HERE.parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
out=read(HERE/'diagnostic_results.json')
windows={'40ps':None,'first20ps':None,'second20ps':None}
colors={'40ps':'#2878B5','first20ps':'#D55E00','second20ps':'#39835B'}
labels={'40ps':'40 ps','first20ps':'First 20 ps','second20ps':'Second 20 ps'}
fig,axs=plt.subplots(1,3,figsize=(13,4.1),layout='constrained')
for w in windows:
    hist=out['automatic'][w]['history'];x=np.array([r['fcut']/1e12 for r in hist]);y=np.array([r['acint']*1000 for r in hist]);e=np.array([r['acint_std']*1000 for r in hist]);c=colors[w]
    axs[0].plot(x,y,color=c,label=labels[w]);axs[0].fill_between(x,y-e,y+e,color=c,alpha=.12)
    axs[1].plot(x,[r['fcut_weight'] for r in hist],color=c,label=labels[w])
    axs[2].plot(x,[r['criterion_zscore'] for r in hist],color=c,label=labels[w])
    a=out['automatic'][w]['summary'];axs[0].errorbar(a['fcut']/1e12,a['acint']*1000,yerr=a['acint_std']*1000,fmt='o',color=c,capsize=3)
for ax in axs:
    ax.set_xscale('log');ax.set_xlim(.65,10.5);ax.set_xlabel('Cutoff frequency (THz)');ax.grid(alpha=.2)
axs[0].set(ylabel='Viscosity (mPa s)',title='Same data, different cutoff',ylim=(1.8,3.7));axs[0].legend(fontsize=8)
axs[1].set(ylabel='Automatic cutoff weight',title='Where automatic fitting puts weight')
axs[2].axhline(2,color='black',ls='--',label='Threshold 2');axs[2].set(ylabel='Cutoff criterion Z',title='Cutoff diagnostic',ylim=(-1.5,10));axs[2].legend(fontsize=8)
fig.suptitle('Pure Ti, PBE 15k, 2100 K: fifteen replicas | bands: +/-1 standard error')
fig.savefig(HERE/'cutoff_diagnostic.png',dpi=180);fig.savefig(HERE/'cutoff_diagnostic.pdf');plt.close(fig)
print('DONE',json.dumps(out['paired_jackknife']),flush=True)