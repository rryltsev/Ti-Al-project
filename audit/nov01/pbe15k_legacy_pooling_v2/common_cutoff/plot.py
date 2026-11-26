from pathlib import Path
import os,sys,json
p=Path(__file__).resolve().parent;w=p.parents[3]
sys.path.insert(0,str(w/'audit/oct20/deps'));os.environ['MPLCONFIGDIR']=str(p/'mplcache')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
x=json.loads((p/'results.json').read_text());sc=json.loads((p/'scan.json').read_text());lo,hi=x['primary_shared_between_temperatures'][0]
cs={'old20':'#9c540a','new40':'#9656a6','new_split20':'#2274a5','pooled20':'#22865d'}
labels={'old20':'Old 20 ps','new40':'New full 40 ps','new_split20':'New 2 x 20 ps','pooled20':'Pooled 20 ps'}
fig,axs=plt.subplots(2,3,figsize=(14,7.2),layout='constrained')
for row,T in enumerate(['1873','2000']):
 for k in cs:
  rr=[r for r in sc[T][k] if .95e12<=r['fcut']<=1.55e12]
  f=np.array([r['fcut']/1e12 for r in rr])
  z=np.array([max(r['cost_zscore'],r['criterion_zscore']) for r in rr])
  axs[row,0].plot(f,z,color=cs[k],label=labels[k],lw=1.7)
  sub=[r for r in rr if (lo-.015)*1e12<=r['fcut']<=(hi+.015)*1e12]
  ff=np.array([r['fcut']/1e12 for r in sub])
  axs[row,1].plot(ff,[r['acint']*1000 for r in sub],color=cs[k],label=labels[k],lw=1.6)
  axs[row,2].plot(ff,[r['acint_std']*1000 for r in sub],color=cs[k],label=labels[k],lw=1.6)
 rr=[r for r in sc[T]['new_first20'] if .95e12<=r['fcut']<=1.55e12]
 axs[row,0].plot([r['fcut']/1e12 for r in rr],[max(r['cost_zscore'],r['criterion_zscore']) for r in rr],color='#bb3344',ls='--',label='New first 20 ps',lw=1.4)
 for col,ax in enumerate(axs[row]):
  ax.axvspan(lo,hi,color='#a3be8c',alpha=.22);ax.grid(alpha=.2);ax.set_xlabel('Cutoff (THz)')
  ax.set_xlim((.95,1.55) if col==0 else (lo-.008,hi+.008))
 axs[row,0].axhline(2,color='black',ls='--');axs[row,0].set_ylim(-2,4);axs[row,0].set_ylabel(f'{T} K\nmax(cost Z, cutoff Z)')
 axs[row,1].set_ylabel('Viscosity (mPa s)');axs[row,2].set_ylabel('Formal standard error (mPa s)')
axs[0,0].set_title('Fit diagnostics');axs[0,1].set_title('Within the shared interval');axs[0,2].set_title('Uncertainty sensitivity')
axs[0,0].legend(fontsize=8,loc='lower right')
fig.suptitle(f'Common interval: {lo:.4f} < cutoff < {hi:.4f} THz | both temperatures, four main datasets')
fig.savefig(p/'common_cutoff.png',dpi=180);fig.savefig(p/'common_cutoff.pdf');plt.close(fig)

