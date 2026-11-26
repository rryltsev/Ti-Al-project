from pathlib import Path
import json,sys,os
p=Path(__file__).resolve().parent;w=p.parents[2]
sys.path.insert(0,str(w/'audit/oct20/deps'))
os.environ['MPLCONFIGDIR']=str(p/'mplcache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
x=json.loads((p/'combined_results.json').read_text())
colors={'old20':'#9c540a','new_split20':'#2274a5','pooled20':'#22865d','new40':'#9656a6'}
labels={'old20':'Old 20 ps','new_split20':'New 40 ps split into 2 x 20 ps','pooled20':'Pooled 20 ps segments','new40':'New full 40 ps'}
fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for row,(T,e) in enumerate(x['temperatures'].items()):
 for k in colors:
  r=e['scan'][k];fc=np.array([a['fcut']/1e12 for a in r]);y=np.array([a['acint']*1000 for a in r]);se=np.array([a['acint_std']*1000 for a in r])
  axs[row,0].plot(fc,y,'o-',ms=3,color=colors[k],label=labels[k]);axs[row,0].fill_between(fc,y-se,y+se,color=colors[k],alpha=.10)
  axs[row,1].plot(fc,[a['criterion_zscore'] for a in r],'o-',ms=3,color=colors[k])
 for ax in axs[row]:
  ax.axvline(1.5,ls=':',color='gray');ax.set_xlabel('Common cutoff (THz)');ax.grid(alpha=.2);ax.set_xlim(.75,2.05)
 axs[row,0].set_title(f'{T} K: {e["old_n"]} old + 10 new trajectories');axs[row,0].set_ylabel('Viscosity (mPa s)')
 axs[row,0].set_ylim((2.85,3.45) if T=='1873' else (2.55,3.15))
 axs[row,1].set_ylabel('Cutoff diagnostic Z');axs[row,1].set_ylim(-2,10);axs[row,1].axhline(2,ls='--',color='black',label='Threshold 2')
axs[0,0].legend(fontsize=8);axs[0,1].legend(fontsize=8)
fig.suptitle('Separate pooling diagnostic | shaded bands: formal STACIE SE')
fig.savefig(p/'pooling_comparison_zoom.png',dpi=180);fig.savefig(p/'pooling_comparison_zoom.pdf');plt.close(fig)

