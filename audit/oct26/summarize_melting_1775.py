"""Compare the completed 100 ps 15k run with the archived 50 ps 5k run."""
from pathlib import Path
import sys,json
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
sys.path.insert(0,str(ROOT/'audit/melting'))
from summarize_melting import np,plt,slab_width,THRESHOLDS
OUT=BASE/'pbe15k_melting_1775/analysis'
cases={}
with np.load(OUT/'15k_1775.npz') as f:cases['15k']={k:f[k] for k in f.files}
with np.load(ROOT/'audit/melting/pbe_1775_001.npz') as f:
 cases['5k']={'time':f['step']*.001,'fraction':f['solid_fraction'],'width':np.array([slab_width(p) for p in f['profile']]),'profile':f['profile'],'bonds_fraction':(f['bonds']>=7).mean(axis=1),'cell':f['lengths'],'source_sha256':f['source_sha256']}
summaries={}
for key,a in cases.items():
 t=a['time'];f=a['fraction'][:,1];w=a['width'][:,0]
 s={'duration_ps':float(t[-1]),'initial_fraction':float(f[0]),'final_fraction':float(f[-1]),'mean_last10_fraction':float(f[t>=t[-1]-10].mean()),'source_sha256':str(a['source_sha256']), 'thresholds':{},'windows':{},'snapshots':{}}
 for tm in [0,10,25,50,75,100]:
  if tm<=t[-1]:
   i=int(np.argmin(abs(t-tm)));s['snapshots'][str(tm)]={'fraction':float(f[i]),'slab_width_Lz':float(w[i]),'bonds_fraction':float(a['bonds_fraction'][i]),'Lz_A':float(a['cell'][i,2])}
 for lo,hi in [(0,25),(5,25),(25,50),(50,75),(75,100),(50,100)]:
  if hi<=t[-1]:
   mask=(t>=lo)&(t<=hi);valid=mask&np.isfinite(w)
   s['windows'][f'{lo}-{hi}']={'mean_fraction':float(f[mask].mean()),'fraction_range':[float(f[mask].min()),float(f[mask].max())],'descriptive_fraction_slope_per_ps':float(np.polyfit(t[mask],f[mask],1)[0]),'descriptive_width_slope_per_ps':float(np.polyfit(t[valid],w[valid],1)[0]) if valid.sum()>3 else None}
 for i,th in enumerate(THRESHOLDS):s['thresholds'][str(th)]={'initial':float(a['fraction'][0,i]),'final':float(a['fraction'][-1,i]),'last10_mean':float(a['fraction'][t>=t[-1]-10,i].mean())}
 for label,cond in [('solid',f>.9),('liquid',f<.02)]:s['first_persistent_'+label+'_ps']=next((float(t[i]) for i in range(len(t)) if t[i]<=t[-1]-5 and np.all(cond[i:])),None)
 valid=np.flatnonzero(np.isfinite(w));s['last_resolved_interface_ps']=float(t[valid[-1]]) if len(valid) else None
 if 'end_qbar4' in a:s['end_qbar4_median']=float(np.median(a['end_qbar4']))
 summaries[key]=s
def clean(a):
 if isinstance(a,dict):return {k:clean(v) for k,v in a.items()}
 if isinstance(a,list):return [clean(v) for v in a]
 if isinstance(a,float) and not np.isfinite(a):return None
 return a
summaries=clean(summaries)
(OUT/'summary.json').write_text(json.dumps(summaries,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps(summaries,indent=2),flush=True)
fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for key,a in cases.items():
 for j,col in enumerate(['fraction','width']):
  val=a[col][:,1 if col=='fraction' else 0]
  for i,limit in enumerate([50,100]):axs[j,i].plot(a['time'],val,label=f'PBE {key}',lw=1.5)
for j in range(2):
 for i,limit in enumerate([50,100]):
  axs[j,i].set(xlim=(0,limit),ylim=(-.02,1.03),xlabel='Production time (ps)',ylabel=['Solid-like atom fraction','Solid slab width / Lz'][j]);axs[j,i].grid(alpha=.2);axs[j,i].legend()
axs[0,0].set_title('Matched observation window: 0–50 ps');axs[0,1].set_title('Full 15k trajectory: 0–100 ps')
fig.suptitle('Ti at 1775 K: identical qbar6 > 0.25 analysis; 1296 atoms')
for ext in ['png','pdf']:fig.savefig(OUT/f'comparison_1775.{ext}')
plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for ax,(key,a) in zip(axs,cases.items()):
 im=ax.pcolormesh(a['time'],(np.arange(36)+.5)/36,a['profile'].T,shading='nearest',vmin=.08,vmax=.4,cmap='viridis')
 for col in [1,2]:
  y=a['width'][:,col].copy();y[np.r_[False,abs(np.diff(y))>.5]]=np.nan;ax.plot(a['time'],y,color='white',lw=.9)
 ax.set(title=f'PBE {key}, 1775 K',xlabel='Production time (ps)',ylabel='z / Lz')
fig.colorbar(im,ax=axs,label='Layer mean qbar6')
for ext in ['png','pdf']:fig.savefig(OUT/f'interfaces_1775.{ext}')
plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for ax,(key,a) in zip(axs,cases.items()):
 for i,th in enumerate(THRESHOLDS):ax.plot(a['time'],a['fraction'][:,i],label=f'qbar6 > {th}')
 ax.plot(a['time'],a['bonds_fraction'],'k--',lw=1,label='>=7 coherent q6 bonds')
 ax.set(title=f'PBE {key}, 1775 K',xlabel='Production time (ps)',ylabel='Classified solid-like fraction',ylim=(-.02,1.03));ax.legend(fontsize=8);ax.grid(alpha=.2)
for ext in ['png','pdf']:fig.savefig(OUT/f'classifier_checks.{ext}')
plt.close(fig)
