"""Matched 0.5 ps sampling and unchanged structural classifier for 5k/15k."""
from pathlib import Path
import sys,json,hashlib,re
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
sys.path.insert(0,str(ROOT/'audit/melting'))
from summarize_melting import read_dump,order,slab_width,THRESHOLDS,np,plt,ideal_check,cKDTree,sph_harm_y
RUN=BASE/'pbe15k_melting/attempt2';SRC=RUN/'collected';OUT=RUN/'analysis';OUT.mkdir(exist_ok=True)
assert json.loads((SRC/'collection.json').read_text())['status']=='verified';ideal_check()
def qbar4(xyz,cell):
 idx=cKDTree(xyz,boxsize=cell).query(xyz,k=15)[1][:,1:];dr=xyz[idx]-xyz[:,None,:];dr-=np.rint(dr/cell)*cell;r=np.linalg.norm(dr,axis=2)
 theta=np.arccos(np.clip(dr[:,:,2]/r,-1,1));phi=np.arctan2(dr[:,:,1],dr[:,:,0]);qlm=np.stack([sph_harm_y(4,m,theta,phi).mean(axis=1) for m in range(-4,5)],axis=1)
 qb=(qlm+qlm[idx].sum(axis=1))/15;return np.sqrt(4*np.pi/9*np.sum(abs(qb)**2,axis=1))
def thermo(path):
 sections=[];active=None
 for line in path.read_text().splitlines():
  cols=line.split()
  if cols and cols[0]=='Step' and 'Temp' in cols:active={'columns':cols,'rows':[]};sections.append(active)
  elif active is not None:
   if line.startswith('Loop time'):active=None
   elif len(cols)==len(active['columns']):
    try:active['rows'].append([float(x) for x in cols])
    except ValueError:pass
 a=sections[-1];data=np.array(a['rows']);headers=a['columns'];assert data[-1,headers.index('Step')]==50000
 keep=data[:,headers.index('Step')]>=5000
 return {'mean_T_5_50_K':float(data[keep,headers.index('Temp')].mean()),'mean_P_5_50_GPa':float(data[keep,headers.index('Press')].mean()/10000),'thermo_rows':len(data)}
cases={};summaries={}
for T in [1700,1750,1800,1850]:
 path=SRC/f'T{T}K/melt/trajectory.lammpstrj';cache=OUT/f'15k_{T}.npz'
 if not cache.exists():
  frames=read_dump(path);assert len(frames)==501 and frames[-1][0]==50000
  frames=frames[::5];data={k:[] for k in ['time','fraction','width','profile','bonds_fraction','cell','qbar6']}
  for step,cell,xyz,pe in frames:
   q,qb,bonds=order(xyz,cell);bins=np.minimum((xyz[:,2]/cell[2]*36).astype(int),35);profile=np.bincount(bins,weights=qb,minlength=36)/np.maximum(np.bincount(bins,minlength=36),1)
   vals=[step*.001,(qb[:,None]>THRESHOLDS).mean(axis=0),slab_width(profile),profile,(bonds>=7).mean(),cell,qb]
   for k,v in zip(data,vals):data[k].append(v)
  data={k:np.array(v) for k,v in data.items()};data['end_qbar4']=qbar4(frames[-1][2],frames[-1][1]);data['source_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();np.savez_compressed(cache,**data)
 with np.load(cache) as f:cases[f'15k_{T}']={k:f[k] for k in f.files}
for T in [1700,1725,1750,1775,1800]:
 with np.load(ROOT/f'audit/melting/pbe_{T}_001.npz') as f:
  cases[f'5k_{T}']={'time':f['step']*.001,'fraction':f['solid_fraction'],'width':np.array([slab_width(p) for p in f['profile']]),'profile':f['profile'],'bonds_fraction':(f['bonds']>=7).mean(axis=1),'cell':f['lengths'],'source_sha256':f['source_sha256']}
for key,a in cases.items():
 t=a['time'];f=a['fraction'][:,1];s={'initial_fraction':float(f[0]),'final_fraction':float(f[-1]),'mean_last10_fraction':float(f[t>=40].mean()),'range_last10_fraction':[float(f[t>=40].min()),float(f[t>=40].max())],
 'initial_width_fraction':float(a['width'][0,0]),'final_width_fraction':float(a['width'][-1,0]),'threshold_checks':{},'source_sha256':str(a['source_sha256'])}
 s['descriptive_slopes_per_ps']={f'{lo}-{hi}':float(np.polyfit(t[(t>=lo)&(t<=hi)],f[(t>=lo)&(t<=hi)],1)[0]) for lo,hi in [(5,25),(10,50),(25,50)]}
 for i,th in enumerate(THRESHOLDS):s['threshold_checks'][str(th)]={'initial_final':a['fraction'][[0,-1],i].tolist(),'mean_last10':float(a['fraction'][t>=40,i].mean())}
 s['bond_fraction_initial_final']=a['bonds_fraction'][[0,-1]].tolist()
 for label,condition in [('solid',f>.90),('liquid',f<.02)]:s['first_persistent_'+label+'_ps']=next((float(t[i]) for i in range(len(t)) if t[i]<=45 and np.all(condition[i:])),None)
 if key.startswith('15k'):
  s.update(thermo(SRC/f'T{key.split("_")[1]}K/melt/log.lammps'));s['median_end_qbar4']=float(np.median(a['end_qbar4']))
 summaries[key]=s;print(key,json.dumps(s),flush=True)
def clean(a):
 if isinstance(a,dict):return {k:clean(v) for k,v in a.items()}
 if isinstance(a,list):return [clean(v) for v in a]
 if isinstance(a,float) and not np.isfinite(a):return None
 return a
(OUT/'summary.json').write_text(json.dumps(clean(summaries),indent=2,allow_nan=False))
fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
colors={1700:'#277da8',1725:'#679d69',1750:'#26a269',1775:'#b79028',1800:'#e28b21',1850:'#c64850'}
for ax,cp in zip(axs,['5k','15k']):
 for key,a in cases.items():
  if key.split('_')[0]==cp:
   T=int(key.split('_')[1]);ax.plot(a['time'],a['fraction'][:,1],label=f'{T} K',color=colors[T])
 ax.set(title='PBE '+cp,xlabel='Time after preparation (ps)',ylabel='Solid-like atom fraction (qbar6 > 0.25)',xlim=(0,50),ylim=(-.02,1.03));ax.legend();ax.grid(alpha=.2)
fig.suptitle('Ti coexistence: 1296 atoms, 50 ps; identical structural analysis')
for ext in ['png','pdf']:fig.savefig(OUT/('solid_fraction_comparison.'+ext),bbox_inches='tight')
plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for col,T in enumerate([1750,1800]):
 for cp,ls in [('5k','--'),('15k','-')]:
  a=cases[f'{cp}_{T}'];axs[0,col].plot(a['time'],a['fraction'][:,1],ls,label=cp);axs[1,col].plot(a['time'],a['width'][:,0],ls,label=cp)
 axs[0,col].set_title(f'{T} K');axs[0,col].set_ylabel('Solid-like fraction');axs[1,col].set_ylabel('Solid slab width / Lz')
 for ax in axs[:,col]:ax.set_xlim(0,50);ax.set_ylim(-.02,1.03);ax.set_xlabel('Time (ps)');ax.legend();ax.grid(alpha=.2)
fig.suptitle('Matched temperatures; undefined interface after loss of coexistence is not extrapolated')
for ext in ['png','pdf']:fig.savefig(OUT/('matched_1750_1800.'+ext),bbox_inches='tight')
plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for ax,key in zip(axs.flat,['5k_1750','15k_1750','5k_1800','15k_1800']):
 a=cases[key];im=ax.pcolormesh(a['time'],(np.arange(36)+.5)/36,a['profile'].T,shading='nearest',vmin=.08,vmax=.4,cmap='viridis')
 for col in [1,2]:
  y=a['width'][:,col].copy();y[np.r_[False,abs(np.diff(y))>.5]]=np.nan;ax.plot(a['time'],y,color='white',lw=.9)
 ax.set(title=key.replace('_',' / ')+' K',xlabel='Time (ps)',ylabel='z / Lz')
fig.colorbar(im,ax=axs,label='Mean qbar6');fig.suptitle('Interface profiles: white contours at smoothed qbar6=0.25')
for ext in ['png','pdf']:fig.savefig(OUT/('interface_maps_comparison.'+ext),bbox_inches='tight')
plt.close(fig)
