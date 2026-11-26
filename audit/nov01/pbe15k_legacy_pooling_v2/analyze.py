"""Separate fixed-cutoff pooling; formal and grouped resampling uncertainties."""
from pathlib import Path
import sys,os,json,time,hashlib
OUT=Path(__file__).resolve().parent;WORK=OUT.parents[2];CAMP=WORK/'audit/oct29/pbe15k_ti_temperature'
sys.path.insert(0,str(WORK/'audit/oct20/deps'))
sys.path.insert(0,str(CAMP/'diagnostics/Ti_2000K_N10_cutoff/source'))
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MPLCONFIGDIR']=str(OUT/'mplcache')
import numpy as np,attrs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stacie import compute_spectrum,estimate_acint,LorentzModel,CV2LCriterion
from stacie.estimate import fit_model_spectrum
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,x): (OUT/name).write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def ch(p):
 xx,yy,zz,xy,xz,yz=p.T
 return np.array([(xx-.5*yy-.5*zz)/np.sqrt(3),.5*(yy-zz),xy,xz,yz])
def merge(ss,weights=None):
 n=np.array([s.ndofs[0] for s in ss]); w=n/n.sum() if weights is None else np.asarray(weights)
 return attrs.evolve(ss[0],mean=sum(x*s.mean for x,s in zip(w,ss)),variance=sum(x*s.variance for x,s in zip(w,ss)),amplitudes=sum(x*s.amplitudes for x,s in zip(w,ss)),ndofs=sum(s.ndofs for s in ss))
def fixed(s,fc):
 r=fit_model_spectrum(s,LorentzModel(),fc*1e12,8.,CV2LCriterion(),np.random.default_rng(42),100)
 assert 'acint' in r,r.get('msg')
 return {k:float(r[k]) for k in ['acint','acint_std','cost_zscore','criterion_zscore','neff','fcut']}
def jk(vals):
 a=np.array(vals);return float(np.sqrt((len(a)-1)/len(a)*np.sum((a-a.mean())**2)))
model=read(OUT/'model_check.json');assert model['passed'],'Do not pool incompatible inputs'
out={'version':2,'created_unix':time.time(),'purpose':'Separate exploratory pooling, fixed cutoff, no production replacement','model_check':'model_check.json','main_cutoff_THz':1.5,'cutoff_choice':'Pre-existing 1.5 THz diagnostic anchor, set before historical fit inspection','normalization':'per replica V/(kB*T), pressure bar to Pa, five shear channels','segmentation':'Both 20001-sample halves of each new 40001-sample trajectory; shared boundary sample retained to match existing analyses; halves and all channels stay together in resampling','uncertainty':'Formal STACIE SE plus stratified delete-one-replica jackknife; sensitivity groups five consecutive old replicas, retaining ten whole new trajectories. These do not prove independence between preparation reservoirs.','temperatures':{}}
grid=[.8,1.,1.2,1.5,1.8,2.,2.5,3.,4.]
for T in [1873,2000]:
 old=np.load(OUT/f'old_{T}.npz'); ospec=[compute_spectrum(s,prefactors=p,timestep=1e-15) for s,p in zip(old['seq'],old['pref'])]
 nspec=[];full=[];first=[];second=[];manifest=[]
 ref=read(CAMP/f'collected/analysis/Ti_{T}K/batch_0010/summary.json')
 for i in range(1,11):
  d=OUT/'source/states'/f'Ti_{T}K/replicas/replica_{i:04d}'
  m=read(d/'summary.json');f=OUT/'source'/m['stress_file']
  assert sha(f)==m['stress_sha256']==ref['points'][0]['sources'][i-1]['sha256']
  a=np.load(f);p=ch(a['pressure_bar']*1e5);v=float(a['volume_A3']);pf=np.full(5,v*1e-30/(1.380649e-23*T))
  full.append(compute_spectrum(p,prefactors=pf,timestep=1e-15))
  s1=compute_spectrum(p[:,:20001],prefactors=pf,timestep=1e-15);s2=compute_spectrum(p[:,20000:],prefactors=pf,timestep=1e-15)
  first.append(s1);second.append(s2);nspec.append(merge([s1,s2]))
  manifest.append({'replica':i,'stress':str(f),'sha256':sha(f),'volume_A3':v})
 so=merge(ospec);sn=merge(nspec);sp=merge([so,sn]);sf=merge(full)
 # Verify direct pooling matches the algebraically combined per-replica spectrum.
 direct=compute_spectrum(np.concatenate(old['seq']),prefactors=np.concatenate(old['pref']),timestep=1e-15)
 assert np.allclose(direct.amplitudes,so.amplitudes,rtol=1e-12,atol=1e-15)
 entry={'old_n':len(ospec),'new_n':10,'old_ps':20*len(ospec),'new_ps':400,'new_manifest':manifest,'scan':{},'jackknife':{}}
 spectra={'old20':so,'new_split20':sn,'pooled20':sp,'new40':sf,'new_first20':merge(first),'new_second20':merge(second)}
 for k,s in spectra.items():
  entry['scan'][k]=[fixed(s,fc) for fc in grid]
  print('SCAN',T,k,flush=True)
 # Validate new automatic full result exactly reproduces the published batch.
 auto=estimate_acint(sf,LorentzModel(),verbose=False)
 assert abs(auto.acint*1000-ref['points'][0]['windows']['40ps']['eta_mPa_s'])<2e-5
 entry['new40_automatic']={'eta_mPa_s':float(auto.acint*1000),'std_mPa_s':float(auto.acint_std*1000)}
 # In stratified deletion keep original old/new observation weights fixed.
 weights=np.array([len(ospec),2*len(nspec)],float);weights/=weights.sum()
 for block in [1,5]:
  oldgroups=[list(range(i,min(i+block,len(ospec)))) for i in range(0,len(ospec),block)]
  valsold=[];valspoolold=[]
  for group in oldgroups:
   sub=merge([s for j,s in enumerate(ospec) if j not in group])
   valsold.append(fixed(sub,1.5)['acint']*1000)
   valspoolold.append(fixed(merge([sub,sn],weights),1.5)['acint']*1000)
  valsnew=[];valspoolnew=[]
  for omit in range(len(nspec)):
   sub=merge([s for j,s in enumerate(nspec) if j!=omit])
   valsnew.append(fixed(sub,1.5)['acint']*1000)
   valspoolnew.append(fixed(merge([so,sub],weights),1.5)['acint']*1000)
  entry['jackknife'][f'old_block_{block}']={'old_se':jk(valsold),'new_se':jk(valsnew),'pooled_se':float(np.hypot(jk(valspoolold),jk(valspoolnew))),'old_new_difference_se':float(np.hypot(jk(valsold),jk(valsnew))),'old_group_count':len(oldgroups),'old_leaveout_eta':valsold,'new_leaveout_eta':valsnew,'pool_old_leaveout_eta':valspoolold,'pool_new_leaveout_eta':valspoolnew}
 # Pair the new full trajectory in its SE; compare at exactly the same fc.
 vfull=[fixed(merge([s for j,s in enumerate(full) if j!=i]),1.5)['acint']*1000 for i in range(10)]
 entry['new40_jackknife_se']=jk(vfull)
 out['temperatures'][str(T)]=entry;save('combined_results.json',out)
 print('RESULT',T,json.dumps({k:v[3] for k,v in entry['scan'].items()}),json.dumps({k:{a:b for a,b in v.items() if 'leaveout' not in a} for k,v in entry['jackknife'].items()}),flush=True)
out['finished_unix']=time.time();save('combined_results.json',out)
fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
colors={'old20':'#9c540a','new_split20':'#2274a5','pooled20':'#22865d','new40':'#9656a6'}
labels={'old20':'Old 20 ps','new_split20':'New 40 ps split into 2 x 20 ps','pooled20':'Pooled 20 ps segments','new40':'New full 40 ps'}
for row,(T,e) in enumerate(out['temperatures'].items()):
 for k in colors:
  r=e['scan'][k];x=np.array([a['fcut']/1e12 for a in r]);y=np.array([a['acint']*1000 for a in r]);se=np.array([a['acint_std']*1000 for a in r])
  axs[row,0].plot(x,y,'o-',ms=3,color=colors[k],label=labels[k]);axs[row,0].fill_between(x,y-se,y+se,color=colors[k],alpha=.1)
  axs[row,1].plot(x,[a['criterion_zscore'] for a in r],'o-',ms=3,color=colors[k],label=labels[k])
 for ax in axs[row]:
  ax.axvline(1.5,ls=':',color='gray');ax.set_xlabel('Common cutoff (THz)');ax.grid(alpha=.2)
 axs[row,0].set_title(f'{T} K: {e["old_n"]} old + 10 new trajectories')
 axs[row,0].set_ylabel('Viscosity (mPa s)');axs[row,1].set_ylabel('Cutoff diagnostic Z');axs[row,1].axhline(2,ls='--',color='black')
axs[0,0].legend(fontsize=8)
fig.suptitle('Separate pooling diagnostic | shaded bands: formal STACIE SE')
fig.savefig(OUT/'pooling_comparison.png',dpi=180);fig.savefig(OUT/'pooling_comparison.pdf');plt.close(fig)
print('DONE',flush=True)

