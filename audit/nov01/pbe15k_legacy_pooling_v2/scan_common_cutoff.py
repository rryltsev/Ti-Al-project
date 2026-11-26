"""Dense common-cutoff audit; fixed datasets and fit settings from pooling v2."""
from pathlib import Path
import os,sys,json,time,hashlib
BASE=Path(__file__).resolve().parent;OUT=BASE/'common_cutoff';OUT.mkdir(exist_ok=True)
WORK=BASE.parents[2];CAMP=WORK/'audit/oct29/pbe15k_ti_temperature'
sys.path.insert(0,str(WORK/'audit/oct20/deps'))
sys.path.insert(0,str(CAMP/'diagnostics/Ti_2000K_N10_cutoff/source'))
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MPLCONFIGDIR']=str(OUT/'mplcache')
import numpy as np,attrs
from scipy.optimize import brentq,minimize_scalar
from stacie import compute_spectrum,LorentzModel,CV2LCriterion
from stacie.estimate import fit_model_spectrum
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x): (OUT/n).write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
def merge(ss,weights=None):
 n=np.array([s.ndofs[0] for s in ss]);w=n/n.sum() if weights is None else np.asarray(weights)
 return attrs.evolve(ss[0],mean=sum(x*s.mean for x,s in zip(w,ss)),variance=sum(x*s.variance for x,s in zip(w,ss)),amplitudes=sum(x*s.amplitudes for x,s in zip(w,ss)),ndofs=sum(s.ndofs for s in ss))
def ch(p):
 xx,yy,zz,xy,xz,yz=p.T
 return np.array([(xx-.5*yy-.5*zz)/np.sqrt(3),.5*(yy-zz),xy,xz,yz])
keys=['old20','new_split20','new40','pooled20','new_first20','new_second20']
primary=keys[:4]
cache={};spectra={};replicas={}
for T in [1873,2000]:
 old=np.load(BASE/f'old_{T}.npz');oldspec=[compute_spectrum(s,prefactors=p,timestep=1e-15) for s,p in zip(old['seq'],old['pref'])]
 ns=[];nf=[];n1=[];n2=[]
 for i in range(1,11):
  d=BASE/'source/states'/f'Ti_{T}K/replicas/replica_{i:04d}';m=read(d/'summary.json');f=BASE/'source'/m['stress_file'];assert sha(f)==m['stress_sha256']
  a=np.load(f);p=ch(a['pressure_bar']*1e5);pf=np.full(5,float(a['volume_A3'])*1e-30/(1.380649e-23*T))
  sf=compute_spectrum(p,prefactors=pf,timestep=1e-15);s1=compute_spectrum(p[:,:20001],prefactors=pf,timestep=1e-15);s2=compute_spectrum(p[:,20000:],prefactors=pf,timestep=1e-15)
  nf.append(sf);n1.append(s1);n2.append(s2);ns.append(merge([s1,s2]))
 so=merge(oldspec);sn=merge(ns)
 spectra[T]={'old20':so,'new_split20':sn,'new40':merge(nf),'pooled20':merge([so,sn]),'new_first20':merge(n1),'new_second20':merge(n2)}
 replicas[T]={'old':oldspec,'new':ns,'full':nf};cache[T]={k:{} for k in keys}
def fit(s,fc):
 r=fit_model_spectrum(s,LorentzModel(),fc*1e12,8.,CV2LCriterion(),np.random.default_rng(42),100)
 assert 'acint' in r, (fc,r.get('msg'))
 return {k:float(r[k]) for k in ['acint','acint_std','cost_zscore','criterion_zscore','neff','fcut']}
def get(T,k,fc):
 f=round(float(fc),9)
 if f not in cache[T][k]:cache[T][k][f]=fit(spectra[T][k],f)
 return cache[T][k][f]
def z(T,ks,fc):return max(max(get(T,k,fc)['cost_zscore'],get(T,k,fc)['criterion_zscore']) for k in ks)
previous=read(BASE/'combined_results.json')
for T in [1873,2000]:
 for k in keys:
  for r in previous['temperatures'][str(T)]['scan'][k]:cache[T][k][round(r['fcut']/1e12,9)]=r
 # Reproduce the reference anchor with a fresh fit.
 for k in primary:
  r=fit(spectra[T][k],1.5);assert abs(r['acint']-get(T,k,1.5)['acint'])<1e-10
 # Broad scan of the old data determines whether any further candidate regions exist.
 broad=np.round(np.arange(.6,8.0001,.05),9)
 for fc in broad:get(T,'old20',fc)
 print('BROAD',T,flush=True)
 # Dense main scan; evaluate all windows, not just a favourable aggregate.
 dense=np.round(np.arange(.8,1.8001,.01),9)
 for k in keys:
  for fc in dense:get(T,k,fc)
  print('DENSE',T,k,flush=True)
 # Any additional old-data pass windows outside dense domain get all-window checks.
 for fc in broad:
  if (fc<.8 or fc>1.8) and z(T,['old20'],fc)<2:
   for k in keys:get(T,k,fc)
 save('scan_checkpoint.json',{str(t):{k:[v for _,v in sorted(rows.items())] for k,rows in by.items()} for t,by in cache.items()})
out={'created_unix':time.time(),'threshold':'one-sided cost_zscore < 2 AND criterion_zscore < 2, as production protocol; not absolute Z','broad_old_scan_THz':[.6,8.,.05],'dense_scan_THz':[.8,1.8,.01],'root_tolerance_THz':.00002,'primary_groups':primary,'strict_groups':keys,'not_a_stopping_rule':'neff>=60, N>=30 and sequential stability not included in these fit-quality intervals','temperatures':{}}
def intervals(T,ks):
 grid=sorted(set(np.round(np.arange(.8,1.8001,.01),9))|{f for f in np.round(np.arange(.6,8.0001,.05),9)})
 # If old data fail, the intersection fails regardless of other fits.
 def score(fc):
  zo=z(T,['old20'],fc)
  return zo if zo>=2 else z(T,ks,fc)
 vals=[score(fc)-2 for fc in grid];runs=[];start=None
 for i,(f,v) in enumerate(zip(grid,vals)):
  if v<0 and start is None:
   start=f if i==0 else brentq(lambda q:z(T,ks,q)-2,grid[i-1],f,xtol=.00002)
  if start is not None and (v>=0 or i==len(grid)-1):
   end=brentq(lambda q:z(T,ks,q)-2,grid[i-1],f,xtol=.00002) if v>=0 else f
   runs.append([start,end]);start=None
 return runs
def summary(T,lo,hi):
 grid=sorted(set([lo+1e-5,hi-1e-5,(lo+hi)/2]+[float(f) for f in np.arange(np.ceil(lo*1000)/1000,hi,.001)]))
 ans={}
 for k in keys:
  rows=[get(T,k,f) for f in grid];eta=np.array([r['acint']*1000 for r in rows]);se=np.array([r['acint_std']*1000 for r in rows])
  imin=int(np.argmin(eta));imax=int(np.argmax(eta))
  ans[k]={'eta_min':float(eta.min()),'eta_max':float(eta.max()),'eta_span_pct_of_midpoint':float((eta.max()-eta.min())/((eta.max()+eta.min())/2)*100),'formal_se_min':float(se.min()),'formal_se_max':float(se.max()),'eta_at_low':float(eta[0]),'eta_at_high':float(eta[-1]),'se_at_low':float(se[0]),'se_at_high':float(se[-1]),'fc_at_eta_min':grid[imin],'fc_at_eta_max':grid[imax],'max_cost_z':max(r['cost_zscore'] for r in rows),'max_cutoff_z':max(r['criterion_zscore'] for r in rows),'neff_min':min(r['neff'] for r in rows),'neff_max':max(r['neff'] for r in rows)}
 return ans
for T in [1873,2000]:
 e={'primary_intervals':intervals(T,primary),'strict_intervals':intervals(T,keys)}
 coarse=sorted(cache[T]['old20'].items());best=min(coarse,key=lambda x:x[1]['criterion_zscore'])[0]
 mini=minimize_scalar(lambda f:get(T,'old20',f)['criterion_zscore'],bounds=(max(.6,best-.05),min(8,best+.05)),method='bounded',options={'xatol':1e-6})
 e['old_cutoff_minimum']={'cutoff_THz':float(mini.x),**get(T,'old20',mini.x)}
 e['primary_ranges']=[{'interval':r,'summaries':summary(T,*r)} for r in e['primary_intervals']]
 e['strict_ranges']=[{'interval':r,'summaries':summary(T,*r)} for r in e['strict_intervals']]
 out['temperatures'][str(T)]=e
 print('INTERVALS',T,json.dumps({k:v for k,v in e.items() if not k.endswith('_ranges')}),flush=True)
 save('results.json',out)
# Across-temperature intersections.
for kind in ['primary','strict']:
 rs=[]
 for a,b in out['temperatures']['1873'][kind+'_intervals']:
  for c,d in out['temperatures']['2000'][kind+'_intervals']:
   lo=max(a,c);hi=min(b,d)
   if hi>lo:rs.append([lo,hi])
 out[kind+'_shared_between_temperatures']=rs
 out[kind+'_shared_ranges']=[{'interval':r,'temperatures':{str(T):summary(T,*r) for T in [1873,2000]}} for r in rs]
out['finished_unix']=time.time();save('results.json',out)
save('scan.json',{str(t):{k:[v for _,v in sorted(rows.items())] for k,rows in by.items()} for t,by in cache.items()})
print('FINISHED',flush=True)

