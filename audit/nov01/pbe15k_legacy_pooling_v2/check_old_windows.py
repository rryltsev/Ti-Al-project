from pathlib import Path
import os,sys,json
p=Path('audit/nov01/pbe15k_legacy_pooling_v2')
sys.path.insert(0,'audit/oct20/deps');sys.path.insert(0,'audit/oct29/pbe15k_ti_temperature/diagnostics/Ti_2000K_N10_cutoff/source')
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np,attrs
from stacie import compute_spectrum,LorentzModel,CV2LCriterion
from stacie.estimate import fit_model_spectrum
def fit(ss):
 n=np.array([s.ndofs[0] for s in ss]);w=n/n.sum()
 s=attrs.evolve(ss[0],mean=sum(x*s.mean for x,s in zip(w,ss)),variance=sum(x*s.variance for x,s in zip(w,ss)),amplitudes=sum(x*s.amplitudes for x,s in zip(w,ss)),ndofs=sum(s.ndofs for s in ss))
 r=fit_model_spectrum(s,LorentzModel(),1.5e12,8.,CV2LCriterion(),np.random.default_rng(42),100)
 return {k:float(r[k]) for k in ['acint','acint_std','cost_zscore','criterion_zscore']}
out={}
for T in [1873,2000]:
 data=np.load(p/f'old_{T}.npz');seq=data['seq'];pref=data['pref']
 spectra=[[compute_spectrum(s[:,sl],prefactors=pf,timestep=1e-15) for s,pf in zip(seq,pref)] for sl in [slice(0,10001),slice(10000,20001)]]
 a,b=[fit(ss) for ss in spectra];vals=[]
 for i in range(len(seq)):
  aa,bb=[fit([s for j,s in enumerate(ss) if j!=i]) for ss in spectra]
  vals.append((aa['acint']-bb['acint'])*1000)
 vals=np.array(vals)
 out[T]={'first10':a,'second10':b,'difference_mPa_s':(a['acint']-b['acint'])*1000,'paired_replica_jackknife_se':float(np.sqrt((len(vals)-1)/len(vals)*np.sum((vals-vals.mean())**2))),'cutoff_THz':1.5,'caveat':'10 ps spectra have fewer low-frequency points; this is an exploratory stationarity check.'}
 print(T,json.dumps(out[T]),flush=True)
(p/'old_time_windows.json').write_text(json.dumps(out,indent=2)+'\n')

