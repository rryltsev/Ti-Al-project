"""CPU-only tests of stress normalization, spectral scale and stopping safeguards."""
import ast,json
from pathlib import Path
import numpy as np
import analysis_common as a

for name in ['md_core.py','analyze_point.py','analysis_common.py','pooled_analysis.py']:
    ast.parse((Path(__file__).parent/name).read_text())

# A hydrostatic offset must have no effect on any shear channel.
rng=np.random.default_rng(712093)
p=rng.normal(size=(100,6));q=p.copy();q[:,:3]+=1e5
assert np.allclose(a.channels(p),a.channels(q),atol=3e-11)
assert np.allclose(a.channels(p)[2:],p[:,3:].T)

# Known AR(1) integral, physical seconds. Detect missing factor 2 or 1/5 and unit errors.
phi=.9;noise=rng.normal(size=(40,8192));seq=np.zeros_like(noise)
seq[:,0]=noise[:,0]
for i in range(1,seq.shape[1]):seq[:,i]=phi*seq[:,i-1]+np.sqrt(1-phi**2)*noise[:,i]
out,_,_=a.fit(seq,np.ones(40))
exact=.5e-15*(1+phi)/(1-phi)*1000
assert abs(out['eta_mPa_s']-exact)<max(5*out['eta_std_mPa_s'],.12*exact),(out,exact)

import pooled_analysis as pool
x=rng.normal(size=(5,20001));y=rng.normal(size=(10,20001));px=np.arange(1,6);py=np.arange(1,11)
m=pool.merge([pool.spectrum(x,px),pool.spectrum(y,py)]);direct=pool.spectrum(np.concatenate([x,y]),np.r_[px,py])
for key in ['amplitudes','ndofs','mean','variance']:assert np.allclose(getattr(m,key),getattr(direct,key),rtol=1e-12,atol=1e-25)
assert 'min_replicas' not in a.CFG['convergence']
window=dict(eta_mPa_s=3.,eta_std_mPa_s=.3,relative_std=.1,neff=25.,cost_z=4.,criterion_z=5.)
row=dict(id=a.CFG['states'][0]['id'],windows={k:dict(window) for k in ['40ps','first20ps','second20ps']},combined_20ps=dict(eta_mPa_s=3.,eta_std_mPa_s=.06,relative_std=.02,cost_z=.1,criterion_z=.2),new_batch_z=.5)
r=a.convergence([row],[],5)
assert r['total_replica_clusters']==a.CFG['legacy']['n_replicas']+5 and r['assessment_pending'] and not r['stop_recommended']
assert r['checks'][0]['flags']['combined_precision'] and r['checks'][0]['new40_diagnostics']['40ps']['neff']==25
baseline=json.loads((a.ROOT/'verified_legacy_baseline.json').read_text());assert baseline['status']=='PASS' and baseline['legacy_manifest_sha256']==a.CFG['legacy']['sha256'] and baseline['checkpoint_sha256']==a.CFG['checkpoint']['sha256']
assert a.sha(a.ROOT/'legacy/manifest.json')==a.CFG['legacy']['sha256']
print(json.dumps(dict(status='PASS',tests=['hydrostatic invariance','AR1 normalization','weighted spectrum merge','old+new cluster count','no new30 quota','diagnostics do not extend finite plan','verified legacy model audit retained'])))
