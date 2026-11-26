"""CPU-only tests of stress normalization, spectral scale and stopping safeguards."""
import ast,json
from pathlib import Path
import numpy as np
import analysis_common as a

for name in ['workflow.py','md_core.py','analyze_point.py','analysis_common.py']:
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

def mock(n,eta=3.,neff=80):
    points=[]
    for s in a.CFG['states']:
        w=dict(eta_mPa_s=eta,eta_std_mPa_s=.06,relative_std=.02,neff=neff,cost_z=.4,criterion_z=.5)
        points.append(dict(id=s['id'],windows={name:dict(w) for name in ['40ps','first20ps','second20ps']},new_batch_z=.5,new_batch_40ps=dict(w)))
    return dict(n_replicas=n,points=points)
h=[mock(n) for n in [20,25]];cur=mock(30)
res=a.convergence(cur['points'],h,30)
assert res['all_passed'] and not res['stop_recommended'] and res['pass_streak']==1
cur.update(res);h.append(cur)
assert a.convergence(mock(35)['points'],h,35)['stop_recommended']
assert not a.convergence(mock(5)['points'],[],5)['stop_recommended']
assert not a.convergence(mock(35,neff=30)['points'],h,35)['all_passed']
assert not a.convergence(mock(35,eta=4)['points'],h,35)['all_passed']
bad=mock(35);bad['points'][0]['windows']['first20ps']['eta_mPa_s']=4
assert not a.convergence(bad['points'],h,35)['all_passed']
bad=mock(35);bad['points'][0]['windows']['second20ps']['criterion_z']=8.
assert not a.convergence(bad['points'],h,35)['stop_recommended']
# Equal pressure/volume data must scale as 1/T, not use 1873 K for all states.
scaled,_,_=a.fit(seq,np.ones(40)*a.prefactor(10000,2100))
assert np.isclose(scaled['eta_mPa_s']/out['eta_mPa_s'],a.prefactor(10000,2100),rtol=1e-8)
assert np.isclose(a.prefactor(10000,1873)/a.prefactor(10000,2100),2100/1873)
# Exercise scheduling transitions without MD and without touching replica files.
import workflow as wf
counts={s['id']:0 for s in a.CFG['states']};reports={}
wf.md.completed=lambda s:[None]*counts[s['id']]
wf.latest=lambda s:reports.get(s['id'],{})
assert wf.choose()['temperature_K']==2100
counts['Ti_2100K']=5;assert wf.choose()['temperature_K']==2100
counts['Ti_2100K']=10;assert wf.choose()['temperature_K']==1873
counts['Ti_1873K']=10;assert wf.choose()['temperature_K']==2000
counts['Ti_2000K']=10;assert wf.choose()['temperature_K']==2100
reports['Ti_2100K']={'review_required':True};assert wf.choose()['temperature_K']==1873
reports.update({s:{'stop_recommended':True} for s in counts});assert wf.choose() is None
print(json.dumps(dict(status='PASS',tests=['hydrostatic invariance','shear normalization',
    'AR1 integral units and normalization','minimum samples','repeated pass requirement',
    'spectral resolution rejection','drift rejection','duration rejection','window cutoff guard','temperature prefactor','priority and review scheduling'],
    ar1_expected_mPa_s=exact,ar1_estimate_mPa_s=out['eta_mPa_s'])))
