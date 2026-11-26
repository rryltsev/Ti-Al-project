"""Portable reproduction from saved means and joint blocks; no project paths.
Usage: python reproduce_C46_phase_contacts.py DATA.json [--atoms5120] [--kindQHA]
"""
from pathlib import Path
import numpy as np,json,sys
from scipy.interpolate import PchipInterpolator
from scipy.optimize import root
def tangent(x,y,cls=PchipInterpolator):
 x=np.asarray(x);y=np.asarray(y);order=np.argsort(x);x=x[order];y=y[order]
 if np.min(np.diff(x))<=0:raise ValueError('nonmonotone mean concentrations')
 shift=np.median(y);mu=cls(x,y-shift,extrapolate=False);F=mu.antiderivative()
 grid=np.linspace(x[0],x[-1],3001);f=F(grid);h=[]
 for i in range(len(grid)):
  while len(h)>1 and (f[h[-1]]-f[h[-2]])/(grid[h[-1]]-grid[h[-2]]) >= (f[i]-f[h[-1]])/(grid[i]-grid[h[-1]]):h.pop()
  h.append(i)
 gaps=np.diff(h)
 if not len(gaps) or max(gaps)<20:return dict(resolved=False,reason='no separated convex-hull contact pair')
 k=int(np.argmax(gaps));ini=grid[np.array(h)[[k,k+1]]]
 def eq(z):
  a,b=z
  if b<=a:return [1e2,1e2]
  return [mu(a)-mu(b),(F(b)-F(a))/(b-a)-mu(a)]
 r=root(eq,ini,tol=1e-10);a,b=r.x
 if not r.success or not(x[0]<a<b<x[-1]) or b-a<.005 or max(abs(np.array(eq(r.x))))>1e-7:
  return dict(resolved=False,reason='common tangent not resolved inside sampled composition domain',candidate=r.x.tolist(),sampled_c=[float(x[0]),float(x[-1])])
 slope=float(mu(a));fg=F(grid);gap=fg-(F(a)+slope*(grid-a));bar=float(np.max(gap[(grid>=a)&(grid<=b)]))
 if float(np.min(gap)) < -1e-7:return dict(resolved=False,reason='tangent is not globally supporting',min_gap_eV_atom=float(np.min(gap)))
 return dict(resolved=True,c_alpha=float(a),c_alpha2=float(b),delta_mu_eV=float(slope+shift),finite_cell_barrier_eV_atom=bar,root_residual_eV=float(max(abs(np.array(eq(r.x))))),sampled_c=[float(x[0]),float(x[-1])])

def reproduce(group,attempts,seed):
 windows=group['windows'];x=np.array([r['mean_c'] for r in windows]);y=np.array([r['mu_occ_eV'] for r in windows]);result=tangent(x,y)
 if not result['resolved']:return result
 rng=np.random.default_rng(seed);draw=[]
 blocks=[np.array(r['joint_blocks_c_mu_eV']) for r in windows]
 for i in range(attempts):
  pair=np.array([b[rng.integers(0,len(b),len(b))].mean(axis=0) for b in blocks]);r=tangent(pair[:,0],pair[:,1])
  if r['resolved']:draw.append([r['c_alpha'],r['c_alpha2'],r['delta_mu_eV'],r['finite_cell_barrier_eV_atom']])
 result['bootstrap_resolved']=len(draw)
 if draw:result['bootstrap_percentiles']=np.percentile(draw,[2.5,50,97.5],axis=0).tolist()
 return result
if __name__=='__main__':
 d=json.loads(Path(sys.argv[1]).read_text());results=[]
 for g in d['groups']:
  if '--atoms5120' in sys.argv and g['atoms']!=5120:continue
  if '--kindQHA' in sys.argv and g['kind']!='QHA':continue
  expected=g['expected'];attempts=expected.get('bootstrap',{}).get('attempts',0);r=reproduce(g,attempts,d['bootstrap_seed']);assert r['resolved']==expected['resolved']
  if r['resolved']:
   for k in ['c_alpha','c_alpha2','delta_mu_eV','finite_cell_barrier_eV_atom']:assert abs(r[k]-expected[k])<1e-10,(g['kind'],k,r[k],expected[k])
   assert r['bootstrap_resolved']==expected['bootstrap']['resolved']
   if r.get('bootstrap_percentiles'):
    for actual,k in zip(r['bootstrap_percentiles'],['percentile_2p5','median','percentile_97p5']):assert np.max(abs(np.array(actual)-expected['bootstrap'][k]))<1e-10
  results.append(dict(kind=g['kind'],T_K=g['T_K'],atoms=g['atoms'],reproduced=True,result=r))
 print(json.dumps(dict(groups=results,qualification=d['qualification']),indent=2))
