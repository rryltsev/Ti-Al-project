"""Matched 0.5 ps sampling and unchanged structural classifier for 5k/15k."""
from pathlib import Path
import sys,json,hashlib,re
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
sys.path.insert(0,str(ROOT/'audit/melting'))
from summarize_melting import read_dump,order,slab_width,THRESHOLDS,np,plt,ideal_check,cKDTree,sph_harm_y
RUN=BASE/'pbe15k_melting_1775';SRC=RUN/'collected';OUT=RUN/'analysis';OUT.mkdir(exist_ok=True)
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
 a=sections[-1];data=np.array(a['rows']);headers=a['columns'];assert data[-1,headers.index('Step')]==100000
 keep=data[:,headers.index('Step')]>=5000
 return {'mean_T_5_100_K':float(data[keep,headers.index('Temp')].mean()),'mean_P_5_100_GPa':float(data[keep,headers.index('Press')].mean()/10000),'thermo_rows':len(data)}
cases={};summaries={}
for T in [1775]:
 path=SRC/f'T{T}K/melt/trajectory.lammpstrj';cache=OUT/f'15k_{T}.npz'
 if not cache.exists():
  frames=read_dump(path);assert len(frames)==1001 and frames[-1][0]==100000
  frames=frames[::5];data={k:[] for k in ['time','fraction','width','profile','bonds_fraction','cell','qbar6']}
  for step,cell,xyz,pe in frames:
   q,qb,bonds=order(xyz,cell);bins=np.minimum((xyz[:,2]/cell[2]*36).astype(int),35);profile=np.bincount(bins,weights=qb,minlength=36)/np.maximum(np.bincount(bins,minlength=36),1)
   vals=[step*.001,(qb[:,None]>THRESHOLDS).mean(axis=0),slab_width(profile),profile,(bonds>=7).mean(),cell,qb]
   for k,v in zip(data,vals):data[k].append(v)
  data={k:np.array(v) for k,v in data.items()};data['end_qbar4']=qbar4(frames[-1][2],frames[-1][1]);data['source_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();np.savez_compressed(cache,**data)
 with np.load(cache) as f:cases[f'15k_{T}']={k:f[k] for k in f.files}

print(json.dumps(thermo(SRC/"T1775K/melt/log.lammps")),flush=True)
print("Analysis cache ready",flush=True)
