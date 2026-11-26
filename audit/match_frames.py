from pathlib import Path
from collections import Counter
import numpy as np
import re, json
B=Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project')
def readxyz(p):
 out=[]
 with p.open(encoding='utf8') as f:
  while l:=f.readline():
   n=int(l);h=f.readline();box=np.fromstring(re.search(r'Lattice="([^"]+)"',h)[1],sep=' ').reshape(3,3)
   r=[f.readline().split() for _ in range(n)];s=np.array([x[0]=='Ti' for x in r]);c=np.array([x[1:4] for x in r],float)
   out.append((s,c,box))
 return out
ref=readxyz(B/'MACE-dataset-gen/MACE-database/TiAl-liq-relax-ss_select-1000.xyz')
boxes=np.array([x[2] for x in ref]);counts=np.array([(len(x[0]),sum(x[0])) for x in ref])
def match(frame):
 s,c,b=frame
 cand=np.where((np.max(abs(boxes-b),axis=(1,2))<5e-5)&(counts[:,0]==len(s))&(counts[:,1]==sum(s)))[0]
 for i in cand:
  rs,rc,rb=ref[i];ok=True
  for t in [False,True]:
   a=c[s==t];z=rc[rs==t]
   if not len(a):continue
   delta=(a[:,None,:]-z[None,:,:])@np.linalg.inv(rb);delta-=np.rint(delta);delta=delta@rb
   if np.max(np.min(np.linalg.norm(delta,axis=2),axis=1))>5e-5:ok=False;break
  if ok:return int(i)
 return None
results={}
for p in sorted((B/'DPA4-finetune').glob('*/npy_format')):
 d={}
 for f in p.glob('*.xyz'):
  frames=readxyz(f);ids=[match(x) for x in frames];d['all']=ids
  print(p.parent.name,'all',len(ids),'matched',sum(x is not None for x in ids),'unmatched_positions',[i for i,x in enumerate(ids) if x is None],flush=True)
 for split in ['train','validate']:
  ids=[];bad=0
  for f in (p/split).rglob('coord.npy'):
   sys=f.parent.parent;types=np.loadtxt(sys/'type.raw',dtype=int,ndmin=1);mp=(sys/'type_map.raw').read_text().split();s=np.array([mp[t]=='Ti' for t in types]);n=len(s)
   cs=np.load(f).reshape(-1,n,3);bs=np.load(f.with_name('box.npy')).reshape(-1,3,3)
   for kind in ['coord','box','energy','force','virial']:bad+=int(not np.isfinite(np.load(f.with_name(kind+'.npy'))).all())
   ids.extend(match((s,c,b)) for c,b in zip(cs,bs))
  d[split]=ids
  print(p.parent.name,split,len(ids),'matched',sum(x is not None for x in ids),'nonfinite_arrays',bad,flush=True)
 print('train-val duplicate IDs',set(d['train'])&set(d['validate']),flush=True)
 results[p.parent.name]=d
for a in results:
 for b in results:
  if a<b:print('PAIR',a,b,{s+'-'+t:len((set(results[a][s])-{None})&(set(results[b][t])-{None})) for s,t in [('all','all'),('train','train'),('validate','validate'),('train','validate'),('validate','train')]},flush=True)
Path('audit/frame_matches.json').write_text(json.dumps(results,indent=2))
