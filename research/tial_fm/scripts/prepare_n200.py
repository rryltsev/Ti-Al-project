"""Nested label-free train subsets and exact DeepMD row subsets."""
from pathlib import Path
from collections import defaultdict,Counter
import hashlib,json,shutil
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'datasets/common992-grouped-v1-20261106-02'
OUT=ROOT/'datasets/nested-train-v1-20261106'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n',encoding='utf8')
def main():
 OUT.mkdir(exist_ok=False);splits=read(DATA/'splits.json');records=[json.loads(x) for x in (DATA/'data_manifest.jsonl').read_text(encoding='utf8').splitlines()]
 rr={r['geometry_id']:r for r in records if r['geometry_id'] in splits['split_ids']['train']};assert len(rr)==694
 strata=defaultdict(list)
 def stratum(r):return r['structural_family'],min(4,int(5*r['composition'].get('Al',0)/r['natoms']))
 for sid,r in rr.items():strata[stratum(r)].append(sid)
 rng=np.random.default_rng(20261106);tie={i:float(rng.random()) for i in sorted(rr)}
 order=[];chosen=set();counts=Counter();groupcounts=Counter();vol={i:float(np.log(np.linalg.det(r['cell_A'])/r['natoms'])) for i,r in rr.items()}
 def add(sid):order.append(sid);chosen.add(sid);counts[stratum(rr[sid])]+=1;groupcounts[rr[sid]['group_id']]+=1
 # Force elemental references and each source family into every nested subset.
 for pred in [lambda r:r['composition'].get('Al',0)==r['natoms'],lambda r:r['composition'].get('Ti',0)==r['natoms']]:
  add(min((i for i,r in rr.items() if pred(r) and i not in chosen),key=lambda i:tie[i]))
 for fam in sorted({r['structural_family'] for r in rr.values()}):
  if not any(rr[i]['structural_family']==fam for i in chosen):add(min((i for i,r in rr.items() if r['structural_family']==fam),key=lambda i:tie[i]))
 while len(order)<694:
  s=max((s for s,ids in strata.items() if any(i not in chosen for i in ids)),key=lambda s:((len(order)+1)*len(strata[s])/694-counts[s],str(s)))
  candidates=[i for i in strata[s] if i not in chosen]
  # First diversify groups within each stratum, then spread volume within group.
  def score(i):
   g=rr[i]['group_id'];prev=[j for j in order if rr[j]['group_id']==g];dist=min((abs(vol[i]-vol[j]) for j in prev),default=1.)
   return (-groupcounts[g]/sum(r['group_id']==g for r in rr.values()),dist,tie[i])
  add(max(candidates,key=score))
 subsets={str(n):order[:n] for n in [100,200,400,694]}
 manifest={'version':OUT.name,'seed':20261106,'parent_split_sha256':sha(DATA/'splits.json'),'algorithm':'Sequential largest-deficit sampling over source-family x 5 Al-fraction bins; pure Al/Ti and each family mandatory; prefer underrepresented composition groups, then max distance in log(volume/atom). Fixed RNG tie order. No DFT/model values.','train_order':order,'subsets':subsets,'validation_ids':splits['split_ids']['validation'],'test_ids':splits['split_ids']['test'],'subsets_may_partially_sample_train_groups':True,'unused_train_frames_are_not_validation_or_test':True}
 assert set(order)==set(splits['split_ids']['train']) and len(order)==len(set(order))
 ids=set(subsets['200']);checks=[]
 for xc in ['PBE','PBEsol','r2SCAN']:
  systems=[];exported=[]
  for src in sorted((DATA/'deepmd'/xc/'train').iterdir()):
   if not src.is_dir():continue
   oldids=read(src/'geometry_ids.json');idx=[j for j,i in enumerate(oldids) if i in ids]
   if not idx:continue
   dst=OUT/'deepmd'/xc/'train'/src.name;(dst/'set.000').mkdir(parents=True)
   for name in ['type.raw','type_map.raw']:shutil.copyfile(src/name,dst/name)
   keep=[oldids[j] for j in idx];dump(dst/'geometry_ids.json',keep);exported+=keep
   for k in ['coord','box','energy','force','virial']:
    original=np.load(src/'set.000'/(k+'.npy'),allow_pickle=False);a=original[idx];np.save(dst/'set.000'/(k+'.npy'),a)
    assert np.array_equal(np.load(dst/'set.000'/(k+'.npy'),allow_pickle=False),original[idx])
   systems.append(str(dst.resolve()))
  assert set(exported)==ids and len(exported)==200
  dump(OUT/'deepmd'/xc/'train_systems.json',systems);checks.append(xc)
 manifest['N200_counts']={'geometries':200,'atoms':sum(rr[i]['natoms'] for i in ids),'groups':len({rr[i]['group_id'] for i in ids}),'families':dict(Counter(rr[i]['structural_family'] for i in ids)),'label_budget_per_XC':{'train':200,'development':149,'train_plus_development':349,'test':149}}
 dump(OUT/'subsets.json',manifest);shutil.copyfile(__file__,OUT/'prepare_n200.py')
 dump(OUT/'verification.json',{'status':'PASS','N200_exact_rows_verified_for':checks,'train_validation_test_overlap':0,'nested_sizes':[100,200,400,694],'counts':manifest['N200_counts']})
 dump(OUT/'FILES_SHA256.json',{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()})
 shutil.copyfile(OUT/'subsets.json',ROOT/'manifests/nested_train_subsets.json')
 print(json.dumps(manifest['N200_counts'],indent=2))
if __name__=='__main__':main()
