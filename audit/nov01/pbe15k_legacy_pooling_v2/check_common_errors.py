"""Grouped uncertainty checks inside the common interval; same v2 inputs."""
from pathlib import Path
import json,time
path=Path(__file__).resolve().parent/'scan_common_cutoff.py'
namespace={'__file__':str(path),'__name__':'cutoff_helpers'}
exec(compile(path.read_text(encoding='utf-8-sig').split("previous=read(BASE/'combined_results.json')")[0],str(path),'exec'),namespace)
np=namespace['np'];fit=namespace['fit'];merge=namespace['merge'];replicas=namespace['replicas'];spectra=namespace['spectra']
outdir=path.parent/'common_cutoff'
r=json.loads((outdir/'results.json').read_text())
common=r['primary_shared_between_temperatures'];assert len(common)==1
lo,hi=common[0];fcs=[lo+.001,(lo+hi)/2,hi-.001]
def jk(v):
 a=np.array(v);return float(np.sqrt((len(a)-1)/len(a)*np.sum((a-a.mean())**2)))
out={'cutoffs_THz':fcs,'scope':'Three interior anchors in shared primary interval; whole-trajectory deletion, both new halves retained together; not a continuous extrema guarantee.','temperatures':{}}
for T in [1873,2000]:
 old=replicas[T]['old'];new=replicas[T]['new'];full=replicas[T]['full'];so=merge(old);sn=merge(new);weights=np.array([len(old),2*len(new)],float);weights/=weights.sum()
 rows=[]
 for fc in fcs:
  row={'fcut_THz':fc}
  for block in [1,5]:
   go=[list(range(i,min(i+block,len(old)))) for i in range(0,len(old),block)]
   vo=[];vpo=[];vn=[];vpn=[]
   for group in go:
    sub=merge([s for i,s in enumerate(old) if i not in group])
    vo.append(fit(sub,fc)['acint']*1000);vpo.append(fit(merge([sub,sn],weights),fc)['acint']*1000)
   for omit in range(len(new)):
    sub=merge([s for i,s in enumerate(new) if i!=omit])
    vn.append(fit(sub,fc)['acint']*1000);vpn.append(fit(merge([so,sub],weights),fc)['acint']*1000)
   row[f'old_block_{block}']={'old_se':jk(vo),'new_split_se':jk(vn),'pooled_se':float(np.hypot(jk(vpo),jk(vpn)))}
  vf=[fit(merge([s for j,s in enumerate(full) if j!=i]),fc)['acint']*1000 for i in range(len(full))]
  row['new40_se']=jk(vf);rows.append(row);print('JK',T,json.dumps(row),flush=True)
 out['temperatures'][str(T)]=rows
 (outdir/'grouped_errors.json').write_text(json.dumps(out,indent=2)+'\n')
out['finished_unix']=time.time()
(outdir/'grouped_errors.json').write_text(json.dumps(out,indent=2)+'\n')

