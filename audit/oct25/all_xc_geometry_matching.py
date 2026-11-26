"""Match training geometries across functional exports allowing small rounding differences."""
from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'all_xc_audit';DEST=OUT/'analysis'
sys.path.insert(0,str(ROOT/'audit/oct20/deps'))
import numpy as np
read=lambda p:json.loads(Path(p).read_text())
cfg=read(OUT/'protocol.json');datasets={}
for xc,ds in cfg['datasets'].items():
    a=np.load(OUT/ds['geometry_file']);rr=read(OUT/ds['rows_file']);d=[]
    for r in rr:
        if r['split']=='test':continue
        i=r['i'];p=a[f'coords_{i}'];typ=a[f'types_{i}'];cell=a[f'cell_{i}'];frac=p@np.linalg.inv(cell);frac%=1
        frac[np.isclose(frac,1,atol=1e-6)]=0
        # Rounded ordering removes irrelevant near-equal coordinate sorting noise.
        q=np.round(frac,5);order=np.lexsort((q[:,2],q[:,1],q[:,0],typ))
        d.append({'row':r,'cell':cell,'frac':frac[order],'types':typ[order],
                  'group':(len(p),int(typ.sum()))})
    datasets[xc]=d
result={}
for xc in ['PBEsol','r2SCAN']:
    pool={}
    for d in datasets[xc]:pool.setdefault(d['group'],[]).append(d)
    matches=[];unmatched=[];ambiguous=[]
    for a in datasets['PBE']:
        candidates=[]
        for b in pool.get(a['group'],[]):
            if np.max(abs(a['cell']-b['cell']))>1e-4:continue
            f=a['frac']-b['frac'];f-=np.rint(f);err=np.max(abs(f@a['cell']))
            if err<1e-4 and np.array_equal(a['types'],b['types']):candidates.append((b,float(err)))
        if len(candidates)==1:
            b,err=candidates[0];matches.append({'PBE_id':a['row']['id'],'other_id':b['row']['id'],'PBE_split':a['row']['split'],
                 'other_split':b['row']['split'],'max_coordinate_difference_A':err})
        elif not candidates:unmatched.append(a['row']['id'])
        else:ambiguous.append(a['row']['id'])
    result[xc]={'matched':len(matches),'same_split':sum(r['PBE_split']==r['other_split'] for r in matches),
                'different_split':sum(r['PBE_split']!=r['other_split'] for r in matches),
                'unmatched_PBE':unmatched,'ambiguous_PBE':ambiguous,'matches':matches,
                'scope':'Approximate matching, 1e-4 Å cell/coordinate tolerance, atom-order canonicalization. Unmatched entries are not proof of different physical configurations.'}
(DEST/'cross_functional_geometry_matching.json').write_text(json.dumps(result,indent=2))
print(json.dumps({x:{k:v[k] for k in ['matched','same_split','different_split','unmatched_PBE','ambiguous_PBE']} for x,v in result.items()},indent=2))
