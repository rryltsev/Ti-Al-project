"""Check exact geometry overlap, functional coverage and shared hyperparameters."""
from pathlib import Path
import sys,json,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'all_xc_audit';DEST=OUT/'analysis'
sys.path.insert(0,str(ROOT/'audit/oct20/deps'))
import numpy as np
read=lambda p:json.loads(Path(p).read_text())
cfg=read(OUT/'protocol.json');sets={};details={}
for xc,ds in cfg['datasets'].items():
    data=np.load(OUT/ds['geometry_file']);rows=read(OUT/ds['rows_file']);by_split={s:{} for s in ['train','validation','test']}
    for r in rows:
        i=r['i'];h=hashlib.sha256()
        for f in ['coords','cell']:
            a=np.round(data[f'{f}_{i}'],8).astype('<f8');a[a==0]=0;h.update(a.tobytes())
        h.update(data[f'types_{i}'].astype('<i4').tobytes());by_split[r['split']].setdefault(h.hexdigest(),[]).append(r['id'])
    details[xc]={'counts':ds['counts'],'unique_geometry_counts':{s:len(v) for s,v in by_split.items()},
      'exact_overlap_train_validation':sorted(set(by_split['train'])&set(by_split['validation'])),
      'exact_overlap_test_training':sorted(set(by_split['test'])&(set(by_split['train'])|set(by_split['validation']))),
      'coverage':{s:{'x_Ti_range':[min(r['x_Ti'] for r in rows if r['split']==s),max(r['x_Ti'] for r in rows if r['split']==s)],
       'V_A3_atom_range':[min(r['volume_A3_atom'] for r in rows if r['split']==s),max(r['volume_A3_atom'] for r in rows if r['split']==s)]} for s in by_split}}
    sets[xc]=by_split
common={s:len(set.intersection(*(set(sets[x][s]) for x in sets))) for s in ['train','validation','test']}
settings=read(DEST/'training_settings.json');shared={}
for f in ['learning_rate','loss','optimizer','training','training_data','validation_data','validating','model_fields']:
    shared[f]=all(settings[x][f]==settings['PBE'][f] for x in settings)
pairwise={}
for a,b in [('PBE','PBEsol'),('PBE','r2SCAN'),('PBEsol','r2SCAN')]:
    aa={h:s for s in ['train','validation'] for h in sets[a][s]};bb={h:s for s in ['train','validation'] for h in sets[b][s]}
    same=set(aa)&set(bb)
    pairwise[a+'/'+b]={'identical_geometries':len(same),'different_split':sum(aa[h]!=bb[h] for h in same)}
out={'functionals':details,'exact_geometry_common_counts':common,'pairwise_exact_geometry_matching':pairwise,'shared_settings':shared,
     'overlap_scope':'Geometry fingerprint rounds coordinates/cell to 1e-8 Å and preserves atom ordering. Absence of identical geometry does not establish trajectory-level independence.'}
(DEST/'input_integrity.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
