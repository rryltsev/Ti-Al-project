"""Freeze geometry-only inference input and separate local DFT labels for PBE sweep."""
from pathlib import Path
import hashlib,json,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'pbe_checkpoint_sweep'
sys.path.insert(0,str(ROOT/'audit/oct20/deps'))
import numpy as np
SOURCE=Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project')
RUN=SOURCE/'DPA4-checkpoints/TiAl_tune_pbe_Neo_DPA4_shard_20260920_140047'
BASE=SOURCE/'DPA4-finetune/TiAl_PBE_upd/npy_format'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
OUT.mkdir(exist_ok=True)
assert not (OUT/'input-manifest.json').exists(),'Frozen input exists; do not regenerate'
models={str(s):{'local_path':str(RUN/f'output_model/ckpt/model.ckpt-{s}.pt'),
               'sha256':sha(RUN/f'output_model/ckpt/model.ckpt-{s}.pt')} for s in [5000,10000,15000,20000,23400]}
assert models['5000']['sha256']==read(ROOT/'testset/monitor/model_registry.json')['models']['PBE']['checkpoint_sha256']
geo={};labels={};rows=[];sources={}
def add(coords,cell,types,energy,forces,virial,meta):
    i=len(rows);n=len(types)
    geo[f'coords_{i}']=np.array(coords,dtype='<f8').reshape(n,3)
    geo[f'cell_{i}']=np.array(cell,dtype='<f8').reshape(3,3)
    geo[f'types_{i}']=np.array(types,dtype='<i4')
    for k,v in [('energy',float(energy)),('forces',np.array(forces).reshape(n,3)),('virial',np.array(virial).reshape(3,3))]:labels[f'{k}_{i}']=v
    rows.append({**meta,'i':i,'n_atoms':n,'x_Ti':float(sum(t==1 for t in types)/n),
                 'volume_A3_atom':float(abs(np.linalg.det(geo[f'cell_{i}']))/n)})
for split,listing in [('train','training_systems.json'),('validation','validation_systems.json')]:
    p=BASE/listing;sources[str(p)]=sha(p)
    systems=read(p)
    for rel in sorted(s.split('/npy_format/',1)[1] for s in systems):
        folder=BASE/rel;types=np.loadtxt(folder/'type.raw',dtype=int,ndmin=1)
        names=(folder/'type_map.raw').read_text().split();types=[['Al','Ti'].index(names[t]) for t in types];n=len(types)
        for p in [folder/'type.raw',folder/'type_map.raw']:sources[str(p)]=sha(p)
        for sd in sorted(folder.glob('set.*')):
            arrays={}
            for k in ['coord','box','energy','force','virial']:
                p=sd/(k+'.npy');sources[str(p)]=sha(p);arrays[k]=np.load(p,allow_pickle=False)
            for j,e in enumerate(arrays['energy'].reshape(-1)):
                add(arrays['coord'][j],arrays['box'][j],types,e,arrays['force'][j],arrays['virial'][j],
                    {'split':split,'id':f'{split}/{folder.name}/{sd.name}/{j}','system':rel,'set':sd.name,'frame':j})
bundle=read(HERE/'test20_bundle.json')
for r in bundle['records']:
    cell=np.array(r['cell_A']);coords=np.array(r['fractional_positions'])@cell;lab=r['labels']['PBE']
    add(coords,cell,[['Al','Ti'].index(s) for s in r['symbols']],lab['energy_eV'],lab['forces_eV_A'],
        -np.array(lab['stress_eV_A3'])*lab['volume_A3'],{'split':'test20','id':r['metadata']['structure_id']})
counts={s:sum(r['split']==s for r in rows) for s in ['train','validation','test20']}
assert counts=={'train':850,'validation':149,'test20':20},counts
assert all(np.isfinite(a).all() for a in geo.values()) and all(np.isfinite(a).all() for a in labels.values())
# Verify exact original train energies against the separately audited frozen training cache.
trainref={(r['system'],r['set'],r['frame']):r for r in read(ROOT/'testset/monitor/alignment/train-offsets-v1/training-labels.json') if r['functional']=='PBE'}
for r in rows:
    if r['split']=='train': assert labels[f"energy_{r['i']}"]==trainref[(r['system'],r['set'],r['frame'])]['energy_eV']
np.savez_compressed(OUT/'geometries.npz',**geo);np.savez_compressed(OUT/'labels-local-only.npz',**labels)
(OUT/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
manifest={'models':models,'counts':counts,'total':len(rows),'geometry_sha256':sha(OUT/'geometries.npz'),
          'labels_sha256':sha(OUT/'labels-local-only.npz'),'rows_sha256':sha(OUT/'rows.json'),
          'source_files_sha256':sources,'test_bundle_sha256':sha(HERE/'test20_bundle.json'),
          'policy':'Compare all five supplied checkpoints; no training or energy shifts. Article model/benchmark caches remain unchanged. Geometry inputs only sent for inference.'}
(OUT/'input-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'counts':counts,'models':models,'geometry_bytes':(OUT/'geometries.npz').stat().st_size},indent=2))
