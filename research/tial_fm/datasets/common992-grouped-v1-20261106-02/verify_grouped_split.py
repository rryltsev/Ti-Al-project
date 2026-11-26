"""Independent checks of grouped IDs, source identity and every exported label."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PARTS=('train','validation','test');XCS=('PBE','PBEsol','r2SCAN')
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--version',default='common992-grouped-v1-20261106-02');args=ap.parse_args()
    out=ROOT/'datasets'/args.version;s=read(out/'splits.json');groups=read(out/'groups.json')
    rows=[json.loads(l) for l in (out/'data_manifest.jsonl').read_text(encoding='utf8').splitlines()]
    original=[json.loads(l) for l in (ROOT/'runs/initial-audit-20261106-01/data_manifest.jsonl').read_text(encoding='utf8').splitlines()]
    original_by_id={r['geometry_id']:r for r in original};by_id={r['geometry_id']:r for r in rows}
    assert len(by_id)==1000
    for r in rows:
        for key,val in original_by_id[r['geometry_id']].items():assert r[key]==val,(r['geometry_id'],key)
    paired={r['geometry_id'] for r in rows if len(r['labels'])==3};assert len(paired)==992
    assert set.union(*(set(s['split_ids'][p]) for p in PARTS))==paired
    for i,p in enumerate(PARTS):
        for q in PARTS[i+1:]:assert not set(s['split_ids'][p])&set(s['split_ids'][q])
    role_by_parent=defaultdict(set);role_by_composition=defaultdict(set);role_by_group=defaultdict(set)
    for r in rows:
        role_by_group[r['group_id']].add(r['split'])
        role_by_composition[(r['composition'].get('Ti',0),r['composition'].get('Al',0))].add(r['split'])
        for p in r['parent_ids']:role_by_parent[p].add(r['split'])
        if r['paired_benchmark_eligible']:assert r['geometry_id'] in s['split_ids'][r['split']]
    for collection in [role_by_parent,role_by_composition,role_by_group]:assert all(len(v)==1 for v in collection.values())
    for g in groups:
        assert all(by_id[i]['group_id']==g['group_id'] and by_id[i]['split']==g['split'] for i in g['paired_geometry_ids']+g['incomplete_geometry_ids'])
    srcmanifest=read(ROOT/'runs/initial-audit-20261106-01/source_hashes.json')
    for name,h in read(out/'input_hashes.json').items():assert sha(Path(name))==h,name
    source_hashes=read(out/'label_source_hashes.json')
    for name,h in source_hashes.items():
        assert sha(Path(name))==h,name
        assert srcmanifest[name]['sha256']==h,('source differs from initial snapshot',name)
    for name,r in read(out/'export_hashes.json').items():assert sha(out/name)==r['sha256'],name
    export_index=read(out/'export_index.json');assert len(export_index)==2976
    cache={};sourcecache={};seen=defaultdict(list)
    for e in export_index:
        r=by_id[e['geometry_id']];assert r['split']==e['split'] and r['group_id']==e['group_id']
        original_label=r['labels'][e['functional']]
        assert original_label['coord_path']==e['source_coord_path'] and original_label['frame_index']==e['source_frame_index']
        system=out/e['system'];source=Path(e['source_coord_path']);i=e['source_frame_index'];j=e['row'];n=r['natoms']
        if str(system) not in cache:
            cache[str(system)]={k:np.load(system/'set.000'/(k+'.npy'),allow_pickle=False) for k in ['coord','box','energy','force','virial']}
            assert (system/'type_map.raw').read_text().split()==['Al','Ti']
            types=np.loadtxt(system/'type.raw',dtype=int,ndmin=1)
            assert np.array_equal(types,np.array([0]*r['composition'].get('Al',0)+[1]*r['composition'].get('Ti',0)))
        if str(source) not in sourcecache:sourcecache[str(source)]={k:np.load(source.with_name(k+'.npy'),allow_pickle=False) for k in ['coord','box','energy','force','virial']}
        a=cache[str(system)];orig=sourcecache[str(source)];perm=np.array(e['source_permutation_to_Al_Ti'])
        assert np.array_equal(np.sort(perm),np.arange(n))
        source_system=source.parent.parent;mp=(source_system/'type_map.raw').read_text().split();typ=np.loadtxt(source_system/'type.raw',dtype=int,ndmin=1)
        assert [mp[t] for t in typ[perm]]==['Al']*r['composition'].get('Al',0)+['Ti']*r['composition'].get('Ti',0)
        for k in ['coord','force']:
            assert np.array_equal(a[k][j].reshape(n,3),orig[k][i].reshape(n,3)[perm]),(e,k)
        for k in ['energy','box','virial']:assert np.array_equal(a[k][j].ravel(),np.asarray(orig[k][i]).ravel()),(e,k)
        assert read(system/'geometry_ids.json')[j]==e['geometry_id']
        assert all(np.isfinite(v[j]).all() for v in a.values())
        seen[(e['functional'],e['split'])].append(e['geometry_id'])
    for xc in XCS:
        for p in PARTS:assert sorted(seen[xc,p])==s['split_ids'][p]
    result={'status':'PASS','version':args.version,'all_pool_geometries':1000,'paired_geometries':992,'paired_label_records_verified':2976,'incomplete_geometries_kept_separate':8,'counts':{p:len(s['split_ids'][p]) for p in PARTS},'known_trajectory_ids':len(role_by_parent),'known_trajectory_cross_split_overlap':0,'conservative_composition_groups':len(role_by_composition),'conservative_group_cross_split_overlap':0,'geometry_id_cross_split_overlap':0,'labels_and_native_geometry_unchanged_after_species_sort':True,'all_functionals_identical_geometry_ID_splits':True,'label_source_files_hash_verified':len(source_hashes),'source_files_match_initial_snapshot':True,'lost_trajectory_identity_recovered':False,'guarantee_scope':'Fixed-nTi/fixed-nAl trajectories and their relaxation descendants, per documented scripts','archival_test49':'Not part of this split; not an independent test for new model selection','new_model_training':False}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
