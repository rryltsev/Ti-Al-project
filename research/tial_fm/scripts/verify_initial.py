"""Verify recorded source identity, scientific accounting and design gates."""
from pathlib import Path
import hashlib, json

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/initial-audit-20261106-01'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

source=read(RUN/'source_hashes.json');changed=[]
for name,record in source.items():
    p=Path(name)
    if not p.exists() or digest(p)!=record['sha256']:changed.append(name)
assert not changed,changed
for p in (RUN/'snapshot').rglob('*'):
    if p.is_file():assert digest(p) in {v['sha256'] for v in source.values()},str(p)
records=[json.loads(x) for x in (RUN/'data_manifest.jsonl').read_text(encoding='utf8').splitlines()]
assert len(records)==1000 and len({r['geometry_id'] for r in records})==1000
assert sum(len(r['labels']) for r in records)==2990
assert sum(len(r['labels'])==3 for r in records)==992
splits=read(RUN/'splits.json');assert splits['new_grouped_benchmark']['state']=='blocked'
archived_matrix=ROOT/'protocol/experiment_matrix_initial_20261106.yaml'
matrix=read(archived_matrix if archived_matrix.exists() else ROOT/'protocol/experiment_matrix.yaml')
assert len(matrix['training_designs'])==16 and matrix['new_DFT_allowed'] is False
assert all(not r['launch_ready'] and r['split_hash'] is None for r in matrix['training_designs'])
for p in [ROOT/'runs/backend-source-20261106/source_manifest.json']:
    for r in read(p):assert digest(ROOT.parents[1]/r['local_path'])==r['sha256']
result={'status':'PASS','source_files_rehashed':len(source),'changed_source_files':changed,'selected_geometry_ids':1000,'geometry_functional_pairs':2990,'common_triples':992,'historical_splits_preserved':True,'new_split_released':False,'blocked_training_designs':16,'new_DFT_jobs':0,'checks':['source hashes and real snapshot bytes','geometry and label accounting','historical split and training gate states','approved backend source copy hashes']}
(ROOT/'reports/verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result,indent=2))
