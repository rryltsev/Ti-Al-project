"""Final registry/config/backend consistency checks for this continuation."""
from pathlib import Path
import copy,hashlib,json,shlex
import cluster_pilot as c
from prepare_n694_repeats import ROOT,RUN
from package_grace_preparation import RUN as GRACE,REMOTE,ENV

def load(p):return json.loads(p.read_text(encoding='utf8'))
def main():
    state=load(ROOT/'state.json');matrix=load(ROOT/'protocol/experiment_matrix.yaml')
    designs=matrix['training_designs'];assert len({d['run_id'] for d in designs})==len(designs)==32
    dp=[d for d in designs if d['architecture']=='DPA4-Neo'];assert len(dp)==24
    assert sum(d['state']=='completed' for d in dp)==16
    new=[d for d in dp if d['train_size']==694 and d['training_seed'] in [20,30]];assert len(new)==8
    for d in new:
        p=Path(d['submitted_configuration']);assert p.is_file()
        assert hashlib.sha256(p.read_bytes()).hexdigest()==d['configuration_sha256']
        assert d['state'] in ['running','submitted'] and d['actual_cost']['GPU_hours'] is None
    for xc in ['PBE','r2SCAN']:
        a=load(GRACE/'package/draft-configs'/f'{xc}-full-N200-seed10/input.yaml')
        b=load(GRACE/'package/draft-configs'/f'{xc}-restricted-N200-seed10/input.yaml')
        mask=b['fit'].pop('trainable_variable_names');assert a==b
        assert mask==load(GRACE/'package/preparation-plan.json')['restricted_patterns']
    result=load(GRACE/'collected-preflight/summary.json');assert result['status']=='PASS' and result['training_steps']==0
    code='''from pathlib import Path
import hashlib,json,yaml
root=Path(REMOTE)
manifest=json.loads((root/'backend-manifest.json').read_text())
for rel,info in manifest['files'].items():
    assert hashlib.sha256((Path(manifest['root'])/rel).read_bytes()).hexdigest()==info['sha256'],rel
configs=list((root/'draft-configs').glob('*/input.yaml'));assert len(configs)==4
for p in configs:
    d=yaml.safe_load(p.read_text());assert d['seed']==10 and d['fit']['maxiter']==50
    assert Path(d['data']['filename']).is_file() and Path(d['data']['test_filename']).name=='validation.extxyz'
    assert Path(d['potential']['filename']).is_file()
    assert Path(d['potential']['checkpoint_name']+'.index').is_file()
print(json.dumps(dict(status='PASS',backend_source_files_verified=len(manifest['files']),native_YAML_configs_parsed=len(configs))))
'''.replace('REMOTE',repr(REMOTE))
    remote=json.loads(c.ssh(ENV+'/bin/python -c '+shlex.quote(code)))
    answer=dict(status='PASS',registry_total_designs=32,DPA4_completed=16,DPA4_new_submitted=8,GRACE_draft_pairs_equal_except_mask=True,GRACE_preflight='PASS',remote=remote)
    (RUN/'final-verification.json').write_text(json.dumps(answer,indent=2)+'\n')
    print(json.dumps(answer))

if __name__=='__main__':main()
