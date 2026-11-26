"""Fail before optimization if the staged training data/configuration changed."""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    work=Path.cwd();c=json.loads((work/'case.json').read_text());cfg=json.loads((work/'input.json').read_text())
    assert sha(work/'input.json')==c['configuration_sha256']
    assert sha(c['source']['path'])==c['source']['sha256']
    manifest=ROOT/c['dataset_manifest'];assert sha(manifest)==c['dataset_manifest_sha256']
    hashes=json.loads(manifest.read_text())
    for name,digest in hashes.items():assert sha(ROOT/name)==digest,name
    assert cfg['training']['seed']==c['seed'] and cfg['training']['num_epochs']==50
    assert cfg['training']['stat_file_mode']=='update'
    assert not (work/cfg['training']['stat_file']).exists(), 'Statistics must be computed anew in this case.'
    assert not any('test' in s.split('/') for k in ['training_data','validation_data'] for s in cfg['training'][k]['systems'])
    for part,key,expected in [('train','training_data',c['N']),('validation','validation_data',149)]:
        ids=[]
        for system in cfg['training'][key]['systems']:
            p=Path(system);assert p.parent==Path(c['data_root'])/c['XC']/part
            ids+=json.loads((p/'geometry_ids.json').read_text())
        assert len(ids)==len(set(ids))==expected
        wanted=json.loads((ROOT/'subsets.json').read_text())['subsets'][str(c['N'])] if part=='train' else json.loads((ROOT/'splits.json').read_text())['split_ids']['validation']
        assert set(ids)==set(wanted)
    (work/'input-verification.json').write_text(json.dumps(dict(status='PASS',configuration_sha256=c['configuration_sha256'],dataset_manifest_sha256=c['dataset_manifest_sha256'],files_verified=len(hashes),N=c['N'],seed=c['seed'],test_access=False),indent=2)+'\n')
    print('INPUTS_VERIFIED',c['name'])

if __name__=='__main__':main()
