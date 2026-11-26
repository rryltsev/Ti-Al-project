"""Independent stdlib audit of the actual 150 input directories."""
import hashlib,json,re
from pathlib import Path
ROOT=Path('/home6/u3129/tial-testsets/test50-dft-20261021-01')
C=json.loads((ROOT/'protocol.json').read_text());DATA=Path(C['dataset_root'])
M=json.loads((DATA/'manifest.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tags(p):
    return {k.strip().upper():v.strip() for line in p.read_text().splitlines() for part in line.split('#')[0].split(';') if '=' in part for k,v in [part.split('=',1)]}
records=[]
for row in M:
    sid=row['structure_id'];hashes=[]
    for xc in ['PBE','PBEsol','r2SCAN']:
        d=ROOT/'calculations'/xc/sid;t=tags(d/'INCAR');rec=json.loads((d/'input-manifest.json').read_text())
        assert all(sha(d/n)==v for n,v in rec['input_sha256'].items())
        assert sha(d/'POSCAR')==row['poscar_sha256']==sha(DATA/row['poscar'])
        assert t['NSW']=='0' and t['IBRION']=='-1' and t['ENCUT']=='550'
        assert t['ISIF']=='2' and t['ISPIN']=='1' and t['LASPH']=='.TRUE.'
        assert float(t['EDIFF'])==1e-6 and t['ISMEAR']=='1' and float(t['SIGMA'])==.1
        if xc=='r2SCAN':assert t['METAGGA']=='R2SCAN' and 'GGA' not in t
        else:assert t['GGA']==('PE' if xc=='PBE' else 'PS') and 'METAGGA' not in t
        hashes.append({n:sha(d/n) for n in ['POSCAR','KPOINTS','POTCAR']})
    assert hashes[0]==hashes[1]==hashes[2]
    records.append({'structure_id':sid,'identical_inputs':hashes[0]})
assert len(records)==50 and len(list((ROOT/'calculations').glob('*/*/INCAR')))==150
result={'status':'PASS','structures':50,'static_calculations':150,'all_three_geometry_hashes_equal_to_frozen_dataset':True,'all_three_KPOINTS_and_POTCAR_identical_per_structure':True,'NSW':0,'IBRION':-1,'ENCUT_eV':550,'records':records}
(ROOT/'independent-input-audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
