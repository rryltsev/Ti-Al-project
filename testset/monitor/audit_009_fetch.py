"""Read-only retrieval of original Test009 outputs for an independent audit."""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import sys
import time

HERE = Path(__file__).resolve().parent
DEST = HERE / 'audits' / 'TiAl-test-009-20261023'
sys.path.insert(0, str(HERE.parent / 'dft'))
import cluster_test50 as cluster

REMOTE_CODE = r'''
import base64,hashlib,json,pathlib,time
root=pathlib.Path('/home6/u3129/tial-testsets/test50-dft-20261021-01')
files={}
names=['POSCAR','CONTCAR','INCAR','KPOINTS','OSZICAR','OUTCAR','vasprun.xml','vasp.stdout','label.json','input-manifest.json','output-manifest.json','execution-intent.json','status.json']
for xc in ['PBE','PBEsol','r2SCAN']:
    d=root/'calculations'/xc/'TiAl-test-009'
    for name in names:
        p=d/name
        data=p.read_bytes()
        files[xc+'/'+name]={'sha256':hashlib.sha256(data).hexdigest(),'base64':base64.b64encode(data).decode()}
    data=(d/'POTCAR').read_bytes()
    files[xc+'/POTCAR-metadata.json']={'text':json.dumps({'sha256':hashlib.sha256(data).hexdigest(),'headers':[line.strip() for line in data.decode().splitlines() if 'TITEL' in line or 'LEXCH' in line or 'ZVAL' in line]})}
for d in (root/'calculations'/'PBE').iterdir():
    if (d/'label.json').is_file() and (d/'status.json').is_file():
        if json.loads((d/'status.json').read_text())['state']=='validated':
            data=(d/'label.json').read_bytes()
            files['PBE-labels/'+d.name+'.json']={'sha256':hashlib.sha256(data).hexdigest(),'base64':base64.b64encode(data).decode()}
print(json.dumps({'unix':time.time(),'remote_root':str(root),'files':files}))
'''

if __name__ == '__main__':
    DEST.mkdir(parents=True, exist_ok=True)
    bundle = json.loads(cluster.cluster.run('python3 -c ' + shlex.quote(REMOTE_CODE)))
    manifest = {'unix': time.time(), 'remote_unix': bundle['unix'], 'remote_root': bundle['remote_root'], 'files': {}}
    for name, record in bundle['files'].items():
        data = base64.b64decode(record['base64']) if 'base64' in record else record['text'].encode()
        digest = hashlib.sha256(data).hexdigest()
        if 'sha256' in record:
            assert digest == record['sha256'], name
        path = DEST / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            assert path.read_bytes() == data, 'Audit file changed: ' + name
        else:
            path.write_bytes(data)
        manifest['files'][name] = digest
    (DEST / 'retrieval.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'retrieved_files': len(manifest['files']), 'directory': str(DEST)}))
