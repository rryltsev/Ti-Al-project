from pathlib import Path
from collections import Counter
import hashlib,json,re,time
ROOT=Path('/home6/u3129/tial-testsets/test50-dft-20261021-01')
reports=[]
for p in sorted((ROOT/'calculations').glob('*/*/status.json')):
    status=json.loads(p.read_text());d=p.parent;f=d/'OUTCAR';s=f.read_text(errors='replace') if f.exists() else ''
    entry=dict(status)
    entry['outcar_exists']=f.exists()
    entry['vasp_version']=s.splitlines()[0].strip() if s else None
    entry['settings']={}
    for key in ['NSW','IBRION','ENCUT','EDIFF','ISMEAR','SIGMA','ISPIN','METAGGA','GGA','NKPTS']:
        found=re.findall(r'\b'+key+r'\s*=\s*([^\s;#]+)',s)
        if found:entry['settings'][key]=found[-1]
    log=d/'OSZICAR';lines=log.read_text(errors='replace').splitlines() if log.exists() else []
    entry['last_oszicar_lines']=lines[-2:]
    entry['poscar_sha256']=hashlib.sha256((d/'POSCAR').read_bytes()).hexdigest()
    reports.append(entry)
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'workflow.py',ROOT/'production.sbatch',ROOT/'setup.sbatch',ROOT/'protocol.json']}
result={'unix':time.time(),'states':dict(Counter(r['state'] for r in reports)),'execution_code_sha256':hashes,'reports':reports}
(ROOT/'runtime-snapshot.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
