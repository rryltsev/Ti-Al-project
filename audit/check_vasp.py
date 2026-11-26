from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import re,json
B=Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project')
def check(f):
 s=f.read_text(errors='replace')
 def get(p):
  m=re.search(p,s);return m.group(1).strip() if m else None
 return {'path':str(f.relative_to(B)),'completed':'General timing and accounting' in s,'ediff_reached':'aborting loop because EDIFF is reached' in s,'cutoff':get(r'ENCUT\s*=\s*([\d.]+)'),'nkpts':get(r'NKPTS\s*=\s*(\d+)'),'kpoints_file':f.with_name('KPOINTS').is_file()}
out={}
for p in (B/'VASP-output').iterdir():
 with ThreadPoolExecutor(max_workers=8) as pool:r=list(pool.map(check,p.rglob('OUTCAR')))
 out[p.name]=r
 print(p.name,'runs',len(r),'completed',sum(x['completed'] for x in r),'ediff_reached',sum(x['ediff_reached'] for x in r),'cutoff',Counter(x['cutoff'] for x in r),'nkpts',Counter(x['nkpts'] for x in r),'KPOINTS',sum(x['kpoints_file'] for x in r),flush=True)
 print('flagged',[(x['path'],x['completed'],x['ediff_reached']) for x in r if not x['completed'] or not x['ediff_reached']],flush=True)
Path('audit/vasp_checks.json').write_text(json.dumps(out,indent=2))
