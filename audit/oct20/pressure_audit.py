from pathlib import Path
import json, re, hashlib, collections
import numpy as np
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project')
OUT=Path(__file__).parent
inv=json.loads((OUT/'inventory.json').read_text())
fs=[f for f in inv if '/03_stress_autocorr/' in f['path'] and Path(f['path']).name.startswith('pressure_')]
def one(f):
    p=ROOT/f['path']; raw=p.read_bytes(); a=np.loadtxt(raw.decode().splitlines())
    mol=(p.parent/'mol.data').read_text()
    N=int(re.search(r'(\d+) atoms',mol)[1])
    bounds=[tuple(map(float,re.search(r'([-+\d.eE]+)\s+([-+\d.eE]+)\s+'+label,mol).groups())) for label in ['xlo xhi','ylo yhi','zlo zhi']]
    V=np.prod([hi-lo for lo,hi in bounds])
    stem=f['path'].split('/AlTi/')[1].split('/')
    T=float(stem[0][:-1]);m=stem[1];c=float(stem[3].split('_')[-1]); host='cluster' if 'models_check_cluster' in f['path'] else 'wsl2'
    pressure=np.mean(a[:,1:4],axis=1)/10000
    aniso=np.array([(a[:,1]-.5*a[:,2]-.5*a[:,3])/np.sqrt(3),.5*(a[:,2]-a[:,3]),a[:,4],a[:,5],a[:,6]])
    return dict(T=T,dft=m,c_Ti=c,host=host,replica=p.parent.name,source=f['path'],N=N,V_A3=float(V),density_nominal_g_cm3=float(N*((1-c)*26.98+c*47.88)/V*1.66053906660),steps=len(a),step_stride=list(np.unique(np.diff(a[:,0]))),P_mean_GPa=float(pressure.mean()),P_first_half_GPa=float(pressure[:len(a)//2].mean()),P_second_half_GPa=float(pressure[len(a)//2:].mean()),pressure_sha256=hashlib.sha256(raw).hexdigest(),mol_sha256=hashlib.sha256(mol.encode()).hexdigest(),aniso_means_bar=aniso.mean(axis=1).tolist(),aniso_std_bar=aniso.std(axis=1).tolist())
with ThreadPoolExecutor(max_workers=6) as pool: rows=list(pool.map(one,fs))
(OUT/'pressure_replicas.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
groups=collections.defaultdict(list)
for r in rows:groups[(r['T'],r['dft'],r['c_Ti'],r['host'])].append(r)
summary=[]
for key,rs in sorted(groups.items()):
    p=np.array([r['P_mean_GPa'] for r in rs]);rho=np.array([r['density_nominal_g_cm3'] for r in rs]);
    d=dict(T=key[0],dft=key[1],c_Ti=key[2],host=key[3],nrep=len(rs),N=sorted({r['N'] for r in rs}),P_GPa=float(p.mean()),P_replicate_std_GPa=float(p.std(ddof=1)),P_range_GPa=[float(p.min()),float(p.max())],rho_g_cm3=float(rho.mean()),rho_replicate_std_g_cm3=float(rho.std(ddof=1)),steps=sorted({r['steps'] for r in rs}))
    summary.append(d)
    print(f"{key} n={len(rs)} N={d['N']} P={p.mean():.3f} GPa range[{p.min():.3f},{p.max():.3f}] rho={rho.mean():.4f} +/- {rho.std(ddof=1):.4f}")
print('DUPLICATE PRESSURE HASHES',[n for n in collections.Counter(r['pressure_sha256'] for r in rows).values() if n>1])
(OUT/'pressure_cases.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
