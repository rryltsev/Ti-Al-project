"""Inventory, periodic geometry audit and trajectory provenance; no labels used."""
from pathlib import Path
from collections import Counter, defaultdict
import io, json, re, zipfile, pickle, hashlib, sys
import numpy as np
from scipy.spatial import cKDTree
from ase import Atoms
from ase.io import read

ROOT = Path(__file__).resolve().parent
import os
BASE = Path(os.environ.get('TIAL_SOURCE_ROOT',r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project'))
GEN = BASE / 'MACE-dataset-gen'
SOLID = GEN / 'MD-Solid-Solutions/Ti-Al_SS/RESULT'
CACHE = ROOT / 'cache'
CACHE.mkdir(exist_ok=True)

def xyz(path):
    f = path.open(encoding='utf8') if isinstance(path, Path) else path
    try:
        frame = 0
        while line := f.readline():
            if not line.strip(): continue
            n = int(line); header = f.readline()
            cell = np.fromstring(re.search(r'Lattice="([^"]+)"', header)[1], sep=' ').reshape(3,3)
            lines = [f.readline().split() for _ in range(n)]
            a = Atoms([x[0] for x in lines], positions=np.array([x[1:4] for x in lines], float), cell=cell, pbc=True)
            yield frame, a
            frame += 1
    finally:
        f.close()

def composition(a):
    return len(a), int(np.count_nonzero(a.numbers == 22))

def same(a,b,tol=5e-5):
    if composition(a)!=composition(b) or np.max(abs(a.cell.array-b.cell.array))>tol: return False
    # Cell axes are preserved throughout the source export pipeline. PBC and
    # species permutations and a common translation are explicitly handled.
    inv = np.linalg.inv(b.cell.array)
    pa, pb = a.positions @ inv, b.positions @ inv
    def check(shift):
        for z in (13,22):
            x,y=pa[a.numbers==z]+shift,pb[b.numbers==z]
            if not len(x): continue
            d=x[:,None,:]-y[None,:,:]; d-=np.rint(d)
            dist=np.linalg.norm(d @ b.cell.array,axis=2)
            nearest=dist.argmin(axis=1)
            if len(set(nearest))!=len(x) or np.max(dist[np.arange(len(x)),nearest])>tol: return False
        return True
    if check(np.zeros(3)):return True
    z=min(set(a.numbers),key=lambda v:np.count_nonzero(a.numbers==v))
    anchor=pa[np.where(a.numbers==z)[0][0]]
    return any(check(target-anchor) for target in pb[b.numbers==z])

class Index:
    def __init__(self, frames):
        self.frames=frames; self.groups=defaultdict(list)
        for i,a in enumerate(frames):self.groups[composition(a)].append(i)
        self.trees={k:cKDTree([frames[i].cell.array.ravel() for i in ids]) for k,ids in self.groups.items()}
    def matches(self,a):
        k=composition(a)
        if k not in self.groups:return []
        ids=self.trees[k].query_ball_point(a.cell.array.ravel(),5e-5,p=np.inf)
        return [self.groups[k][j] for j in ids if same(a,self.frames[self.groups[k][j]])]

def main():
    if '--scan-only' in sys.argv:
        data=pickle.load((CACHE/'reference.pkl').open('rb'))
        return scan_trajectories(Index(data['refs']))
    refs=[]; origins=[]; inventory=[]
    def add_file(p, frames):
        n=0
        for i,a in frames:
            refs.append(a);origins.append({'file':p.relative_to(BASE).as_posix(),'frame':i}); n+=1
        inventory.append({'file':p.relative_to(BASE).as_posix(),'frames':n,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    add_file(GEN/'MACE-database/TiAl-liq-relax-ss_select-1000.xyz', xyz(GEN/'MACE-database/TiAl-liq-relax-ss_select-1000.xyz'))
    for p in sorted((BASE/'DPA4-finetune').rglob('*.xyz')): add_file(p,xyz(p))
    for top in ('DPA4-finetune','VASP-output'):
        for p in sorted((BASE/top).rglob('coord.npy')):
            system=p.parent.parent
            typ=np.loadtxt(system/'type.raw',dtype=int,ndmin=1)
            names=(system/'type_map.raw').read_text().split()
            syms=[names[t] for t in typ]
            cs=np.load(p).reshape(-1,len(syms),3);bs=np.load(p.with_name('box.npy')).reshape(-1,3,3)
            add_file(p,((i,Atoms(syms,positions=c,cell=b,pbc=True)) for i,(c,b) in enumerate(zip(cs,bs))))
    for p in sorted((BASE/'VASP-output').rglob('POSCAR*')):
        if p.is_file():add_file(p,[(0,read(p,format='vasp'))])
    print('reference records',len(refs),flush=True)
    # Deduplicate reference records using tolerant matching, preserving all origins.
    unique=[]; sources=[]; buckets=defaultdict(list)
    for a,origin in zip(refs,origins):
        k=composition(a);found=None
        for j in buckets[k]:
            if same(a,unique[j]):found=j;break
        if found is None:
            found=len(unique);unique.append(a);sources.append([]);buckets[k].append(found)
        sources[found].append(origin)
    idx=Index(unique)
    print('unique reference geometries',len(unique),flush=True)
    zpath=GEN/'AlTi-liq_MACE-dataset_ASAP-select-200_POSCARS.zip'
    with zipfile.ZipFile(zpath) as z:
        liquids=[]
        for name in sorted((n for n in z.namelist() if '/POSCAR.' in n),key=lambda n:int(n.rsplit('.',1)[1])):
            a=read(io.StringIO(z.read(name).decode()),format='vasp')
            liquids.append({'atoms':a,'member':name,'matches':idx.matches(a)})
    ss200=[{'frame':i,'matches':idx.matches(a)} for i,a in xyz(SOLID/'TiAl-SS-mace_select-200.xyz')]
    print('liquid overlap',sum(bool(r['matches']) for r in liquids),'solid200 overlap',sum(bool(r['matches']) for r in ss200),flush=True)
    pickle.dump({'refs':unique,'origins':sources,'liquids':liquids},(CACHE/'reference.pkl').open('wb'))
    summary={'reference_records':len(refs),'unique_reference_geometries':len(unique),'reference_files':inventory,
             'liquid_archive_sha256':hashlib.sha256(zpath.read_bytes()).hexdigest(),
             'liquid_overlap':[{'member':r['member'],'refs':r['matches']} for r in liquids], 'solid200_overlap':ss200,
             'matching_tolerance_A':5e-5,'matching_scope':'Same cell axes; periodic images, atom permutations and common translations. No arbitrary lattice-basis transformations.'}
    (ROOT/'source_audit.json').write_text(json.dumps(summary,indent=2))
    scan_trajectories(idx)

def scan_trajectories(idx):
    candidates=[];traj_reports=[];ref_provenance=defaultdict(list)
    for p in sorted(SOLID.glob('*/*/xyz.xyz')):
        hits=[];sample=[];nframes=0
        stats=np.loadtxt(p.with_name('stat.csv'),delimiter=';')
        # Determine whether XYZ includes initial step 0 from frame counts.
        for i,a in xyz(p):
            matches=idx.matches(a);nframes+=1
            if matches:
                hits.append(i)
                for r in matches:ref_provenance[r].append({'trajectory':p.relative_to(BASE).as_posix(),'frame':i})
            if i>=100 and i%25==0:
                sample.append({'atoms':a,'frame':i,'trajectory':p.relative_to(BASE).as_posix(),'family':p.parent.parent.name,'composition_dir':p.parent.name})
        # Verify frame/time mapping against output count and source parameters.
        script=p.with_name('mace_md_npt.py').read_text(encoding='utf8')
        assert re.search(r'time_step\s*=\s*1\.0\s',script) and re.search(r'num_interval\s*=\s*10\s',script)
        assert nframes in (len(stats),len(stats)+1) and np.array_equal(stats[:,0],np.arange(1,len(stats)+1)*10),(str(p),nframes,stats.shape)
        initial_frame=int(nframes==len(stats)+1)
        for r in sample:
            gap=min((abs(r['frame']-h)*.01 for h in hits),default=float('inf'))
            if gap>=0.5:
                r['training_gap_ps']=None if np.isinf(gap) else gap
                si=r['frame']-initial_frame
                r['md_step']=int(stats[si,0]);r['time_ps']=float(stats[si,0]*.001);r['temperature_K']=float(stats[si,6]);candidates.append(r)
        traj_reports.append({'file':p.relative_to(BASE).as_posix(),'frames':nframes,'includes_initial_step0':bool(initial_frame),'reference_frames':hits,'candidate_count':sum(r['trajectory']==p.relative_to(BASE).as_posix() for r in candidates),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
        print(p.parent.parent.name,p.parent.name,nframes,'reference',len(hits),'candidates',traj_reports[-1]['candidate_count'],flush=True)
    pickle.dump(candidates,(CACHE/'solid_candidates.pkl').open('wb'))
    (ROOT/'trajectory_audit.json').write_text(json.dumps({'trajectories':traj_reports,'reference_provenance':ref_provenance,'equilibration_exclusion_ps':1.0,'minimum_training_gap_ps':0.5,'candidate_stride_ps':0.25},indent=2))
    print('solid candidates',len(candidates),flush=True)

if __name__=='__main__':main()
