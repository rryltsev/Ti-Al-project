"""Independent export/provenance checks and matching regression checks."""
from audit_sources import BASE, GEN, SOLID, ROOT, CACHE, Index, same
from select_testset import aligned_rmsd, QUOTAS
from collections import Counter
import hashlib, io, json, pickle, re, zipfile
import numpy as np
from ase.io import read

def main():
    out=ROOT/'TiAl-test50-v1'
    manifest=json.loads((out/'manifest.json').read_text())
    data=pickle.load((CACHE/'reference.pkl').open('rb'));index=Index(data['refs'])
    frames=read(out/'test50.extxyz',':')
    assert len(frames)==len(manifest)==50
    assert dict(Counter(r['family'] for r in manifest))==QUOTAS
    assert len({r['trajectory'] for r in manifest if r['family']!='liquid'})==25
    regression=[]
    for a in (data['refs'][0],frames[0],frames[-1]):
        b=a[np.arange(len(a))[::-1]];b.positions+=np.array([.231,-.488,.719]);b.wrap()
        assert same(a,b),'Matching must tolerate permutation/translation/PBC'
        b=a.copy();b.positions[0]+=[.02,0,0]
        assert not same(a,b),'Matching must reject displaced configurations'
        b=a.copy();b.cell[0,0]+=.01
        assert not same(a,b),'Matching must reject changed cells'
        regression.append(True)
    with zipfile.ZipFile(GEN/'AlTi-liq_MACE-dataset_ASAP-select-200_POSCARS.zip') as z:
        xname=next(n for n in z.namelist() if n.endswith('.xyz'))
        liquid_xyz=read(io.StringIO(z.read(xname).decode()),':',format='extxyz')
        assert len(liquid_xyz)==200
        assert all(same(a,r['atoms']) for a,r in zip(liquid_xyz,data['liquids']))
    provenance=[]
    for i,(a,r) in enumerate(zip(frames,manifest)):
        p=out/r['poscar'];b=read(p,format='vasp')
        assert hashlib.sha256(p.read_bytes()).hexdigest()==r['poscar_sha256']
        assert same(a,b) and not index.matches(b)
        assert a.calc is None and set(a.arrays)=={'numbers','positions'}
        assert len(a)==r['n_atoms'] and sum(a.numbers==13)==r['n_Al']
        assert not any(same(a,x) for x in frames[:i])
        if r['family']=='liquid':
            src=liquid_xyz[r['frame']]
            assert same(a,src)
            provenance.append({'structure_id':r['structure_id'],'source':'archive XYZ and POSCAR agree','kinetic_temperature_K':float(src.get_temperature()),'original_trajectory_time_unknown':True})
        else:
            p=BASE/r['trajectory'];stat=np.loadtxt(p.with_name('stat.csv'),delimiter=';')
            with p.open(encoding='utf8') as f:
                for frame in range(r['frame']+1):
                    n=int(f.readline());header=f.readline();lines=[f.readline() for _ in range(n)]
            src=read(io.StringIO(str(n)+'\n'+header+''.join(lines)),format='extxyz')
            assert same(a,src)
            step=r['md_step'];row=stat[np.where(stat[:,0]==step)[0][0]]
            energy=float(re.search(r'(?:^|\s)energy=([^\s]+)',header)[1])
            assert abs(energy/len(a)-row[5])<1e-6,'Incorrect XYZ/stat alignment'
            assert abs(src.get_temperature()-row[6])<1e-3
            assert r['time_ps']>=1.0 and (r['training_gap_ps'] is None or r['training_gap_ps']>=.5)
            provenance.append({'structure_id':r['structure_id'],'source_geometry_verified':True,'step':step,'kinetic_temperature_K':float(src.get_temperature()),'energy_and_temperature_alignment_verified':True})
    failed=[]
    for p in sorted((SOLID/'POSCAR_AlTi3_D0_19').glob('*/slurm-*.out')):
        s=p.read_text(encoding='utf8',errors='replace')
        assert 'upper triangular matrix' in s
        failed.append(p.relative_to(BASE).as_posix())
    report={'status':'PASS','n_structures':50,'exact_reference_overlaps':0,'internal_exact_duplicates':0,'matching_regression_cases_passed':len(regression),'poscar_extxyz_roundtrips_passed':50,'source_provenance_passed':50,'labels_removed':True,'liquid_archive_xyz_poscar_pairs_passed':200,'D019_failed_logs':failed,'source_checks':provenance}
    (out/'validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('D019_failed_logs','source_checks')},indent=2))

if __name__=='__main__':main()
