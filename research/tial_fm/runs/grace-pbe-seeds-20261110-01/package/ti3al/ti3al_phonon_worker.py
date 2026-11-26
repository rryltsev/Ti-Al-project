"""Archived-geometry Ti3Al preparation and matched DOS analysis, CPU Slurm only."""
from pathlib import Path
import copy,hashlib,json,os,sys
import numpy as np
import phonopy,spglib
ROOT=Path(__file__).resolve().parent
SCALES=[.5,1.,2.]
def dump(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_ph(scale=1.):
    p=phonopy.load(ROOT/'source/Ti3Al_D019/phonopy_disp.yaml',primitive_matrix='P',produce_fc=False,symmetrize_fc=False,is_compact_fc=False,lang='C')
    assert len(p.unitcell)==8 and len(p.primitive)==8 and len(p.supercell)==64
    ds=copy.deepcopy(p.dataset)
    assert len(ds['first_atoms'])==8
    for x in ds['first_atoms']:x['displacement']=np.array(x['displacement'])*scale
    p.dataset=ds
    return p
def geometry(cell):return dict(cell=cell.cell.tolist(),positions=cell.positions.tolist(),symbols=cell.symbols)
def spg(cell):return (np.array(cell['cell']),np.array(cell['positions'])@np.linalg.inv(cell['cell']),[13 if s=='Al' else 22 for s in cell['symbols']])
def prepare():
    out=ROOT/'prepared';out.mkdir(exist_ok=False)
    p=load_ph();src=json.loads((ROOT/'source/DFT_Ti_Ti3Al_reference.json').read_text())['systems'][1]
    assert np.allclose(p.unitcell.cell,src['unit_cell']['lattice'],atol=1e-12)
    assert np.allclose(p.primitive.masses,[x['mass'] for x in src['primitive_cell']['points']],atol=1e-8)
    for x,y in zip(p.dataset['first_atoms'],src['phonopy']['independent_displacements']):
        assert x['number']==y['atom_1based']-1 and np.allclose(x['displacement'],y['vector_A'],atol=1e-14)
    structures=[geometry(p.supercell)]
    for scale in SCALES:structures += [geometry(c) for c in load_ph(scale).supercells_with_displacements]
    assert len(structures)==25
    dump(out/'structures.json',structures)
    # Exact-composition, periodic species-resolved pair-distribution screening.
    # This identifies crystalline near duplicates despite different supercells;
    # it is not a guarantee of independence from unknown FM pretraining corpora.
    from ase import Atoms
    from ase.neighborlist import neighbor_list
    def fingerprint(g):
        a=Atoms(g['symbols'],positions=g['positions'],cell=g['cell'],pbc=True)
        i,j,d=neighbor_list('ijd',a,5.0);z=a.numbers
        bins=np.linspace(0,5,501);v=[]
        for x,y in [(13,13),(13,22),(22,22)]:
            mask=((z[i]==x)&(z[j]==y))|((z[i]==y)&(z[j]==x))
            v.extend(np.histogram(d[mask],bins)[0]/len(a))
        return np.array(v)
    base=geometry(p.supercell);fp=fingerprint(base);candidates=json.loads((ROOT/'overlap-candidates.json').read_text());screen=[]
    for g in candidates:
        dist=float(np.linalg.norm(fingerprint(g)-fp)/np.linalg.norm(fp))
        prim=spglib.find_primitive(spg(g),symprec=1e-4)
        screen.append(dict(id=g['id'],split=g['split'],family=g['family'],relative_pair_histogram_distance=dist,primitive_atoms=None if prim is None else len(prim[2])))
    dump(out/'overlap-screen.json',dict(method='Composition exact 3:1; species-resolved 0.01A radial histogram within5A, per atom; primitive reduction at1e-4A. Screening only, not a proof for distorted equivalents.',candidates=screen,FM_pretraining_overlap='unknown'))
    sg=spglib.get_spacegroup(spg(base),symprec=1e-5)
    dump(out/'summary.json',dict(status='PASS',phonopy=phonopy.__version__,spacegroup=sg,unitcell_atoms=8,supercell_atoms=64,primitive_atoms=8,masses=p.primitive.masses,displacements=8,structures=25,geometry_sha256=sha(out/'structures.json'),source_force_constants_available=False,DFT_relaxation_convergence='unknown',frequency_factor=p.unit_conversion_factor))
    print('PREPARED',sg,flush=True)
def band_points():
    vertices=np.array([[0,0,0],[.5,0,0],[.3333,.3333,0],[0,0,0],[0,0,.5],[.5,0,.5]])
    return np.concatenate([np.linspace(a,b,101) for a,b in zip(vertices[:-1],vertices[1:])])
def mesh(p,n=14,eigen=False):
    p.run_mesh([n,n,n],is_gamma_center=False,is_mesh_symmetry=not eigen,with_eigenvectors=eigen)
    return p.mesh
def mesh_info(m):return dict(min_THz=float(m.frequencies.min()),max_THz=float(m.frequencies.max()),imaginary_modes_weighted=int(np.sum(m.weights[:,None]*(m.frequencies<-.001))),total_modes_weighted=int(m.weights.sum()*m.frequencies.shape[1]))
def dos_metrics(freq,dos,df,dd):
    # Fixed extended frequency grid includes model tails and negative modes.
    r=np.interp(freq,df,dd,left=0,right=0)
    area=float(np.trapezoid(dos,freq));rarea=float(np.trapezoid(r,freq))
    c=np.r_[0,np.cumsum((dos[1:]+dos[:-1])/2*np.diff(freq))]/area
    cr=np.r_[0,np.cumsum((r[1:]+r[:-1])/2*np.diff(freq))]/rarea
    return dict(DOS_integral_per_atom=area,DFT_tabulated_integral_per_atom=rarea,normalized_DOS_L1=float(np.trapezoid(abs(dos/area-r/rarea),freq)),DOS_Wasserstein_THz=float(np.trapezoid(abs(c-cr),freq)),DOS_peak_THz=float(freq[np.argmax(dos)]))
def analyze_case(c):
    name=c['name'];w=ROOT/'forces'/name
    s=json.loads((w/'summary.json').read_text());assert s['status']=='PASS' and s['case']==c
    assert sha(w/'predictions.npz')==s['predictions_sha256']
    assert s['geometry_sha256']==sha(ROOT/'prepared/structures.json')
    f=np.load(w/'predictions.npz')['forces'];assert f.shape==(25,64,3)
    out=ROOT/'analysis'/name;out.mkdir(parents=True,exist_ok=False)
    ref=np.loadtxt(ROOT/'source/Ti3Al_D019/total_dos.dat');pd=np.loadtxt(ROOT/'source/Ti3Al_D019/projected_dos.dat')
    q=band_points();bands={};scales={};main=None
    for i,scale in enumerate(SCALES):
        p=load_ph(scale);p.forces=f[1+8*i:1+8*(i+1)]
        p.produce_force_constants(fc_calculator='traditional',calculate_full_force_constants=True,show_drift=False)
        fc=p.force_constants.copy();assert np.isfinite(fc).all()
        np.save(out/f'force_constants_scale{scale}.npy',fc)
        p.run_qpoints(q,with_eigenvectors=False);bands[scale]=p.qpoints.frequencies.copy()
        m=mesh(p);scales[str(scale)]=mesh_info(m)
        if scale==1.:
            main=p
            np.savez_compressed(out/'mesh14.npz',qpoints=m.qpoints,frequencies=m.frequencies,weights=m.weights)
            p.run_total_dos(freq_min=-5,freq_max=20,freq_pitch=.01,use_tetrahedron_method=True)
            freq=p.total_dos.frequency_points;dos=p.total_dos.dos/8
            np.savez_compressed(out/'dos_extended.npz',frequency=freq,dos=dos)
            metrics=dos_metrics(freq,dos,ref[:,0],ref[:,1]/8)
            assert abs(metrics['DOS_integral_per_atom']-3)<.03,metrics
            pitch=(ref[-1,0]-ref[0,0])/(len(ref)-1)
            p.run_total_dos(freq_min=ref[0,0],freq_max=ref[-1,0],freq_pitch=pitch,use_tetrahedron_method=True)
            np.savez_compressed(out/'dos_archived_grid.npz',frequency=p.total_dos.frequency_points,dos=p.total_dos.dos/8)
            mesh(p,eigen=True)
            p.run_projected_dos(freq_min=pd[0,0],freq_max=pd[-1,0],freq_pitch=(pd[-1,0]-pd[0,0])/(len(pd)-1),use_tetrahedron_method=True)
            projected=p.projected_dos.projected_dos/8
            assert projected.shape[0]==8
            np.savez_compressed(out/'projected_dos.npz',frequency=p.projected_dos.frequency_points,Al=projected[:2].sum(axis=0),Ti=projected[2:].sum(axis=0),total=projected.sum(axis=0))
    np.savez_compressed(out/'bands.npz',qpoints=q,half=bands[.5],main=bands[1.],double=bands[2.])
    convergence={str(scale):dict(band_RMSE_to_001A_THz=float(np.sqrt(np.mean((bands[scale]-bands[1.])**2))),band_max_difference_THz=float(np.max(abs(bands[scale]-bands[1.])))) for scale in [.5,2.]}
    main.symmetrize_force_constants();main.run_qpoints(q)
    asr=dict(band_RMSE_change_THz=float(np.sqrt(np.mean((main.qpoints.frequencies-bands[1.])**2))),mesh=mesh_info(mesh(main)))
    np.save(out/'force_constants_ASR.npy',main.force_constants)
    # Mesh sensitivity keeps the same force constants: this does not test supercell convergence.
    main.force_constants=np.load(out/'force_constants_scale1.0.npy');mesh(main,n=20)
    main.run_total_dos(freq_min=-5,freq_max=20,freq_pitch=.01,use_tetrahedron_method=True)
    mesh20=dos_metrics(main.total_dos.frequency_points,main.total_dos.dos/8,ref[:,0],ref[:,1]/8)
    result=dict(status='PASS',case=c,metrics=metrics,mesh14=scales['1.0'],displacement_mesh=scales,displacement_convergence=convergence,ASR_sensitivity=asr,mesh20_metrics=mesh20,Gamma_THz=bands[1.][0],max_residual_force_eV_A=s['max_residual_force_eV_A'],residual_stress_GPa=s['residual_stress_GPa'],geometry_sha256=s['geometry_sha256'],source_summary_sha256=sha(w/'summary.json'))
    dump(out/'summary.json',result);print(name,'PASS',metrics,flush=True);return result
def analyze():
    cases={};failures={}
    for c in json.loads((ROOT/'models.json').read_text()):
        try:cases[c['name']]=analyze_case(c)
        except Exception as e:
            import traceback
            traceback.print_exc();failures[c['name']]=repr(e)
    result=dict(status='PASS' if len(cases)==len(json.loads((ROOT/'models.json').read_text())) and not failures else 'INCOMPLETE',cases=cases,failures=failures,reference='Archived PBE DOS; DFT FORCE_SETS unavailable',seed_counts={'GRACE':2},r2SCAN_tested=False)
    dump(ROOT/'analysis/summary.json',result)
    if failures:raise SystemExit(1)
if __name__=='__main__':
    assert os.environ.get('SLURM_JOB_ID')
    for n,h in json.loads((ROOT/'FILES_SHA256.json').read_text()).items():assert sha(ROOT/n)==h,n
    {'prepare':prepare,'analyze':analyze}[sys.argv[1]]()
