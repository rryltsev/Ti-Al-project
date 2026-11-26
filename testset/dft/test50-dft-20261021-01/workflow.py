"""Prepare and execute one immutable test50 single-point per array element."""
from pathlib import Path
import hashlib, json, os, re, shutil, subprocess, sys, time, traceback
import numpy as np
ROOT=Path(__file__).resolve().parent
C=json.loads((ROOT/'protocol.json').read_text())
DATA=Path(C['dataset_root'])

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def poscar(p):
    ls=p.read_text().splitlines();assert float(ls[1])==1
    cell=np.array([list(map(float,l.split())) for l in ls[2:5]])
    elements=ls[5].split();counts=list(map(int,ls[6].split()));assert ls[7].lower().startswith('d')
    frac=np.array([list(map(float,l.split()[:3])) for l in ls[8:8+sum(counts)]])
    nums=sum(([{'Al':13,'Ti':22}[e]]*n for e,n in zip(elements,counts)),[])
    return cell,frac,nums,elements
def tr_reduce(coords):
    coords=np.mod(np.array(coords,float),1);seen=set();points=[];weights=[]
    for i,k in enumerate(coords):
        if i in seen:continue
        delta=coords+k;delta-=np.rint(delta);ii=np.flatnonzero(np.max(abs(delta),axis=1)<1e-8)
        assert len(ii)==1; j=int(ii[0]);assert j not in seen
        points.append(k.tolist());weights.append(1 if j==i else 2);seen.update((i,j))
    assert sum(weights)==len(coords)
    return points,weights
def grid(p):
    from kpLib.interface import get_kpoints
    import spglib
    cell,frac,nums,_=poscar(p)
    g=get_kpoints(cell,frac,nums,min_distance=35.0,min_total_kpoints=1,include_gamma='auto',symprec=1e-5,use_scale_factor=False)
    assert g['min_periodic_distance']>=35.0-1e-7
    assert len(g['coords'])==g['num_total_kpts'] and sum(g['weights'])==g['num_total_kpts']
    pts,weights=tr_reduce(g['coords'])
    sym=spglib.get_symmetry_dataset((cell,frac,nums),symprec=1e-5)
    assert len(sym.rotations)==1,'Unexpected spatial symmetry; review reduction before submitting'
    text='kpLib 1.1.1; MINDISTANCE=35 A; INCLUDEGAMMA=AUTO; time reversal\n'+str(len(pts))+'\nReciprocal\n'
    text+=''.join(' '.join(f'{v:.16f}' for v in k)+f' {w}\n' for k,w in zip(pts,weights))
    return text,dict(num_full=int(g['num_total_kpts']),num_irreducible=len(pts),effective_min_distance_A=float(g['min_periodic_distance']),spatial_rotations=len(sym.rotations),gamma_included=bool(np.any(np.max(abs(np.array(pts)-np.rint(pts)),axis=1)<1e-8)))

def prepare():
    assert not (ROOT/'preflight.json').exists()
    for name,h in json.loads((DATA/'SHA256SUMS.json').read_text()).items():assert sha(DATA/name)==h
    manifest=json.loads((DATA/'manifest.json').read_text());assert len(manifest)==50
    for element,h in C['paw_sha256'].items():assert sha(ROOT/'private'/('POTCAR_'+element))==h
    calibration=[]
    for p in sorted((ROOT/'legacy').glob('*/POSCAR')):
        # Old POSCARs may use Cartesian coordinates; ASE conversion is avoided:
        # calibration is performed only if the input is already Direct.
        lines=p.read_text().splitlines()
        if not lines[7].lower().startswith('d'):
            cell=np.array([list(map(float,l.split())) for l in lines[2:5]])*float(lines[1]);n=sum(map(int,lines[6].split()))
            coords=np.array([list(map(float,l.split()[:3])) for l in lines[8:8+n]])*float(lines[1]);frac=coords@np.linalg.inv(cell)
            q=p.with_name('POSCAR.direct');q.write_text(lines[0]+'\n1.0\n'+''.join(' '.join(map(str,v))+'\n' for v in cell)+lines[5]+'\n'+lines[6]+'\nDirect\n'+''.join(' '.join(map(str,v))+'\n' for v in frac))
        else:q=p
        text,meta=grid(q)
        old=p.with_name('KPOINTS').read_text().splitlines();n=int(old[1]);oldpoints=np.array([list(map(float,l.split()[:3])) for l in old[3:3+n]])
        new=text.splitlines();nn=int(new[1]);newpoints=np.array([list(map(float,l.split()[:3])) for l in new[3:3+nn]])
        equal=n==nn
        for k in oldpoints:
            delta=newpoints-k;delta-=np.rint(delta)
            delta2=newpoints+k;delta2-=np.rint(delta2)
            equal=equal and bool(np.any(np.max(abs(delta),axis=1)<1e-7)|np.any(np.max(abs(delta2),axis=1)<1e-7))
        calibration.append(dict(source=str(p.relative_to(ROOT)),old_nkpts=n,generated=meta,same_points_modulo_time_reversal=equal))
    save(ROOT/'grid-calibration.json',calibration)
    checks=[]
    for row in manifest:
        sid=row['structure_id'];common=ROOT/'common'/sid;common.mkdir(parents=True)
        shutil.copyfile(DATA/row['poscar'],common/'POSCAR');assert sha(common/'POSCAR')==row['poscar_sha256']
        kp,grid_info=grid(common/'POSCAR');(common/'KPOINTS').write_text(kp)
        _,_,_,elements=poscar(common/'POSCAR')
        (common/'POTCAR').write_bytes(b''.join((ROOT/'private'/('POTCAR_'+el)).read_bytes() for el in elements))
        hashes={n:sha(common/n) for n in ['POSCAR','KPOINTS','POTCAR']}
        for xc in C['functionals']:
            dest=ROOT/'calculations'/xc/sid;dest.mkdir(parents=True)
            for name in hashes:shutil.copyfile(common/name,dest/name)
            tags=dict(SYSTEM=sid+' '+xc,ENCUT=C['encut_eV'],**C['common_incar'])
            if xc=='r2SCAN':tags['METAGGA']='R2SCAN'
            else:tags['GGA']='PE' if xc=='PBE' else 'PS'
            (dest/'INCAR').write_text(''.join(f'{k} = {v}\n' for k,v in tags.items()))
            assert tags['NSW']==0 and tags['IBRION']==-1
            assert all(sha(dest/n)==h for n,h in hashes.items())
            save(dest/'input-manifest.json',dict(structure_id=sid,functional=xc,input_sha256={**hashes,'INCAR':sha(dest/'INCAR')},incar=tags))
        checks.append(dict(structure_id=sid,source_poscar_sha256=row['poscar_sha256'],shared_input_sha256=hashes,grid=grid_info,all_three_identical=True))
        print('prepared',sid,grid_info,flush=True)
    import importlib.metadata
    save(ROOT/'preflight.json',dict(status='PASS',structures=50,calculations=150,identical_geometries=True,identical_kpoints=True,identical_paw_per_structure=True,NSW=0,IBRION=-1,records=checks,vasp_sha256=sha(C['vasp_binary']),software={'python':sys.version,'kpLib':importlib.metadata.version('kpLib'),'numpy':np.__version__},protocol_sha256=sha(ROOT/'protocol.json')))

def run(xc,index):
    assert xc in C['functionals'] and 0<=index<50
    pre=json.loads((ROOT/'preflight.json').read_text());assert pre['status']=='PASS' and pre['protocol_sha256']==sha(ROOT/'protocol.json')
    row=pre['records'][index];sid=row['structure_id'];p=ROOT/'calculations'/xc/sid
    intent=p/'execution-intent.json'
    with intent.open('x') as f:json.dump(dict(job_id=os.environ['SLURM_JOB_ID'],array_job_id=os.environ.get('SLURM_ARRAY_JOB_ID'),index=index,started_unix=time.time()),f)
    status=p/'status.json'
    try:
        inputs=json.loads((p/'input-manifest.json').read_text());tags=inputs['incar']
        assert tags['NSW']==0 and tags['IBRION']==-1
        assert all(sha(p/n)==h for n,h in inputs['input_sha256'].items())
        assert sha(p/'POSCAR')==row['source_poscar_sha256'] and sha(C['vasp_binary'])==pre['vasp_sha256']
        assert not any((p/n).exists() for n in ['OUTCAR','vasprun.xml','WAVECAR','CHGCAR'])
        save(status,dict(state='running',structure_id=sid,functional=xc,job_id=os.environ['SLURM_JOB_ID']))
        with (p/'vasp.stdout').open('w') as f:
            result=subprocess.run(['srun','--ntasks=24',C['vasp_binary']],cwd=p,stdout=f,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError('VASP exit '+str(result.returncode))
        validate(p,inputs,row)
        save(status,dict(state='validated',structure_id=sid,functional=xc,job_id=os.environ['SLURM_JOB_ID'],finished_unix=time.time()))
    except Exception as e:
        save(status,dict(state='failed',structure_id=sid,functional=xc,error=str(e),job_id=os.environ['SLURM_JOB_ID']))
        raise

def validate(p,inputs,row):
    import xml.etree.ElementTree as ET
    from ase.io import read
    out=(p/'OUTCAR').read_text(errors='replace')
    assert 'General timing and accounting' in out,'Missing normal termination'
    assert 'aborting loop because EDIFF is reached' in out,'SCF not confirmed converged'
    tree=ET.parse(p/'vasprun.xml');root=tree.getroot();calcs=root.findall('calculation');assert len(calcs)==1,'Expected exactly one fixed-geometry calculation'
    values={el.attrib.get('name'):el.text.strip() for el in root.findall('./parameters//i') if el.text}
    assert int(values['NSW'])==0 and int(values['IBRION'])==-1 and int(values['ISPIN'])==1
    # VASP records the effective plane-wave cutoff as ENMAX in parameters.
    assert abs(float(values['ENMAX'])-550)<1e-8
    assert int(values['ISMEAR'])==1 and abs(float(values['SIGMA'])-.1)<1e-10 and abs(float(values['EDIFF'])-1e-6)<1e-12
    incar_values={el.attrib.get('name'):el.text.strip() for el in root.findall('./incar/i') if el.text}
    if inputs['functional']=='r2SCAN':assert incar_values['METAGGA'].upper()=='R2SCAN'
    else:
        expected='PE' if inputs['functional']=='PBE' else 'PS'
        # VASP 6.6 omits GGA from XML parameters; INCAR and OUTCAR retain it.
        assert incar_values['GGA'].upper()==expected
        actual=re.findall(r'\bGGA\s*=\s*([^\s;#]+)',out)
        assert actual and actual[-1].upper()==expected
    initial=read(p/'POSCAR');final=read(p/'vasprun.xml',format='vasp-xml',index=-1)
    assert np.array_equal(initial.numbers,final.numbers)
    cell_error=float(np.max(abs(initial.cell.array-final.cell.array)));assert cell_error<1e-7
    delta=initial.get_scaled_positions()-final.get_scaled_positions();delta-=np.rint(delta)
    # XML fractional coordinates are commonly printed to eight decimals.
    geometry_error=float(np.max(np.linalg.norm(delta@initial.cell.array,axis=1)));assert geometry_error<1e-6
    forces=final.get_forces();stress=final.get_stress(voigt=False);energy=final.get_potential_energy();free=final.get_potential_energy(force_consistent=True)
    assert np.isfinite(forces).all() and np.isfinite(stress).all() and np.isfinite([energy,free]).all()
    assert sha(p/'POSCAR')==row['source_poscar_sha256']
    # The archived fine-tuning XYZ stores VASP free energy (TOTEN), verified
    # directly against its corresponding OUTCAR; retain sigma->0 separately.
    save(p/'label.json',dict(structure_id=inputs['structure_id'],functional=inputs['functional'],energy_eV=float(free),energy_definition='VASP free energy TOTEN; matches legacy fine-tuning targets',free_energy_eV=float(free),energy_sigma0_eV=float(energy),forces_eV_A=forces.tolist(),stress_eV_A3=stress.tolist(),stress_convention='ASE tensile positive',virial_eV=(-stress*final.get_volume()).tolist(),volume_A3=final.get_volume(),n_atoms=len(final),max_position_change_A=geometry_error,max_cell_change_A=cell_error,ionic_calculations=1,scf_converged=True,input_sha256=inputs['input_sha256']))
    save(p/'output-manifest.json',{n:sha(p/n) for n in ['OUTCAR','vasprun.xml','OSZICAR','CONTCAR','label.json'] if (p/n).exists()})

if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='run':run(sys.argv[2],int(sys.argv[3]))
    else:raise ValueError('Unknown action')
