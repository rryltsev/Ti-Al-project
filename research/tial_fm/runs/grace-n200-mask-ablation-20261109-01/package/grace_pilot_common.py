"""Shared native fp64 model loading, predictions and numerical qualification."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parent
def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tensor_sha(v):
    a=np.asarray(v.numpy());b=json.dumps([x.hex() if isinstance(x,bytes) else str(x) for x in a.ravel()]).encode() if a.dtype.kind in 'OUS' else a.tobytes()
    return hashlib.sha256(b).hexdigest()
def load_model(yaml,prefix):
    import tensorflow as tf
    from tensorpotential import TensorPotential
    from tensorpotential.instructions.base import load_instructions
    m=TensorPotential(potential=load_instructions(str(yaml)),param_dtype=tf.float64,jit_compile=False)
    m.load_checkpoint(checkpoint_name=str(prefix),model_only=True,expect_partial=True,assert_existing_objects_matched=True)
    return m
def calculator(model,export=False):
    from tensorpotential.calculator import TPCalculator
    if not export:model.model.decorate_compute_function(jit_compile=True)
    return TPCalculator(str(model) if export else model.model,enable_uq_if_available=False,adaptive_padding=False,max_number_reduction_recompilation=0)
def predict(calc,at):
    from ase.calculators.calculator import all_changes
    calc.calculate(at,properties=['energy','forces','stress'],system_changes=all_changes);r=calc.results
    e=float(r['energy']);f=np.array(r['forces']);s=np.array(r['stress'])
    assert np.isfinite(e) and np.isfinite(f).all() and np.isfinite(s).all()
    return e,f,s
def atoms(probe):
    from ase import Atoms
    return Atoms(probe['symbols'],positions=probe['positions'],cell=probe['cell'],pbc=True)
def numerical_qc(calc,export_calc=None):
    from ase.stress import voigt_6_to_full_3x3_stress
    records=[]
    for probe in json.loads((ROOT/'probes.json').read_text()):
        at=atoms(probe);e,f,s=predict(calc,at);n=len(at);rng=np.random.default_rng(103);sym={}
        def delta(ee,ff,ss):return dict(E_eV_atom=abs(ee-e)/n,F_max_eV_A=float(np.max(abs(ff-f))),stress_max_eV_A3=float(np.max(abs(ss-s))))
        order=rng.permutation(n);a=at[order];ee,ff,ss=predict(calc,a);sym['permutation']=delta(ee,ff[np.argsort(order)],ss)
        a=at.copy();a.positions+=np.array([.73,-.29,.41])@at.cell.array;a.positions[::3]+=at.cell.array[1];ee,ff,ss=predict(calc,a);sym['translation_PBC']=delta(ee,ff,ss)
        q,_=np.linalg.qr(rng.normal(size=(3,3)))
        if np.linalg.det(q)<0:q[:,0]*=-1
        a=at.copy();a.positions=at.positions@q;a.set_cell(at.cell.array@q);ee,ff,ss=predict(calc,a)
        ss=q@voigt_6_to_full_3x3_stress(ss)@q.T;ss=ss.ravel()[[0,4,8,5,2,1]];sym['rotation']=delta(ee,ff@q.T,ss)
        if export_calc:sym['export']=delta(*predict(export_calc,at))
        idx=np.unravel_index(np.argmax(abs(f)),f.shape);fd=[]
        for h in [.001,.0003,.0001]:
            a=at.copy();b=at.copy();a.positions[idx]+=h;b.positions[idx]-=h
            num=-(predict(calc,a)[0]-predict(calc,b)[0])/(2*h);fd.append(dict(h=h,error=abs(num-f[idx])))
        stress=[]
        for k,(i,j) in enumerate([(0,0),(1,1),(2,2),(1,2),(0,2),(0,1)]):
            errors=[]
            for h in [.001,.0003,.0001]:
                strain=np.zeros((3,3));strain[i,j]=1 if i==j else .5
                if i!=j:strain[j,i]=.5
                a=at.copy();b=at.copy();a.set_cell(at.cell.array@(np.eye(3)+h*strain),scale_atoms=True);b.set_cell(at.cell.array@(np.eye(3)-h*strain),scale_atoms=True)
                num=(predict(calc,a)[0]-predict(calc,b)[0])/(2*h*at.get_volume());errors.append(dict(h=h,error_GPa=abs(num-s[k])*160.21766208))
            stress.append(errors)
        passed=all(d['E_eV_atom']<1e-8 and d['F_max_eV_A']<1e-7 and d['stress_max_eV_A3']<1e-8 for d in sym.values()) and min(x['error'] for x in fd)<2e-4 and all(min(x['error_GPa'] for x in rr)<.005 for rr in stress)
        records.append(dict(id=probe['id'],status='PASS' if passed else 'FAIL',symmetries=sym,force_FD=fd,stress_FD=stress,base=dict(energy=e,forces=f,stress=s)))
    return dict(status='PASS' if all(r['status']=='PASS' for r in records) else 'FAIL',records=records)
def metrics(rows,offset=None):
    train=[r for r in rows if r['split']=='train'];x=np.array([r['fractions'] for r in train]);y=np.array([(r['reference_energy']-r['energy'])/r['natoms'] for r in train])
    b=np.linalg.lstsq(x,y,rcond=None)[0] if offset is None else np.array(offset);assert np.linalg.matrix_rank(x)==2
    out={}
    for part in ['train','validation']:
        rr=[r for r in rows if r['split']==part];ee=np.array([(r['energy']-r['reference_energy'])/r['natoms'] for r in rr]);aligned=ee+np.array([r['fractions'] for r in rr])@b
        out[part]=dict(structures=len(rr),E_raw_RMSE_meV_atom=float(np.sqrt(np.mean(ee**2))*1000),E_aligned_RMSE_meV_atom=float(np.sqrt(np.mean(aligned**2))*1000),F_RMSE_eV_A=float(np.sqrt(sum(r['force_sq'] for r in rr)/sum(3*r['natoms'] for r in rr))),stress_RMSE_GPa=float(np.sqrt(np.mean([r['stress_sq']/6 for r in rr]))))
    return dict(offset_eV_atom=dict(zip(['Al','Ti'],b.tolist())),alignment='Equal-structure least squares in E/atom, train200 only',metrics=out)
def evaluate_dataset(calc,xc):
    from ase.io import read
    rows=[]
    for part in ['train','validation']:
        aa=read(ROOT/'data'/xc/(part+'.extxyz'),index=':');assert len(aa)==(200 if part=='train' else 149)
        for at in aa:
            e,f,s=predict(calc,at);rows.append(dict(id=at.info['geometry_id'],split=part,natoms=len(at),fractions=[at.get_chemical_symbols().count('Al')/len(at),at.get_chemical_symbols().count('Ti')/len(at)],energy=e,reference_energy=float(at.get_potential_energy()),force_sq=float(np.sum((f-at.get_forces())**2)),stress_sq=float(np.sum(((s-at.get_stress())*160.21766208)**2))))
    return rows
