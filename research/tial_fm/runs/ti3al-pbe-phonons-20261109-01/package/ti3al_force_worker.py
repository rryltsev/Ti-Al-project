"""Slurm-only native-model forces at fixed archived Ti3Al displacements."""
from pathlib import Path
import hashlib,json,os,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
def main():
    assert os.environ.get('SLURM_JOB_ID')
    for n,h in json.loads((ROOT/'FILES_SHA256.json').read_text()).items():assert sha(ROOT/n)==h,n
    assert json.loads((ROOT/'prepared/summary.json').read_text())['status']=='PASS'
    c=json.loads((ROOT/'models.json').read_text())[int(sys.argv[1])]
    out=ROOT/'forces'/c['name'];out.mkdir(parents=True,exist_ok=False)
    structures=json.loads((ROOT/'prepared/structures.json').read_text());start=time.monotonic()
    if c['backend']=='DPA4':
        from deepmd.infer import DeepPot
        assert sha(c['model'])==c['sha256'];dp=DeepPot(c['model'])
        def predict(g):
            e,f,v=dp.eval(np.array(g['positions']).reshape(1,-1),np.array(g['cell']).reshape(1,9),[dp.get_type_map().index(x) for x in g['symbols']])[:3]
            volume=abs(np.linalg.det(g['cell']));stress=-v.reshape(3,3)/volume
            return float(e.ravel()[0]),f.reshape(-1,3),stress
    else:
        from ase import Atoms
        from ase.stress import voigt_6_to_full_3x3_stress
        from grace_pilot_common import load_model,calculator,predict as gp
        w=Path(c['model'])
        for n,h in c['hashes'].items():assert sha(w/'final-checkpoint'/n)==h
        assert sha(w/'seed/10/model.yaml')==c['yaml_sha256']
        calc=calculator(load_model(w/'seed/10/model.yaml',w/'final-checkpoint/checkpoint'))
        def predict(g):
            e,f,s=gp(calc,Atoms(g['symbols'],positions=g['positions'],cell=g['cell'],pbc=True))
            return e,f,voigt_6_to_full_3x3_stress(s)
    forces=[];energies=[];stresses=[]
    for g in structures:
        e,f,s=predict(g);assert np.isfinite(e) and np.isfinite(f).all() and np.isfinite(s).all()
        forces.append(f);energies.append(e);stresses.append(s)
    np.savez_compressed(out/'predictions.npz',forces=forces,energies=energies,stress_eV_A3=stresses)
    dump(out/'summary.json',dict(status='PASS',case=c,job_id=os.environ['SLURM_JOB_ID'],structures=len(structures),seconds=time.monotonic()-start,geometry_sha256=sha(ROOT/'prepared/structures.json'),predictions_sha256=sha(out/'predictions.npz'),max_residual_force_eV_A=float(np.linalg.norm(forces[0],axis=1).max()),residual_stress_GPa=np.array(stresses[0])*160.21766208))
    print(c['name'],'PASS',len(structures),flush=True)
if __name__=='__main__':main()
