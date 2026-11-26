from pathlib import Path
import argparse,json
import numpy as np
from grace_pilot_common import ROOT,load_model,calculator,evaluate_dataset,metrics,numerical_qc,predict,dump,sha
def main():
    from ase.io import read
    ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);a=ap.parse_args();work=ROOT/'cases'/a.case;xc=a.case.split('-')[0]
    t=json.loads((work/'training-summary.json').read_text());assert t['status']=='PASS'
    for n,h in t['hashes'].items():assert sha(work/'final-checkpoint'/n)==h
    seed=t['case']['seed'];seed_folder=work/'seed'/str(seed)
    model=load_model(seed_folder/'model.yaml',work/'final-checkpoint/checkpoint');calc=calculator(model)
    qc=numerical_qc(calc,calculator(seed_folder/'final_model',export=True));dump(work/'numerical-QC.json',qc);assert qc['status']=='PASS'
    rows=evaluate_dataset(calc,xc);m=metrics(rows);dump(work/'metrics.json',m);dump(work/'per-structure.json',rows)
    for key in ['force000','force001','force002']:
        at=read(ROOT/'harmonic_inputs'/xc/key/'POSCAR',format='vasp');_,f,_=predict(calc,at);p=work/'harmonic_forces'/xc;p.mkdir(parents=True,exist_ok=True);np.save(p/(key+'.npy'),f)
    dump(work/'evaluation-summary.json',dict(status='PASS',train=200,validation=149,metrics=m,test_access=False,fixed_final=True))
    if (ROOT/'ti3al/prepared/structures.json').exists():
        from ase import Atoms
        from ase.stress import voigt_6_to_full_3x3_stress
        import os,time
        start=time.monotonic();root=ROOT/'ti3al';out=root/'forces'/a.case;out.mkdir(parents=True,exist_ok=False)
        case=next(c for c in json.loads((root/'models.json').read_text()) if c['name']==a.case)
        fs=[];es=[];ss=[]
        for g in json.loads((root/'prepared/structures.json').read_text()):
            e,f,s=predict(calc,Atoms(g['symbols'],positions=g['positions'],cell=g['cell'],pbc=True));es.append(e);fs.append(f);ss.append(voigt_6_to_full_3x3_stress(s))
        np.savez_compressed(out/'predictions.npz',forces=fs,energies=es,stress_eV_A3=ss)
        dump(out/'summary.json',dict(status='PASS',case=case,job_id=os.environ['SLURM_JOB_ID'],structures=len(fs),seconds=time.monotonic()-start,geometry_sha256=sha(root/'prepared/structures.json'),predictions_sha256=sha(out/'predictions.npz'),max_residual_force_eV_A=float(np.linalg.norm(fs[0],axis=1).max()),residual_stress_GPa=np.array(ss[0])*160.21766208,checkpoint_hashes=t['hashes']))
if __name__=='__main__':main()
