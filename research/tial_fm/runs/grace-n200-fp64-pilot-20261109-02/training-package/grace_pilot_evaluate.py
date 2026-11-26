from pathlib import Path
import argparse,json
import numpy as np
from grace_pilot_common import ROOT,load_model,calculator,evaluate_dataset,metrics,numerical_qc,predict,dump,sha
def main():
    from ase.io import read
    ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);a=ap.parse_args();work=ROOT/'cases'/a.case;xc=a.case.split('-')[0]
    t=json.loads((work/'training-summary.json').read_text());assert t['status']=='PASS'
    for n,h in t['hashes'].items():assert sha(work/'final-checkpoint'/n)==h
    model=load_model(work/'seed/10/model.yaml',work/'final-checkpoint/checkpoint');calc=calculator(model)
    qc=numerical_qc(calc,calculator(work/'seed/10/final_model',export=True));dump(work/'numerical-QC.json',qc);assert qc['status']=='PASS'
    rows=evaluate_dataset(calc,xc);m=metrics(rows);dump(work/'metrics.json',m);dump(work/'per-structure.json',rows)
    for key in ['force000','force001','force002']:
        at=read(ROOT/'harmonic_inputs'/xc/key/'POSCAR',format='vasp');_,f,_=predict(calc,at);p=work/'harmonic_forces'/xc;p.mkdir(parents=True,exist_ok=True);np.save(p/(key+'.npy'),f)
    dump(work/'evaluation-summary.json',dict(status='PASS',train=200,validation=149,metrics=m,test_access=False,fixed_final=True))
if __name__=='__main__':main()
