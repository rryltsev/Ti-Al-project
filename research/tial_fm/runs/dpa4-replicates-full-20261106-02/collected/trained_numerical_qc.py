"""Repeat symmetries and multiple-step derivatives after nonzero adaptation."""
from pathlib import Path
import argparse,json,os
import numpy as np
from deepmd.infer import DeepPot
from baseline_worker import evaluate,differences,sha,dump
ROOT=Path(__file__).resolve().parent
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);ap.add_argument('--step',type=int);a=ap.parse_args()
    assert os.environ.get('SLURM_JOB_ID');case=ROOT/'training-plan01/cases'/a.case
    model=case/'merged'/f'{a.step:05d}.pt' if a.step else Path(json.loads((case/'training-summary.json').read_text())['final_checkpoint'])
    out=case/(f'numerical-check-step{a.step:05d}' if a.step else 'evaluation/numerical-QC');out.mkdir(parents=True,exist_ok=False)
    dp=DeepPot(str(model));records=[]
    for probe in json.loads((ROOT/'probes.json').read_text()):
        c=np.array(probe['positions']);b=np.array(probe['cell']);s=probe['symbols'];n=len(s);base=evaluate(dp,c,b,s);rng=np.random.default_rng(103)
        order=rng.permutation(n);p=evaluate(dp,c[order],b,[s[i] for i in order]);p=(p[0],p[1][np.argsort(order)],p[2])
        shift=c+np.array([.73,-.29,.41])@b;shift[::3]+=b[1];shift[1::3]-=b[2];t=evaluate(dp,shift,b,s)
        q,_=np.linalg.qr(rng.normal(size=(3,3)))
        if np.linalg.det(q)<0:q[:,0]*=-1
        r=evaluate(dp,c@q,b@q,s);r=(r[0],r[1]@q.T,q@r[2]@q.T)
        sym={k:differences(base,v,n) for k,v in [('permutation',p),('translation_PBC',t),('rotation',r)]}
        idx=np.unravel_index(np.argmax(np.abs(base[1])),base[1].shape);fd=[];vd=[]
        for h in [.01,.003,.001]:
            plus=c.copy();minus=c.copy();plus[idx]+=h;minus[idx]-=h
            numeric=-(evaluate(dp,plus,b,s)[0]-evaluate(dp,minus,b,s)[0])/(2*h)
            fd.append(dict(h_A=h,analytic=float(base[1][idx]),numeric=numeric,error=float(abs(numeric-base[1][idx]))))
        for h in [.003,.001,.0003]:
            plus=np.eye(3);minus=np.eye(3);plus[0,0]+=h;minus[0,0]-=h
            numeric=-(evaluate(dp,c@plus,b@plus,s)[0]-evaluate(dp,c@minus,b@minus,s)[0])/(2*h)
            vd.append(dict(strain=h,analytic=float(base[2][0,0]),numeric=numeric,error=float(abs(numeric-base[2][0,0]))))
        passed=all(v['energy_eV_atom']<1e-5 and v['force_max_eV_A']<2e-4 and v['virial_max_eV']<.01 for v in sym.values()) and min(v['error'] for v in fd)<.01 and min(v['error'] for v in vd)<.1
        records.append(dict(id=probe['id'],symmetries=sym,force_FD=fd,virial_FD=vd,status='PASS' if passed else 'FAIL'))
    passed=all(v['status']=='PASS' for v in records)
    dump(out/'summary.json',dict(status='PASS' if passed else 'FAIL',model=str(model),sha256=sha(model),job_id=os.environ['SLURM_JOB_ID'],checks=records,scope='Identical numerical tolerances and probes as original B0; no test labels used.'))
    assert passed,'Post-adaptation numerical gate failed; inspect saved checks'
    print('NUMERICAL_QC_PASS',a.case,a.step)
if __name__=='__main__':main()
