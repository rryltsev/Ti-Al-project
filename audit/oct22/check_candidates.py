"""Additional subset diagnostics on unchanged archived 20-ps ensembles."""
from pathlib import Path
import json, sys, os
HERE=Path(__file__).resolve().parent
WORK=HERE.parents[1]
sys.path.insert(0,str(WORK/'audit/oct20/deps'))
os.environ['MPLBACKEND']='Agg'
import numpy as np
from ase.io.lammpsdata import read_lammps_data
from stacie import compute_spectrum, estimate_acint, LorentzModel
ROOT=Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project')
old=json.loads((WORK/'audit/oct20/viscosity_merged.json').read_text())
checks=json.loads((WORK/'audit/oct20/stacie_recheck.json').read_text())
targets=[('pbe',.3),('pbe',.4),('pbe',.9),('scan',.5),('scan',.6),('scan',.7),('scan',.8)]
out=[]
for model,c in targets:
    r=next(r for r in old if (r['T'],r['dft'],r['c_Ti'])==(1873,model,c))
    host='cluster' if 'cluster' in r['source'] else 'wsl2'
    base=ROOT/f'results-scripts-logs/models_check_{host}/results/viscosity/dpa4/neo/finetune/AlTi/1873K/{model}/03_stress_autocorr/c_Ti_{c:.2f}'
    seq,pre=[],[]
    files=sorted(base.glob('visc*/pressure_*.data'))
    assert len(files)==r['n_replicas'], (base,len(files),r['n_replicas'])
    for p in files:
        a=np.loadtxt(p)*1e5
        vol=read_lammps_data(str(p.parent/'mol.data'),atom_style='atomic').get_volume()*1e-30
        seq.extend([(a[:,1]-.5*a[:,2]-.5*a[:,3])/np.sqrt(3),.5*(a[:,2]-a[:,3]),a[:,4],a[:,5],a[:,6]])
        pre.extend([vol/(1.380649e-23*1873)]*5)
    seq=np.array(seq);pre=np.array(pre)
    result=dict(T=1873,dft=model,c_Ti=c,source=str(base),n_trajectories=len(files),checks={})
    result['checks']['full']=next(q['checks']['full'] for q in checks if (q['T'],q['dft'],q['c_Ti'])==(1873,model,c))
    for variant in ['offdiagonal','first_half','second_half','replicas_odd','replicas_even']:
        s,f=seq,pre
        if variant=='offdiagonal':
            idx=np.arange(len(s))%5>=2;s,f=s[idx],f[idx]
        elif variant=='first_half':s=s[:,:10000]
        elif variant=='second_half':s=s[:,10000:20000]
        else:
            idx=(np.arange(len(s))//5)%2==(0 if variant.endswith('odd') else 1)
            s,f=s[idx],f[idx]
        est=estimate_acint(compute_spectrum(s,prefactors=f,timestep=1e-15),LorentzModel(),verbose=False)
        result['checks'][variant]=dict(eta_mPa_s=float(est.acint*1000),std_mPa_s=float(est.acint_std*1000),neff=float(est.neff))
        print(model,c,variant,result['checks'][variant],flush=True)
    out.append(result)
    (HERE/'candidate_subset_checks.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
