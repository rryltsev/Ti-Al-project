from pathlib import Path
import sys,json,os,time
OUT=Path(__file__).parent
sys.path.insert(0,str(OUT/'deps'))
os.environ['MPLBACKEND']='Agg'
import numpy as np
from ase.io.lammpsdata import read_lammps_data
import stacie
from stacie import compute_spectrum,estimate_acint,LorentzModel
ROOT=Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project')
rows=json.loads((OUT/'viscosity_merged.json').read_text())
results=[]
for r in rows:
  host='cluster' if 'stacie_pipeline_cluster' in r['source'] else 'wsl2'
  base=ROOT/f'results-scripts-logs/models_check_{host}/results/viscosity/dpa4/neo/finetune/AlTi/{r["T"]}K/{r["dft"]}/03_stress_autocorr/c_Ti_{r["c_Ti"]:.2f}'
  seq=[];pre=[]
  for p in sorted(base.glob('visc*/pressure_*.data')):
    a=np.loadtxt(p)*1e5
    vol=read_lammps_data(str(p.parent/'mol.data'),atom_style='atomic').get_volume()*1e-30
    seq.extend([(a[:,1]-.5*a[:,2]-.5*a[:,3])/np.sqrt(3),.5*(a[:,2]-a[:,3]),a[:,4],a[:,5],a[:,6]])
    pre.extend([vol/(1.380649e-23*r['T'])]*5)
  seq=np.array(seq);pre=np.array(pre)
  variants=['full']
  if (r['T'],r['dft'],r['c_Ti']) in [(1873,'pbe',.6),(1873,'pbe',.7),(1873,'scan',.4),(1873,'scan',.55)]:
    variants+=['offdiagonal','first_half','second_half','replicas_odd','replicas_even']
  result_row={k:r[k] for k in ['T','dft','c_Ti','eta_mPa_s','eta_std_mPa_s']};result_row['stacie_version']=getattr(stacie,'__version__','unknown'); result_row['checks']={}
  for v in variants:
    s=seq;f=pre
    if v=='offdiagonal':
      idx=np.arange(len(s))%5>=2;s=s[idx];f=f[idx]
    elif v=='first_half':s=s[:,:10000]
    elif v=='second_half':s=s[:,10000:20000]
    elif v.startswith('replicas_'):
      idx=(np.arange(len(s))//5)%2==(0 if v.endswith('odd') else 1);s=s[idx];f=f[idx]
    start=time.time()
    try:
      estimate=estimate_acint(compute_spectrum(s,prefactors=f,timestep=1e-15),LorentzModel(),verbose=False)
      d=dict(eta_mPa_s=float(estimate.acint*1000),std_mPa_s=float(estimate.acint_std*1000),neff=float(estimate.neff),seconds=time.time()-start)
    except Exception as e:d={'error':str(e)}
    result_row['checks'][v]=d
    print(r['T'],r['dft'],r['c_Ti'],v,d,flush=True)
  results.append(result_row)
  (OUT/'stacie_recheck.json').write_text(json.dumps(results,indent=2),encoding='utf8')
