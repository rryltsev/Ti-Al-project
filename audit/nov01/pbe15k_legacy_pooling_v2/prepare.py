"""Version 2: user-supplied historical raw files; isolated diagnostic."""
from pathlib import Path
import os,sys,json,hashlib,time
WORK=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(WORK/'audit/oct20/deps'))
os.environ['MPLCONFIGDIR']=str(OUT/'mplcache')
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from ase.io.lammpsdata import read_lammps_data
from stacie import compute_spectrum,estimate_acint,LorentzModel
BASE=Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project/results-scripts-logs/stacie_pipline_PBE-15k_27.10.2026')
CAMP=WORK/'audit/oct29/pbe15k_ti_temperature'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,o): (OUT/n).write_text(json.dumps(o,indent=2)+'\n',encoding='utf-8')
cfg=read(CAMP/'protocol.json')
out={'version':2,'created_unix':time.time(),'raw_found':True,'temperatures':{},'model_check_pending':True,'pooling_performed':False,'official_protocol_unchanged':True}
bundle=[]
for T,N in [(1873,25),(2000,20)]:
    root=BASE/f'pbe_{T}_15000/pbe'
    files=sorted((root/'03_stress_autocorr/c_Ti_1.00').glob('visc*/pressure_*.data'),key=lambda f:int(f.parent.name[4:]))
    assert len(files)==N
    seq=[];pref=[];manifest=[]
    for f in files:
        a=np.loadtxt(f); assert a.shape==(20001,7) and np.isfinite(a).all() and np.array_equal(a[:,0],np.arange(20001))
        mol=f.parent/'mol.data'; atoms=read_lammps_data(str(mol),atom_style='atomic',Z_of_type={1:13,2:22})
        assert len(atoms)==512 and np.all(atoms.numbers==22)
        V=float(atoms.get_volume());xx,yy,zz,xy,xz,yz=(a[:,1:]*1e5).T
        seq.append(np.array([(xx-.5*yy-.5*zz)/np.sqrt(3),.5*(yy-zz),xy,xz,yz]));pref.append(np.full(5,V*1e-30/(1.380649e-23*T)))
        i=int(f.parent.name[4:])
        manifest.append({'replica':i,'pressure':str(f),'pressure_sha256':sha(f),'mol':str(mol),'mol_sha256':sha(mol),'volume_A3':V})
        bundle.append({'T':T,'replica':i,'mol_text':mol.read_text(),'pressure_bar':a[0,1:].tolist()})
    assert len(set(x['pressure_sha256'] for x in manifest))==N
    seq=np.array(seq);pref=np.array(pref)
    np.savez_compressed(OUT/f'old_{T}.npz',seq=seq,pref=pref)
    r=estimate_acint(compute_spectrum(np.concatenate(seq),prefactors=np.concatenate(pref),timestep=1e-15),LorentzModel(),verbose=False)
    ref=read(BASE/f'results/AlTi/{T}/pbe/c_Ti_1.00/result.json')
    fit={'eta_mPa_s':float(r.acint*1000),'std_mPa_s':float(r.acint_std*1000),'cost_z':float(r.props['cost_zscore']),'cutoff_z':float(r.props['criterion_zscore'])}
    metas={s:read(root/s/'checkpoint.json') for s in ['01_generate_liquid','02_nvt_snapshots','03_stress_autocorr']}
    out['temperatures'][str(T)]={'n':N,'manifest':manifest,'metadata':metas,'reproduced':fit,'export':ref,'volume_mean_A3':float(np.mean([m['volume_A3'] for m in manifest])),'model_log_lines':[s.strip() for s in (root/'log.lammps').read_text().splitlines() if any(k in s for k in ['pair_style deepmd','fix NVT','timestep '])]}
    print('OLD_REPRODUCED',T,fit,'REFERENCE',ref,flush=True);save('provenance_v2.json',out)
save('model_input.json',bundle)
print('INPUT_READY',flush=True)

