"""Read-only provenance audit of candidate legacy Ti trajectories. No pooling."""
from pathlib import Path
import sys,os,json,hashlib,time
WORK=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(WORK/'audit/oct20/deps'))
os.environ['MPLCONFIGDIR']=str(OUT/'mplcache')
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import numpy as np
from stacie import compute_spectrum,estimate_acint,LorentzModel
from ase.io.lammpsdata import read_lammps_data
BASE=Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project/results-scripts-logs')
CAMPAIGN=WORK/'audit/oct29/pbe15k_ti_temperature'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfg=read(CAMPAIGN/'protocol.json');historical=read(CAMPAIGN/'known_15k.json');baseline=read(CAMPAIGN/'baseline_5k.json')
out={'version':1,'purpose':'Find and validate historical 15k 20ps raw inputs before any pooled analysis','created_unix':time.time(),'production_protocol_sha256':sha(CAMPAIGN/'protocol.json'),'verified_new_checkpoint':cfg['checkpoint'],'verified_new_frozen_model':cfg['frozen_model'],'historical_exports':historical,'candidates':[],'pooling_performed':False,'compatibility_status':'not_established_missing_historical_15k_raw_and_model_binding'}
for T,host in [(1873,'wsl2'),(2000,'wsl2'),(2100,'cluster')]:
    state=BASE/f'models_check_{host}/results/viscosity/dpa4/neo/finetune/AlTi/{T}K/pbe'
    step=state/'03_stress_autocorr';raw=step/'c_Ti_1.00'
    files=sorted(raw.glob('visc*/pressure_*.data'),key=lambda f:int(f.parent.name[4:]))
    meta=read(step/'checkpoint.json');prep=read(state/'02_nvt_snapshots/checkpoint.json')
    seq=[];pref=[];manifest=[]
    for f in files:
        a=np.loadtxt(f);assert a.shape==(20001,7) and np.isfinite(a).all()
        assert np.array_equal(np.diff(a[:,0]),np.ones(20000))
        mol=f.parent/'mol.data';atoms=read_lammps_data(str(mol),atom_style='atomic',Z_of_type={1:13,2:22})
        assert len(atoms)==512 and np.all(atoms.numbers==22)
        vol=float(atoms.get_volume());p=a[:,1:]*1e5
        xx,yy,zz,xy,xz,yz=p.T
        seq.extend([(xx-.5*yy-.5*zz)/np.sqrt(3),.5*(yy-zz),xy,xz,yz]);pref.extend([vol*1e-30/(1.380649e-23*T)]*5)
        manifest.append({'pressure_file':str(f),'pressure_sha256':sha(f),'mol_file':str(mol),'mol_sha256':sha(mol),'volume_A3':vol,'first_step':float(a[0,0]),'last_step':float(a[-1,0])})
    fit=estimate_acint(compute_spectrum(np.array(seq),prefactors=np.array(pref),timestep=1e-15),LorentzModel(),verbose=False)
    ref=next(r for r in baseline if r['T']==T)
    if T==2000:ref=ref['previous_estimate']
    result={'eta_mPa_s':float(fit.acint*1000),'std_mPa_s':float(fit.acint_std*1000),'neff':float(fit.neff)}
    result['difference_from_5k_reference_mPa_s']=result['eta_mPa_s']-ref['eta_mPa_s']
    assert abs(result['difference_from_5k_reference_mPa_s'])<2e-5
    assert abs(result['std_mPa_s']-ref['eta_std_mPa_s'])<2e-5
    out['candidates'].append({'T':T,'host_archive':host,'n_replicas':len(files),'n_atoms':512,'composition':'pure Ti','n_samples':20001,'production_params':meta['_meta']['params_snapshot'],'production_completed_at':meta['cases']['c_Ti_1.00']['completed_at'],'preparation_params':prep['_meta']['params_snapshot'],'preparation_case':prep['cases']['c_Ti_1.00'],'fit':result,'matched_export_5k':ref,'manifest':manifest,'checkpoint_files':{str(f):sha(f) for f in [step/'checkpoint.json',state/'02_nvt_snapshots/checkpoint.json']},'interpretation':'Reproduces earlier 5k-labelled export; NOT the requested historical 15k dataset. Model hash is not recorded in these old metadata.'})
    (OUT/'provenance_audit.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('REPRODUCED',T,len(files),result,flush=True)
out['finished_unix']=time.time()
(OUT/'provenance_audit.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
