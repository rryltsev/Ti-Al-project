"""Preserve raw Gamma artifacts and quantify the effect of ASR on DOS ranking."""
import json,os
import numpy as np
from ti3al_phonon_worker import ROOT,load_ph,band_points,mesh,mesh_info,dos_metrics,dump,sha
assert os.environ.get('SLURM_JOB_ID')
for n,h in json.loads((ROOT/'ASR_FILES_SHA256.json').read_text()).items():assert sha(ROOT/n)==h
ref=np.loadtxt(ROOT/'source/Ti3Al_D019/total_dos.dat');results={}
for c in json.loads((ROOT/'models.json').read_text()):
    name=c['name'];folder=ROOT/'analysis'/name
    original=json.loads((folder/'summary.json').read_text());assert original['status']=='PASS'
    p=load_ph();p.force_constants=np.load(folder/'force_constants_ASR.npy')
    p.run_qpoints(band_points());frequencies=p.qpoints.frequencies
    raw=np.load(folder/'bands.npz')['main']
    m=mesh(p);mi=mesh_info(m)
    p.run_total_dos(freq_min=-5,freq_max=20,freq_pitch=.01,use_tetrahedron_method=True)
    f=p.total_dos.frequency_points;d=p.total_dos.dos/8
    out=ROOT/'analysis-asr'/name;out.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(out/'dos.npz',frequency=f,dos=d)
    np.savez_compressed(out/'bands.npz',qpoints=band_points(),frequencies=frequencies)
    result=dict(status='PASS',raw_Gamma_THz=raw[0],ASR_Gamma_THz=frequencies[0],raw_band_min_THz=float(raw.min()),ASR_band_min_THz=float(frequencies.min()),raw_band_imaginary_count=int((raw<-.001).sum()),ASR_band_imaginary_count=int((frequencies<-.001).sum()),ASR_mesh=mi,ASR_DOS_metrics=dos_metrics(f,d,ref[:,0],ref[:,1]/8),force_constants_sha256=sha(folder/'force_constants_ASR.npy'))
    dump(out/'summary.json',result);results[name]=result
dump(ROOT/'analysis-asr/summary.json',dict(status='PASS',cases=results))
print('ASR sensitivity PASS',len(results))
