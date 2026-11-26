"""Test whether large imaginary branches survive without force/ASR corrections."""
from pathlib import Path
import json,os
import numpy as np
import phonopy
ROOT=Path(__file__).resolve().parent;out=ROOT/'harmonics';results={}
for xc in ['PBE','PBEsol','r2SCAN']:
    results[xc]={}
    for method in ['DFT','original']:
        folder=out/xc/method;p=phonopy.load(folder/'phonopy_params.yaml',produce_fc=False,symmetrize_fc=False,log_level=0);values={}
        for stage,file in [('raw','force_constants_raw.npy'),('central','force_constants_central.npy'),('ASR','force_constants_ASR.npy')]:
            p.force_constants=np.load(folder/file);mesh=p.run_mesh([50,50,50],is_gamma_center=True)
            idx=np.unravel_index(np.argmin(mesh.frequencies),mesh.frequencies.shape)
            values[stage]=dict(mesh_min_THz=float(mesh.frequencies[idx]),q_at_min=mesh.qpoints[idx[0]].tolist(),branch_at_min=int(idx[1]),imaginary_weighted=int(np.sum(mesh.weights[:,None]*(mesh.frequencies<-.001))),force_constants_max_abs_eV_A2=float(np.max(np.abs(p.force_constants))))
        results[xc][method]=values
    assert results[xc]['original']['raw']['mesh_min_THz']<-.5
    assert results[xc]['original']['central']['mesh_min_THz']<-.5
record=dict(status='PASS',job_id=os.environ['SLURM_JOB_ID'],results=results,conclusion='The original-model imaginary branches remain large before mean-drift/central-pair/ASR processing; projection did not create them. Small DFT raw near-Gamma residuals are separately reported, not hidden.')
(out/'projection_diagnostics.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
