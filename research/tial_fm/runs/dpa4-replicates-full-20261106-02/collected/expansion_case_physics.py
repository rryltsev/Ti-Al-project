"""Additional mode/FC/acoustic diagnostics for one completed expansion case."""
from pathlib import Path
import argparse,hashlib,json,os
import numpy as np
from analyze_pilot_physics import inspect,mode_comparison

ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);a=ap.parse_args()
    assert os.environ.get('SLURM_JOB_ID')
    p=ROOT/'training-plan01/cases'/a.case;c=json.loads((p/'case.json').read_text());xc=c['XC']
    folder=p/'evaluation/harmonics';reference=ROOT/'harmonics'/xc/'DFT'
    out=p/'evaluation/physical-diagnostics';out.mkdir(exist_ok=False)
    r=inspect(reference);m=inspect(folder);bands=np.load(folder/'bands.npz');ref=np.load(reference/'bands.npz')
    comparison,indices=mode_comparison(ref,bands)
    comparison['relative_FC_Frobenius_error']=float(np.linalg.norm(np.load(folder/'force_constants_ASR.npy')-np.load(reference/'force_constants_ASR.npy'))/np.linalg.norm(np.load(reference/'force_constants_ASR.npy')))
    comparison['singlet_eigenvalue_ratio_to_DFT']=(m['Gamma']['singlet_c_THz']/r['Gamma']['singlet_c_THz'])**2
    np.savez_compressed(out/'mode_matching.npz',indices=indices,qpoints=ref['qpoints'],frequencies_matched=bands['frequencies'][np.arange(len(indices))[:,None],indices])
    source=folder/'force_constants_ASR.npy'
    result=dict(status='PASS',job_id=os.environ['SLURM_JOB_ID'],case=c['name'],reference=r,model=m,comparison=comparison,FC_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),qualification='Physical development on fixed XC-specific DFT cell. Small-q convergence is not DFT/supercell convergence.')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('PHYSICAL_DIAGNOSTICS_PASS',c['name'])

if __name__=='__main__':main()
