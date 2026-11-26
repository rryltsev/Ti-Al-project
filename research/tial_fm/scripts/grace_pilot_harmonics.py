from pathlib import Path
import argparse,json
import numpy as np
from harmonic_worker import process
from grace_pilot_common import ROOT,dump
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case');a=ap.parse_args();work=ROOT/'cases'/a.case if a.case else ROOT/'baseline'
    out=work/'harmonics';out.mkdir(exist_ok=False);ref=np.load(ROOT/'reference_figure.npz');results={}
    for xc in ([a.case.split('-')[0]] if a.case else json.loads((ROOT/'plan.json').read_text()).get('active_XC',['PBE','r2SCAN'])):
        raw={k:np.load(work/'harmonic_forces'/xc/(k+'.npy')) for k in ['force000','force001','force002']}
        _,bands,info=process(ROOT/'harmonic_inputs'/xc,raw,out/xc,ref['qpoints'])
        results[xc]=dict(model=info,sorted_band_RMSE_to_DFT_THz=float(np.sqrt(np.mean((bands-ref[xc+'_DFT_bands'])**2))))
    dump(out/'summary.json',dict(status='PASS',results=results,scope='Fixed DFT cells, central +/-0.01A, same symfc/ASR and mesh50^3; baseline does not certify DFT convergence.'))
if __name__=='__main__':main()
