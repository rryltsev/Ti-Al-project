from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent.parent;WORK=Path.cwd();sys.path.insert(0,str(ROOT))
from harmonic_worker import process,dump
def main():
    s=json.loads((WORK/'training-summary.json').read_text());xc=s['case']['XC'];out=WORK/'evaluation/harmonics'
    raw={k:np.load(WORK/'evaluation/harmonic_forces'/k/'forces.npy') for k in ['force000','force001','force002']}
    ref=np.load(ROOT/'reference_figure.npz');p,mf,ms=process(ROOT/'harmonic_inputs'/xc,raw,out,ref['qpoints'])
    df=ref[xc+'_DFT_bands'];original=np.load(ROOT/'harmonics'/xc/'original/bands.npz')['frequencies']
    dump(out/'summary.json',dict(status='PASS',XC=xc,method=s['case']['method'],model=ms,sorted_band_RMSE_THz=float(np.sqrt(np.mean((mf-df)**2))),sorted_band_max_error_THz=float(np.max(np.abs(mf-df))),original_sorted_band_RMSE_THz=float(np.sqrt(np.mean((original-df)**2))),qualification='Matched geometry physical development diagnostic; not independent test for tuning.'))
if __name__=='__main__':main()
