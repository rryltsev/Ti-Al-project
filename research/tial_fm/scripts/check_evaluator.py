"""Known physical perturbations check energy offsets, force pooling and virial sign/units."""
from pathlib import Path
import json,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];PACKAGE=ROOT/'runs/dpa4-n200-pilot-20261106-02/package';OUT=ROOT/'runs/evaluator-analytic-check-20261106-01';GPA=160.21766208
def main():
    OUT.mkdir(exist_ok=False)
    for part in ['train','validation','test']:
        dst=OUT/'predictions/PBE'/part;dst.mkdir(parents=True)
        for src in sorted((PACKAGE/'data/PBE'/part).iterdir()):
            ids=json.loads((src/'geometry_ids.json').read_text());types=np.loadtxt(src/'type.raw',dtype=int,ndmin=1);names=(src/'type_map.raw').read_text().split();n=len(types)
            nAl=int(np.count_nonzero(types==names.index('Al')));nTi=n-nAl
            energy=np.load(src/'set.000/energy.npy').ravel()+nAl-2*nTi
            force=np.load(src/'set.000/force.npy').reshape(-1,n,3)+.1
            box=np.load(src/'set.000/box.npy').reshape(-1,3,3);volume=np.linalg.det(box)
            virial=np.load(src/'set.000/virial.npy').reshape(-1,3,3)-volume[:,None,None]/GPA*np.ones((1,3,3))
            np.savez_compressed(dst/(src.name+'.npz'),ids=ids,energy=energy,force=force,virial=virial)
    subprocess.run([sys.executable,str(ROOT/'scripts/analyze_predictions.py'),'--package',str(PACKAGE),'--predictions',str(OUT/'predictions'),'--output',str(OUT/'analysis')],check=True,stdout=subprocess.DEVNULL)
    x=json.loads((OUT/'analysis/metrics.json').read_text())['PBE'];assert np.max(np.abs(np.array(x['alignment']['offset_eV'])-[-1,2]))<1e-12
    for part in ['train','validation','test']:
        m=x['metrics'][part];assert m['E_aligned_meV_atom']['RMSE']<1e-8
        assert abs(m['F_component_eV_A']['RMSE']-.1)<1e-12
        assert abs(m['F_norm_error_eV_A']['RMSE']-np.sqrt(3)*.1)<1e-12
        assert abs(m['stress_component_GPa']['bias']-1)<1e-12
        assert abs(m['stress_component_GPa']['RMSE']-1)<1e-12
    (OUT/'verification.json').write_text(json.dumps(dict(status='PASS',energy_recovered_offsets_Al_Ti_eV=[-1,2],force_component_RMSE_eV_A=.1,force_vector_norm_RMSE_eV_A=float(np.sqrt(3)*.1),stress_tension_error_GPa=1,scope='Analytic synthetic perturbations of existing labels; these are evaluator tests, not model predictions.'),indent=2)+'\n')
    print('PASS: energy offsets, per-atom normalization, component vs vector forces, stress sign and GPa conversion')
if __name__=='__main__':main()
