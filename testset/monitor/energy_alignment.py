"""Frozen elemental energy references calibrated exclusively on training data."""
from pathlib import Path
import hashlib, json
import numpy as np

ROOT=Path(__file__).resolve().parent
CAL=ROOT/'alignment/train-offsets-v1'
XC=['PBE','PBEsol','r2SCAN']
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))

def solve(rows,energies):
    assert rows and all(r['split']=='train' for r in rows)
    n=np.array([r['n_atoms'] for r in rows],float)
    counts=np.array([[r['n_Al'],r['n_Ti']] for r in rows],float)
    assert np.all(n>0) and np.array_equal(counts.sum(axis=1),n)
    x=counts/n[:,None]
    y=(np.array([r['energy_eV'] for r in rows])-np.array([energies[str(r['geometry_id'])] for r in rows]))/n
    assert np.isfinite(x).all() and np.isfinite(y).all()
    c,residual,rank,singular=np.linalg.lstsq(x,y,rcond=None)
    assert rank==2, 'Both elemental offsets must be identifiable'
    error=x@c-y
    return {'coefficients_eV_per_atom':dict(zip(['Al','Ti'],map(float,c))),
            'n_training_frames':len(rows),'rank':int(rank),'condition_number':float(singular[0]/singular[-1]),
            'train_raw_rmse_meV_atom':float(np.sqrt(np.mean(y*y))*1000),
            'train_aligned_rmse_meV_atom':float(np.sqrt(np.mean(error*error))*1000),
            'train_aligned_mae_meV_atom':float(np.mean(abs(error))*1000),
            'normal_equation_max_abs':float(abs(x.T@error).max())}

def apply(prediction,coefficients):
    assert prediction['energy_offset_applied_eV']==0
    assert set(prediction['symbols'])<=set(coefficients)
    offset=sum(coefficients[s] for s in prediction['symbols'])
    return dict(prediction,energy_eV=prediction['energy_eV']+offset,energy_offset_applied_eV=offset)

def load(model_hash):
    pin=read(ROOT/'alignment-registry.json'); path=ROOT/pin['path']
    assert sha(path)==pin['sha256'], 'Frozen training calibration changed'
    fit=read(path)
    assert fit['status']=='frozen' and fit['model_sha256']==model_hash
    assert sha(ROOT.parent/'TiAl-test50-v1/manifest.json')==fit['protocol']['test_geometry_manifest_sha256']
    for name,digest in fit['input_hashes'].items():
        assert sha(CAL/name)==digest, 'Calibration provenance changed: '+name
    assert set(fit['functionals'])==set(XC)
    return fit,pin

def main():
    assert not (CAL/'coefficients.json').exists(), 'Never silently refit a frozen calibration'
    audit=read(CAL/'input-audit.json'); protocol=read(CAL/'protocol.json')
    assert audit['status']=='passed'
    for name,digest in audit['source_files_sha256'].items():
        assert sha(name)==digest, 'Original training/split input changed: '+name
    for name,key in [('protocol.json','protocol_sha256'),('training-geometries.npz','geometries_sha256'),('training-labels.json','labels_sha256')]:
        assert sha(CAL/name)==audit[key]
    rows=read(CAL/'training-labels.json'); energies={}; outputs={}
    for task in range(8):
        path=CAL/f'result-{task}.json'; out=read(path)
        assert sha(path)==(CAL/f'result-{task}.sha256').read_text().strip()
        assert out['status']=='passed' and out['task']==task and out['chunks']==8
        assert out['model_sha256']==protocol['model_sha256'] and out['geometries_sha256']==audit['geometries_sha256']
        assert out['script_sha256']==sha(ROOT/'predict_training.py')
        assert all(int(i)%8==task for i in out['energies_eV'])
        assert not set(energies)&set(out['energies_eV'])
        energies.update(out['energies_eV']); outputs[path.name]=sha(path)
    assert set(energies)=={str(r['geometry_id']) for r in rows}
    assert len(energies)==audit['unique_training_geometries'] and len(rows)==audit['training_rows']
    for xc in XC:
        assert sum(r['functional']==xc for r in rows)==audit['datasets'][xc]['training_frames']
    fit={'status':'frozen','model_sha256':protocol['model_sha256'],'protocol':protocol,
         'fitting_script_sha256':sha(__file__),'preparation_script_sha256':sha(ROOT/'prepare_alignment.py'),
         'inference_script_sha256':sha(ROOT/'predict_training.py'),
         'functionals':{xc:solve([r for r in rows if r['functional']==xc],energies) for xc in XC},
         'input_hashes':{name:sha(CAL/name) for name in ['protocol.json','input-audit.json','training-labels.json','training-geometries.npz',*outputs]},
         'inference_jobs':[read(CAL/f'result-{i}.json')['job_id'] for i in range(8)],
         'uncertainty':'Test bootstrap intervals condition on these fixed coefficients; calibration uncertainty is not included.',
         'selection_caveat':'This analysis variant was requested after raw Test50 inspection; it is not a pre-registered blind model comparison.'}
    (CAL/'coefficients.json').write_text(json.dumps(fit,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    pin={'path':(CAL/'coefficients.json').relative_to(ROOT).as_posix(),'sha256':sha(CAL/'coefficients.json')}
    with (ROOT/'alignment-registry.json').open('x',encoding='utf-8') as stream: json.dump(pin,stream,indent=2)
    print(json.dumps(fit['functionals'],indent=2))

if __name__=='__main__': main()
