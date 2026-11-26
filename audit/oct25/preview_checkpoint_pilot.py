from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]/'audit/oct20/deps'))
import numpy as np
P=HERE/'pbe_checkpoint_sweep';rows=json.loads((P/'rows.json').read_text());refs=np.load(P/'labels-local-only.npz')
result={}
for key in ['5000','23400']:
    pred=np.load(P/f'pilot-{key}.npz');es=[];fs=[];ss=[]
    for r in rows:
        if r['split']!='test20':continue
        i=r['i'];e=1000*(float(pred[f'energy_{i}'])-float(refs[f'energy_{i}']))/r['n_atoms']
        es.append(e);fs.extend((pred[f'forces_{i}']-refs[f'forces_{i}']).ravel().tolist())
        if r['id']=='TiAl-test-009':e009=e
    result[key]={'n_test':len(es),'energy_RMSE_meV_atom':float(np.sqrt(np.mean(np.array(es)**2))),
                 'energy_MAE_meV_atom':float(np.mean(abs(np.array(es)))),'energy_bias_meV_atom':float(np.mean(es)),
                 'force_component_RMSE_eV_A':float(np.sqrt(np.mean(np.array(fs)**2))),
                 'Test009_error_meV_atom':e009}
(P/'pilot_test20_metrics.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
