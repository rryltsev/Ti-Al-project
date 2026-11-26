from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'audit/oct20/deps'))
import numpy as np
p=HERE/'pbe_property_screen';rows=json.loads((p/'rows.json').read_text())
a=np.load(p/'archived_predictions.npz');b=np.load(p/'5000/predictions.npz');d={}
for x in [0,.1,.3,.5,1]:
    rr=[r for r in rows if r['x_Al']==x]
    de=[1000*(float(b['energy_'+str(r['i'])])-float(a['energy_'+str(r['i'])]))/200 for r in rr]
    df=np.concatenate([b['forces_'+str(r['i'])]-a['forces_'+str(r['i'])] for r in rr])
    d[x]={'max_E_difference_meV_atom':float(np.max(np.abs(de))),
          'force_difference_RMS_eV_A':float(np.sqrt(np.mean(df**2)))}
(p/'archive_verification.json').write_text(json.dumps(d,indent=2));print(json.dumps(d,indent=2))
