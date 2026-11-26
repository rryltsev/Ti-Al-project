"""Check order in the five beta-initialized temperature trajectories; no MD."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'audit/oct20/deps'))
sys.path.insert(0,str(ROOT/'audit/melting'))
import numpy as np
from ase.io.trajectory import Trajectory
from analyze_melting import order,ideal_check,THRESHOLDS
sys.stdout.reconfigure(encoding='utf-8')
RUN=ROOT/'audit/oct29/pbe15k_ti_density_temperature';out=RUN/'analysis'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result={'method':'Existing 14-nearest-neighbor Lechner-Dellago qbar6 and coherent bonds; orthogonal periodic cells; every 1 ps. A structural diagnostic, not equilibrium melting or an exact phase fraction.',
        'script_sha256':sha(Path(__file__)),'reference_script_sha256':sha(ROOT/'audit/melting/analyze_melting.py'),
        'ideal_bcc_qbar6':ideal_check(),'thresholds':THRESHOLDS.tolist(),'points':{}}
for T in [1200,1400,1600,1800,1900]:
    state=f'beta-{T}';data={}
    for stage in ['equilibration','production']:
        path=RUN/'collected/results'/state/(stage+'.traj');frames=[]
        with Trajectory(str(path)) as tr:
            for index in range(0,len(tr),10):
                a=tr[index];cell=a.cell.array;lengths=np.diag(cell)
                np.testing.assert_allclose(cell,np.diag(lengths),rtol=0,atol=1e-10)
                xyz=a.positions%lengths
                q,qbar,bonds=order(xyz,lengths)
                frames.append(dict(time_ps=index*.1,qbar6_mean=float(qbar.mean()),qbar6_median=float(np.median(qbar)),
                    fractions=[float(np.mean(qbar>th)) for th in THRESHOLDS],coherent_fraction=float(np.mean(bonds>=7))))
        late=[r for r in frames if r['time_ps']>frames[-1]['time_ps']-10]
        data[stage]={'source':str(path),'sha256':sha(path),'frames':frames,
            'last10_fractions':np.mean([r['fractions'] for r in late],axis=0).tolist(),
            'last10_coherent_fraction':float(np.mean([r['coherent_fraction'] for r in late])),
            'last10_qbar6_mean':float(np.mean([r['qbar6_mean'] for r in late]))}
    result['points'][state]=data
    print(state,'production last10 solid-like fractions',data['production']['last10_fractions'],
        'coherent',data['production']['last10_coherent_fraction'],flush=True)
(out/'beta_structural_qc.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
