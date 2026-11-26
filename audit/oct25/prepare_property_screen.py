"""Freeze paired CPU property screening inputs; never modify archived trajectories."""
from pathlib import Path
import sys,json,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'audit/oct20/deps'))
import numpy as np
from ase.io import read
from ase.units import GPa
OUT=HERE/'pbe_property_screen';OUT.mkdir(exist_ok=True)
assert not (OUT/'protocol.json').exists()
SOURCE=Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project')
BASE=SOURCE/'results-scripts-logs/models_check_wsl2/results/mixing_enthalpy/dpa4/neo/finetune/AlTi/2000K/pbe'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
geo={};cached={};rows=[];sources=[]
for name,x in [('Ti',0.),('Al0.1Ti0.9',.1),('Al0.3Ti0.7',.3),('Al0.5Ti0.5',.5),('Al',1.)]:
    path=BASE/(name+'_2000K.traj');frames=read(path,index=':')
    assert len(frames)==301,(name,len(frames))
    sources.append({'path':str(path),'sha256':sha(path),'n_frames':len(frames)})
    for j in range(6,301,6):
        at=frames[j];i=len(rows);symbols=at.get_chemical_symbols();n=len(at)
        assert n==200 and abs(symbols.count('Al')/n-x)<1e-9 and all(at.pbc)
        geo[f'coords_{i}']=at.positions;geo[f'cell_{i}']=at.cell.array
        geo[f'types_{i}']=np.array([['Al','Ti'].index(s) for s in symbols],dtype='i4')
        row={'i':i,'composition':name,'x_Al':x,'frame':j,'time_ps_from_first_frame':j*.1,
             'n_atoms':n,'V_A3_atom':at.get_volume()/n,'temperature_K':at.get_temperature()}
        results=at.calc.results if at.calc else {}
        row['stored_properties']=list(results)
        for k in ['energy','forces','stress']:
            if k in results:cached[f'{k}_{i}']=np.array(results[k])
        rows.append(row)
np.savez_compressed(OUT/'geometries.npz',**geo)
np.savez_compressed(OUT/'archived_predictions.npz',**cached)
(OUT/'rows.json').write_text(json.dumps(rows,indent=2))
models=json.loads((HERE/'pbe_checkpoint_sweep/remote-input.json').read_text())['models']
cfg={'id':'pbe-property-screen-20261025-01','models':{k:models[k] for k in ['5000','15000']},
     'geometry_sha256':sha(OUT/'geometries.npz'),'rows_sha256':sha(OUT/'rows.json'),
     'archive_predictions_sha256':sha(OUT/'archived_predictions.npz'),'sources':sources,
     'liquid':{'T_K':2000,'n_atoms':200,'compositions_x_Al':[0,.1,.3,.5,1],
               'frames_per_composition':50,'spacing_ps':.6,'total_frames':len(rows)},
     'eos':{'Ti_hcp_reference_V_A3_atom':17.5,'Ti_bcc_reference_V_A3_atom':17.5,
            'Al_fcc_reference_V_A3_atom':16.6,'volume_factors':np.linspace(.92,1.08,11).tolist(),
            'hcp_c_over_a_bracket':[1.45,1.75],'hcp_rule':'zero deviatoric stress at fixed volume',
            'cubic_rule':'fixed cubic symmetry','equilibrium_rule':'zero hydrostatic pressure',
            'bulk_modulus_rule':'central finite difference of pressure on shape-relaxed E(V)'},
     'pressure_convention':'P_config = trace(virial)/(3V), positive compression; excludes kinetic term',
     'policy':'Native checkpoints, CPU only; archived inputs and manuscript unchanged. Liquid differences are paired on the archived PBE ensemble, not DFT errors or new 15k equilibrium MD.'}
(OUT/'protocol.json').write_text(json.dumps(cfg,indent=2))
print(json.dumps({'frames':len(rows),'bytes':(OUT/'geometries.npz').stat().st_size,'stored_properties':rows[0]['stored_properties']}))
