"""Native CPU evaluation of paired EOS and archived liquid snapshots."""
from pathlib import Path
import os,json,hashlib,time,importlib.metadata
import numpy as np
from scipy.optimize import brentq
from ase.build import bulk
from ase.units import GPa
from ase.io import write
from deepmd.infer import DeepPot
import torch
ROOT=Path.cwd();cfg=json.loads((ROOT/'protocol.json').read_text())
assert os.environ.get('SLURM_JOB_ID') and not os.environ.get('CUDA_VISIBLE_DEVICES')
key=['5000','15000'][int(os.environ['SLURM_ARRAY_TASK_ID'])]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
model=cfg['models'][key];assert sha(model['path'])==model['sha256']
assert sha('geometries.npz')==cfg['geometry_sha256'] and sha('rows.json')==cfg['rows_sha256']
state=torch.load(model['path'],map_location='cpu',weights_only=True)
info=state['model']['_extra_state']['train_infos'];assert info['step']==int(key)-1;del state
out=ROOT/key;out.mkdir(exist_ok=False);start=time.monotonic();times=[]
dp=DeepPot(model['path']);tm=dp.get_type_map()
def evaluate(at):
    ts=time.monotonic();types=[tm.index(s) for s in at.get_chemical_symbols()]
    e,f,v=dp.eval(at.positions.reshape(1,-1),at.cell.array.reshape(1,9),types)[:3]
    times.append({'n_atoms':len(at),'seconds':time.monotonic()-ts})
    e=float(e.reshape(-1)[0]);f=f.reshape(-1,3);v=v.reshape(3,3)
    assert np.isfinite(e) and np.isfinite(f).all() and np.isfinite(v).all()
    return {'E_eV_atom':e/len(at),'P_GPa':float(np.trace(v)/(3*at.get_volume())/GPa),
            'stress_GPa':(-(v+v.T)/(2*at.get_volume())/GPa).tolist(),
            'Fmax_eV_A':float(np.linalg.norm(f,axis=1).max()),'forces':f,'virial':v,'energy':e}
def structure(phase,V,ratio=1.58):
    if phase=='Ti_hcp':
        a=(4*V/(np.sqrt(3)*ratio))**(1/3);return bulk('Ti','hcp',a=a,c=ratio*a)
    if phase=='Ti_bcc':return bulk('Ti','bcc',a=(2*V)**(1/3),cubic=True)
    return bulk('Al','fcc',a=(4*V)**(1/3),cubic=True)
def point(phase,V):
    ratio=None
    if phase=='Ti_hcp':
        def dev(r):
            s=np.array(evaluate(structure(phase,V,r))['stress_GPa'])
            return s[2,2]-(s[0,0]+s[1,1])/2
        ratio=brentq(dev,*cfg['eos']['hcp_c_over_a_bracket'],xtol=1e-8)
    at=structure(phase,V,ratio if ratio else 1.58);d=evaluate(at)
    s=np.array(d['stress_GPa']);assert np.max(abs(s-np.eye(3)*np.trace(s)/3))<.005
    assert d['Fmax_eV_A']<1e-4
    return at,{'V_A3_atom':V,'E_eV_atom':d['E_eV_atom'],'P_GPa':d['P_GPa'],
               'c_over_a':ratio,'cell_A':at.cell.array.tolist(),'Fmax_eV_A':d['Fmax_eV_A']}
eos={}
for phase in ['Ti_hcp','Ti_bcc','Al_fcc']:
    ref=cfg['eos'][phase+'_reference_V_A3_atom'];volumes=ref*np.array(cfg['eos']['volume_factors'])
    curve=[point(phase,float(v))[1] for v in volumes]
    v0=brentq(lambda v:point(phase,v)[1]['P_GPa'],volumes[0],volumes[-1],xtol=1e-7)
    at,eq=point(phase,v0);write(out/(phase+'-POSCAR'),at,format='vasp',direct=True)
    hm=.001*v0;lo=point(phase,v0-hm)[1];hi=point(phase,v0+hm)[1]
    eq['B_GPa']=-v0*(hi['P_GPa']-lo['P_GPa'])/(2*hm)
    hm2=.0005*v0;lo2=point(phase,v0-hm2)[1];hi2=point(phase,v0+hm2)[1]
    eq['B_halfstep_GPa']=-v0*(hi2['P_GPa']-lo2['P_GPa'])/(2*hm2)
    assert eq['B_GPa']>0 and abs(eq['B_GPa']-eq['B_halfstep_GPa'])<.2
    # Check analytic virial sign and units against an energy derivative away from P=0.
    vt=ref*.96;test=point(phase,vt)[1];fd_checks=[]
    for relative_h in [1e-4,5e-4,1e-3,2e-3]:
        h=vt*relative_h
        p_fd=-(point(phase,vt+h)[1]['E_eV_atom']-point(phase,vt-h)[1]['E_eV_atom'])/(2*h)/GPa
        fd_checks.append({'relative_volume_step':relative_h,'pressure_FD_GPa':p_fd,
                          'difference_GPa':p_fd-test['P_GPa']})
    # The three larger increments must agree with analytic pressure: the smallest
    # increment can amplify model energy rounding. Keep every check in the record.
    assert max(abs(d['difference_GPa']) for d in fd_checks[1:])<.02,(phase,fd_checks,test['P_GPa'])
    small=evaluate(at);rep=evaluate(at.repeat((3,3,3)))
    check={'delta_E_meV_atom':1000*(rep['E_eV_atom']-small['E_eV_atom']),
           'delta_P_GPa':rep['P_GPa']-small['P_GPa'],'pressure_FD_GPa':p_fd,
           'pressure_virial_GPa':test['P_GPa'],'pressure_FD_step_check':fd_checks}
    assert abs(check['delta_E_meV_atom'])<.02 and abs(check['delta_P_GPa'])<.005
    eos[phase]={'curve':curve,'equilibrium':eq,'verification':check}
    (out/'eos.json').write_text(json.dumps(eos,indent=2));print('EOS',key,phase,eq,flush=True)
from ase import Atoms
geo=np.load('geometries.npz',allow_pickle=False);rows=json.loads(Path('rows.json').read_text());arrays={};liquid=[]
for r in rows:
    i=r['i'];at=Atoms(symbols=[['Al','Ti'][t] for t in geo[f'types_{i}']],positions=geo[f'coords_{i}'],cell=geo[f'cell_{i}'],pbc=True)
    d=evaluate(at)
    for k in ['forces','virial','energy']:arrays[f'{k}_{i}']=d[k]
    liquid.append({**r,**{k:d[k] for k in ['E_eV_atom','P_GPa','Fmax_eV_A']}})
    if i%25==0:print('LIQUID',key,i+1,len(rows),flush=True)
np.savez_compressed(out/'predictions.npz',**arrays)
(out/'liquid.json').write_text(json.dumps(liquid,indent=2))
summary={'status':'passed','checkpoint':key,'model_sha256':model['sha256'],'train_infos':info,
         'protocol_sha256':sha('protocol.json'),'script_sha256':sha(__file__),
         'job_id':os.environ['SLURM_JOB_ID'],'node':os.uname().nodename,
         'wall_seconds':time.monotonic()-start,'inference_timings':times,
         'versions':{p:importlib.metadata.version(p) for p in ['numpy','scipy','ase','torch','deepmd-kit']}}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print('COMPLETE',key,summary['wall_seconds'],flush=True)
