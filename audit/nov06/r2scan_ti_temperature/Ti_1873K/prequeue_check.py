"""CPU pre-submission audit of archived model identity and spectral reproduction."""
import io,json,time,subprocess,sys
import numpy as np
import analysis_common as a
import pooled_analysis as pool
from ase.io.lammpsdata import read_lammps_data
from deepmd.infer import DeepPot
import torch
assert all(a.sha(a.ROOT/name)==expected for name,expected in json.loads((a.ROOT/'staging-manifest.json').read_text()).items())
for key in ['checkpoint','frozen_model']:
    assert a.sha(a.CFG[key]['path'])==a.CFG[key]['sha256']
step=torch.load(a.CFG['checkpoint']['path'],map_location='cpu',weights_only=True)['model']['_extra_state']['train_infos']['step']
assert step==a.CFG['checkpoint']['step']-1
subprocess.run([sys.executable,'-B',str(a.ROOT/'test_analysis.py')],check=True)
clusters,manifest=pool.legacy_clusters()
out,r,uc=pool.combine(clusters,len(clusters),0)
reference=manifest['published']
assert abs(out['eta_mPa_s']-reference['eta_mPa_s'])<2e-5,(out,reference)
assert abs(out['formal_std_mPa_s']-reference['eta_std_mPa_s'])<2e-5,(out,reference)
dp=DeepPot(a.CFG['checkpoint']['path']);idx=dp.get_type_map().index('Ti');torch.set_num_threads(2)
checks=[]
for item in manifest['raw_manifest']:
    path=a.ROOT/item['mol_file'];assert a.sha(path)==item['mol_sha256']
    atoms=read_lammps_data(io.StringIO(path.read_text()),atom_style='atomic',Z_of_type={1:13,2:22})
    assert len(atoms)==512 and set(atoms.get_chemical_symbols())=={'Ti'}
    e,f,v=dp.eval(atoms.positions.reshape(1,-1),atoms.cell.array.reshape(1,9),[idx]*512)[:3]
    kinetic=atoms.get_momenta().T@atoms.get_velocities()
    pred=(v.reshape(3,3)+kinetic).flat[[0,4,8,1,2,5]]/atoms.get_volume()*1602176.6208
    err=float(np.max(abs(pred-np.array(item['first_pressure_bar']))))
    checks.append(dict(replica=item['replica'],max_abs_pressure_error_bar=err))
limit=a.CFG['pooling']['required_model_pressure_max_error_bar'];maximum=max(c['max_abs_pressure_error_bar'] for c in checks)
result=dict(status='PASS' if maximum<limit else 'FAIL',unix=time.time(),protocol_sha256=a.sha(a.ROOT/'protocol.json'),
    legacy_manifest_sha256=a.CFG['legacy']['sha256'],checkpoint_sha256=a.CFG['checkpoint']['sha256'],
    frozen_sha256=a.CFG['frozen_model']['sha256'],checkpoint_internal_step=step,baseline=out,
    baseline_reference=reference,model_pressure_checks=checks,max_error_bar=maximum,threshold_bar=limit)
a.save(a.ROOT/'prequeue-validation.json',result)
print('PREQUEUE_CHECK',json.dumps({k:v for k,v in result.items() if k not in ['model_pressure_checks','baseline_reference','baseline']}),flush=True)
assert maximum<limit,'Archived pressures do not match requested model; no submission authorized until reconciled'
