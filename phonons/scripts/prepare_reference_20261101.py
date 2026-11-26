"""Prepare immutable force000 policies/drivers based on validated predecessors."""
from pathlib import Path
import json
from cluster import ROOT
p=ROOT/'scripts'
for source,num,xc in [('06','09','PBE'),('08','10','r2SCAN')]:
    cfg=json.loads((ROOT/'config'/('execution-policy-'+source+'.json')).read_text())
    cfg.update(revision='production-06',steps=['force-000'],reason=xc+' second displacement independently accepted 2026-11-01; undisplaced reference for residual diagnostics. Scientific settings unchanged.')
    if xc=='r2SCAN':cfg.update(seed_job_id='20972837',seed_source='r2SCAN/production-05/force-002')
    (ROOT/'config'/('execution-policy-'+num+'.json')).write_text(json.dumps(cfg,indent=2)+'\n')
s=(p/'production_step_v6.py').read_text().replace('execution-policy-06.json','execution-policy-09.json')
s=s.replace("        if 'aborting loop because EDIFF is reached' not in text:","        if 'General timing and accounting informations' not in text: raise RuntimeError('No normal termination')\n        if 'aborting loop because EDIFF is reached' not in text:")
(p/'production_step_v9.py').write_text(s)
s=(p/'production_step_v8.py').read_text().replace('execution-policy-08.json','execution-policy-10.json')
s=s.replace("step!='force-002'","step!='force-000'").replace("recovery-01/force-001/POSCAR","recovery-01/force-002/POSCAR")
s=s.replace("abs(norms[0]-.02)","abs(norms[0]-.01)").replace('Expected opposite .01A displacement of atom1','Expected return of atom1 from displaced to reference geometry')
s=s.replace("or not np.allclose(density.atoms[0].positions,atoms.positions,atol=3e-5,rtol=0)","or np.max(np.abs(((density.atoms[0].get_scaled_positions(wrap=False)-atoms.get_scaled_positions(wrap=False)+.5)%1-.5)@atoms.cell.array))>3e-5")
s=s.replace('Opposite displacement from frozen phonopy inputs; same scientific settings','Undisplaced reference from frozen phonopy inputs; same scientific settings')
s=s.replace('converged first displacement as initial guess','converged second displacement as initial guess')
(p/'production_step_v10.py').write_text(s)
s=(p/'launch_v13.sbatch').read_text().replace('production_step_v8.py)', 'production_step_v8.py|production_step_v9.py|production_step_v10.py)')
(p/'launch_v14.sbatch').write_text(s)
s=(p/'archive_superseded_wavecar_lustre_20261031.py').read_text()
s=s.replace('force-001-restart-03','force-001-restart-04')
s=s.replace("root/'r2SCAN/production-02/force-001-restart-04/restart-provenance.json'", "root/'r2SCAN/production-05/force-002/restart-provenance.json'")
s=s.replace('to main-home archive','to Lustre archive')
(p/'archive_first_r2scan_wavecar_20261101.py').write_text(s)
