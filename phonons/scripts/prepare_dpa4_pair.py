"""Prepare exact DFT displacement pairs and immutable CPU inference controller."""
from pathlib import Path
import json,shutil
import cluster as c

for xc,restart in [('PBE','force-001-restart-05'),('r2SCAN','force-001-restart-04')]:
    dest=c.ROOT/'results'/(xc+'-DFT-pair-inputs-20261101')
    dest.mkdir(exist_ok=False)
    snap=c.RECORD/'snapshot'/xc/'recovery-01'
    shutil.copyfile(snap/'phonopy_disp.yaml',dest/'phonopy_disp.yaml')
    for key,revision,step,day in [('force001','production-02',restart,'20261031'),('force002','production-05','force-002','20261101')]:
        folder=dest/key;folder.mkdir()
        for name in ['POSCAR','INCAR','KPOINTS','forces.npy']:
            shutil.copyfile(c.RECORD/'collected'/xc/revision/step/name,folder/name)
        shutil.copyfile(c.ROOT/'results'/(xc+'-'+key+'-validation-'+day+'.json'),folder/'audit.json')
    (dest/'force000').mkdir()
    shutil.copyfile(snap/'force-000/POSCAR',dest/'force000/POSCAR')
    (dest/'reference-status.json').write_text(json.dumps(dict(calculated=False,
        note='Reference geometry only. No synthetic reference forces. Central pair cancels any common reference exactly. Residual force/stress diagnostics pending.'),indent=2))

source=(c.ROOT/'scripts/dpa4_matched_cluster.py').read_text()
source=source.replace("LOCAL=c.RECORD/'DPA4-PBEsol-matched-20261031-01'", "XC=sys.argv[2]\nassert XC in ['PBE','r2SCAN']\nLOCAL=c.RECORD/('DPA4-'+XC+'-matched-20261101-01')")
source=source.replace("REMOTE=c.REMOTE+'/DPA4-PBEsol-matched-20261031-01'", "REMOTE=c.REMOTE+'/'+LOCAL.name")
source=source.replace("DFT=c.ROOT/'results/PBEsol-phonons-preliminary-20261031-02/inputs'", "DFT=c.ROOT/'results'/(XC+'-DFT-pair-inputs-20261101')")
source=source.replace("model=models['PBEsol-23100']", "model=models['PBE-5000' if XC=='PBE' else 'r2SCAN-23200']")
source=source.replace("geometry='exact DFT-PBEsol POSCARs; no relaxation'", "geometry='exact DFT-'+XC+' POSCARs; no relaxation'")
source=source.replace("'--job-name=ti-DPA4-matched'", "'--job-name=ti-DPA4-'+XC")
(c.ROOT/'scripts/dpa4_pair_cluster.py').write_text(source)
