"""Check the exported potentials through LAMMPS run 0 on a fixed geometry."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
import numpy as np
from ase import Atoms
from ase.io import write
from ase.units import bar

assert os.environ.get('SLURM_JOB_ID')
source = Path(sys.argv[1])
registry = json.loads((source/'model_registry.json').read_text())
bundle = json.loads((source/'all50-geometries.json').read_text())
r = next(r for r in bundle['records'] if set(r['symbols']) == {'Al','Ti'})
cell=np.asarray(r['cell_A']);assert np.max(np.abs(cell-np.diag(np.diag(cell))))<1e-10
atoms=Atoms(symbols=r['symbols'],scaled_positions=r['fractional_positions'],cell=cell,pbc=True)
package=Path(sys.prefix)/'lib/python3.12/site-packages'
binary=package/'lammps/lmp';plugin=package/'deepmd/lib/libdeepmd_lmpplugin.so'
result={'status':'running','job_id':os.environ['SLURM_JOB_ID'],'structure_id':r['metadata']['structure_id'],
        'models':{},'binary':str(binary),'plugin':str(plugin),'unix':time.time()}
def save():Path('result.json').write_text(json.dumps(result,indent=2)+'\n')
try:
    for xc, model in registry['models'].items():
        assert hashlib.sha256(Path(model['path']).read_bytes()).hexdigest()==model['sha256']
        folder=Path(xc);folder.mkdir()
        write(folder/'atoms.data',atoms,format='lammps-data',atom_style='atomic',specorder=['Al','Ti'])
        text=f'''plugin load {plugin}
units metal
atom_style atomic
atom_modify map yes
boundary p p p
read_data atoms.data
mass 1 26.9815385
mass 2 47.867
newton on
pair_style deepmd {model['path']}
pair_coeff * * Al Ti
neighbor 1.0 bin
neigh_modify every 1 delay 0 check yes
thermo_style custom step pe pxx pyy pzz pyz pxz pxy
thermo_modify format float %.17g
run 0
print "$(pe:%.17g) $(pxx:%.17g) $(pyy:%.17g) $(pzz:%.17g) $(pyz:%.17g) $(pxz:%.17g) $(pxy:%.17g)" file prediction.txt
write_dump all custom forces.dump id type x y z fx fy fz modify sort id format float %.17g
'''
        (folder/'in.lammps').write_text(text)
        with (folder/'console.log').open('w') as log:
            subprocess.run([str(binary),'-in','in.lammps'],cwd=folder,env=dict(os.environ,LAMMPS_PLUGIN_PATH=''),
                           stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
        output=np.loadtxt(folder/'prediction.txt')
        dump=(folder/'forces.dump').read_text().splitlines();start=next(i for i,s in enumerate(dump) if s.startswith('ITEM: ATOMS'))
        raw=np.array([[float(v) for v in line.split()] for line in dump[start+1:]])
        assert np.array_equal(raw[:,0],np.arange(1,len(atoms)+1))
        ref=json.loads((source/'predictions'/xc/(r['metadata']['structure_id']+'.json')).read_text())
        ref_s=-np.asarray(ref['virial_eV']).reshape(9)[[0,4,8,5,2,1]]/atoms.get_volume()
        errors={'energy_eV_atom':float(abs(output[0]-ref['energy_eV'])/len(atoms)),
                'force_max_eV_A':float(np.max(np.abs(raw[:,-3:]-ref['forces_eV_A']))),
                'stress_max_eV_A3':float(np.max(np.abs(-output[1:]*bar-ref_s)))}
        result['models'][xc]=dict(errors,model_sha256=model['sha256'])
        save()
        assert errors['energy_eV_atom']<1e-6 and errors['force_max_eV_A']<1e-4 and errors['stress_max_eV_A3']<1e-5,errors
        result['models'][xc]['status']='passed';save()
        print(xc,'LAMMPS passed',flush=True)
    result['status']='passed';save()
except Exception:
    result.update(status='failed',traceback=traceback.format_exc());save();raise
Path('outputs.json').write_text(json.dumps({p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
    for p in Path('.').rglob('*') if p.is_file() and
    (p.name in ['result.json','console.log','in.lammps','prediction.txt','forces.dump','atoms.data'])},indent=2)+'\n')
