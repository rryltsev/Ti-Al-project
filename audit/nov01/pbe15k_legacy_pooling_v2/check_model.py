from pathlib import Path
import sys,json,shlex,time
sys.path.insert(0,'audit/oct29/pbe15k_ti_temperature')
import cluster as c
p=Path('audit/nov01/pbe15k_legacy_pooling_v2')
data=json.loads((p/'model_input.json').read_text())
remote="""import io,json,numpy as np,time,torch
from ase.io.lammpsdata import read_lammps_data
from deepmd.infer import DeepPot
torch.set_num_threads(2)
dp=DeepPot(MODEL)
idx=dp.get_type_map().index('Ti')
rows=[]
for item in DATA:
 a=read_lammps_data(io.StringIO(item['mol_text']),atom_style='atomic',Z_of_type={1:13,2:22})
 e,f,v=dp.eval(a.positions.reshape(1,-1),a.cell.array.reshape(1,9),[idx]*512)[:3]
 kin=a.get_momenta().T @ a.get_velocities()
 pred=(v.reshape(3,3)+kin).flat[[0,4,8,1,2,5]]/a.get_volume()*1602176.6208
 diff=pred-np.array(item['pressure_bar'])
 row=dict(T=item['T'],replica=item['replica'],max_abs_pressure_error_bar=float(np.max(abs(diff))),rms_pressure_error_bar=float(np.sqrt(np.mean(diff**2))),energy_eV=float(e.flat[0]),predicted_pressure_bar=pred.tolist(),reference_pressure_bar=item['pressure_bar'])
 rows.append(row)
print(json.dumps(dict(checks=rows,max_error_bar=max(x['max_abs_pressure_error_bar'] for x in rows),threshold_bar=20,passed=all(x['max_abs_pressure_error_bar']<20 for x in rows))))
""".replace('MODEL',repr(c.CFG['checkpoint']['path'])).replace('DATA',repr(data))
command='CUDA_VISIBLE_DEVICES= DEVICE=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 '+c.PYTHON+' -B -'
result=c.run(command,content=remote.encode(),timeout=600)
(p/'model_response.txt').write_text(result)
result=json.loads(result)
result.update(checkpoint=c.CFG['checkpoint'],finished_unix=time.time(),method='Native checkpoint evaluates all 45 initial coordinates; kinetic pressure from stored velocities; compare all six initial pressure components.')
(p/'model_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='checks'}))

