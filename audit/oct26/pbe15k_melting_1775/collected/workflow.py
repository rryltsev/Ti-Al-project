"""Validated native LAMMPS coexistence scan; no automatic phase interpretation."""
from pathlib import Path
import json,os,sys,time,hashlib,traceback,importlib.metadata
import numpy as np
ROOT=Path(__file__).resolve().parent;CFG=json.loads((ROOT/'protocol.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,data):
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)
def check_device():
 import torch
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert torch.cuda.is_available() and torch.cuda.device_count()==1
 assert 'A100' in torch.cuda.get_device_name(0)
 return {'name':torch.cuda.get_device_name(0),'visible':os.environ.get('CUDA_VISIBLE_DEVICES'),'job':os.environ['SLURM_JOB_ID']}
def make_lammps(data,folder):
 from lammps import lammps
 import deepmd
 L=lammps(cmdargs=['-log',str(folder/'log.lammps'),'-screen',str(folder/'screen.txt')])
 if not L.has_style('pair','deepmd'):
  plugin=Path(deepmd.__file__).parent/'lib/libdeepmd_lmpplugin.so';L.command('plugin load '+str(plugin))
 assert L.has_style('pair','deepmd')
 commands=f'''units metal
boundary p p p
atom_style atomic
atom_modify map yes
read_data {data} nocoeff
pair_style deepmd {ROOT}/model_15000.pt2
pair_coeff * * Ti
neighbor 2.0 bin
neigh_modify delay 10 check yes
timestep {CFG['dt_ps']}
compute 2 all pe/atom
thermo {CFG['thermo_every']}
thermo_style custom step time atoms temp pe ke etotal press pxx pyy pzz vol lx ly lz
thermo_modify flush yes lost error
'''
 (folder/'setup.in').write_text(commands);L.commands_string(commands);return L
def export():
 import torch
 device=check_device();checkpoint=CFG['checkpoint'];assert sha(checkpoint['path'])==checkpoint['sha256']
 meta=torch.load(checkpoint['path'],map_location='cpu',weights_only=True)['model']['_extra_state']
 assert meta['train_infos']['step']==14999
 for name,m in CFG['inputs'].items():assert sha(ROOT/name)==m['sha256']
 from deepmd.pt.entrypoints.freeze_pt2 import freeze_sezm_to_pt2
 target=ROOT/'model_15000.pt2';assert not target.exists()
 save(ROOT/'export-started.json',{'device':device,'unix':time.time(),'checkpoint_sha256':checkpoint['sha256']})
 freeze_sezm_to_pt2(checkpoint['path'],str(target),device=torch.device('cuda:0'),atomic_virial=True)
 save(ROOT/'export.json',{'status':'exported','device':device,'unix':time.time(),'checkpoint_sha256':checkpoint['sha256'],'pt2_sha256':sha(target)})
def preflight():
 import torch
 from ase.io import read
 from deepmd.infer import DeepPot
 device=check_device();assert sha(CFG['checkpoint']['path'])==CFG['checkpoint']['sha256'];assert sha(ROOT/'model_15000.pt2')==CFG['frozen_model']['sha256'];dp=DeepPot(CFG['checkpoint']['path']);checks=[]
 for i,name in enumerate(CFG['inputs']):
  folder=ROOT/'preflight'/str(i);folder.mkdir(parents=True,exist_ok=False)
  a=read(ROOT/name,format='lammps-data',atom_style='atomic',Z_of_type={1:22});assert len(a)==CFG['atoms']
  e,f,v=dp.eval(a.positions.reshape(1,-1),a.cell.array.reshape(1,9),[dp.get_type_map().index('Ti')]*len(a))[:3]
  L=make_lammps(ROOT/name,folder);L.command('velocity all set 0 0 0');L.command('run 0')
  ff=np.ctypeslib.as_array(L.gather_atoms('f',1,3),shape=(len(a)*3,)).copy().reshape(-1,3)
  pressure=np.array([L.get_thermo(k) for k in ['pxx','pyy','pzz','pxy','pxz','pyz']])
  vv=v.reshape(3,3);ref=vv.flat[[0,4,8,1,2,5]]/a.get_volume()*1602176.6208
  record={'input':name,'E_diff_eV_atom':abs(L.get_thermo('pe')-float(e.reshape(-1)[0]))/len(a),
   'F_diff_rms_eV_A':float(np.sqrt(np.mean((ff-f.reshape(-1,3))**2))),
   'F_diff_max_eV_A':float(np.max(abs(ff-f.reshape(-1,3)))),
   'pressure_diff_max_GPa':float(np.max(abs(pressure-ref)))/10000}
  checks.append(record);save(ROOT/'preflight-progress.json',checks)
  assert record['E_diff_eV_atom']<1e-5 and record['F_diff_rms_eV_A']<1e-4 and record['pressure_diff_max_GPa']<.002,record
  if i==1:
   L.commands_string('velocity all create 1800 4928459 dist gaussian\nfix bench all npt temp 1800 1800 0.1 aniso 0 0 1.0\nrun 10')
   t=time.monotonic();L.command('run 100');seconds=time.monotonic()-t
  L.close()
 projected=seconds/100*sum(s['steps'] for s in CFG['stages'])*max(len(v) for v in CFG['workers'].values())/3600
 out={'status':'passed','device':device,'checks':checks,'benchmark_100_steps_seconds':seconds,'projected_worker_hours':projected,
  'checkpoint_sha256':CFG['checkpoint']['sha256'],'pt2_sha256':sha(ROOT/'model_15000.pt2'),'protocol_sha256':sha(ROOT/'protocol.json'),
  'versions':{k:importlib.metadata.version(k) for k in ['deepmd-kit','torch','numpy','ase']}}
 assert projected<CFG['allocation_wall_hours']*.8,out
 save(ROOT/'preflight.json',out);print(json.dumps(out),flush=True)
def worker(index):
 device=check_device();pre=json.loads((ROOT/'preflight.json').read_text());assert pre['status']=='passed' and pre['pt2_sha256']==sha(ROOT/'model_15000.pt2')
 completed=[]
 for T in CFG['workers'][index]:
  case=ROOT/f'T{T}K';case.mkdir(exist_ok=False);data=ROOT/'initial.data'
  try:
   for stage in CFG['stages']:
    name=stage['name'];folder=case/name;folder.mkdir(exist_ok=False)
    save(ROOT/f'progress-{index}.json',{'status':'running','temperature_K':T,'stage':name,'completed_temperatures':completed,'unix':time.time()})
    L=make_lammps(data,folder)
    common='''variable zc equal (zlo+zhi)/2
region liquid block INF INF INF INF INF ${zc} units box
group liquid region liquid
group solid subtract all liquid
compute LiquidTemp liquid temp
compute SolidTemp solid temp
thermo_style custom step time atoms temp pe ke etotal press pxx pyy pzz vol lx ly lz c_LiquidTemp c_SolidTemp
thermo_modify flush yes lost error
'''
    if name=='equil':common+='min_style cg\nminimize 1e-10 1e-10 5000 10000\n'
    common+='reset_timestep 0\n'
    if name=='equil':common+=f"velocity all create {T} {CFG['seed']} dist gaussian\n"
    group='liquid' if name in ['heat','cool'] else 'all'
    start=T*1.5 if name=='cool' else T;end=T*1.5 if name=='heat' else T
    common+=f'''fix integrate {group} npt temp {start} {end} {CFG['Tdamp_ps']} aniso 0 0 {CFG['Pdamp_ps']}
dump trajectory all custom {CFG['dump_every']} {folder}/trajectory.lammpstrj id type x y z c_2 vx vy vz
dump_modify trajectory sort id
restart {CFG['restart_every']} {folder}/restart.*
run {stage['steps']}
write_data {folder}/final.data
write_restart {folder}/final.restart
'''
    (folder/'run.in').write_text(common);started=time.monotonic();L.commands_string(common)
    result={'status':'completed','temperature_K':T,'stage':name,'steps':stage['steps'],'final_T_K':L.get_thermo('temp'),'final_volume_A3':L.get_thermo('vol'),'wall_seconds':time.monotonic()-started,'device':device}
    assert L.get_thermo('atoms')==CFG['atoms'] and np.isfinite(L.get_thermo('pe'))
    L.close();save(folder/'summary.json',result);data=folder/'final.data';print('COMPLETED_STAGE',T,name,flush=True)
   completed.append(T);save(case/'summary.json',{'status':'completed','temperature_K':T,'checkpoint_sha256':CFG['checkpoint']['sha256'],'protocol_sha256':sha(ROOT/'protocol.json'),'pt2_sha256':sha(ROOT/'model_15000.pt2')})
  except Exception:
   save(case/'failure.json',{'traceback':traceback.format_exc()});raise
 save(ROOT/f'progress-{index}.json',{'status':'completed','completed_temperatures':completed,'unix':time.time()})
if __name__=='__main__':
 try:
  if sys.argv[1]=='export':export()
  elif sys.argv[1]=='preflight':preflight()
  else:worker(sys.argv[2])
 except Exception:
  save(ROOT/('failure-'+('-'.join(sys.argv[1:]))+'.json'),{'traceback':traceback.format_exc(),'unix':time.time()});raise
