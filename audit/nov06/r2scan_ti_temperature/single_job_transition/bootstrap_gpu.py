"""Device-specific export of the same native checkpoint, before any new MD."""
from pathlib import Path
import os,sys,json,time,hashlib,traceback,copy
ROOT=Path(__file__).resolve().parent
START=float(os.environ.setdefault('SERIAL_ALLOCATION_START_UNIX',str(time.time())))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path,obj):
 path=Path(path);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');tmp.replace(path)
def main():
 assert os.environ['SLURM_JOB_ID']=='20988268'
 cfg=json.loads((ROOT/'protocol.json').read_text());assert sha(cfg['checkpoint']['path'])==cfg['checkpoint']['sha256']
 import torch
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert torch.cuda.is_available() and torch.cuda.device_count()==1 and 'A100' in torch.cuda.get_device_name(0)
 meta=torch.load(cfg['checkpoint']['path'],map_location='cpu',weights_only=True)['model']['_extra_state'];assert meta['train_infos']['step']==23199
 record=ROOT/'gpu-export.json'
 if record.exists():
  export=json.loads(record.read_text());assert export['status']=='passed' and export['checkpoint_sha256']==cfg['checkpoint']['sha256'] and sha(export['path'])==export['sha256']
 else:
  for sid in cfg['states']:assert not list((ROOT/sid/'states'/sid/'replicas').glob('replica_*/summary.json'))
  parent=ROOT/'gpu_export';parent.mkdir(exist_ok=True);i=1
  while (parent/('attempt_%04d'%i)).exists():i+=1
  folder=parent/('attempt_%04d'%i);folder.mkdir();target=folder/'model_23200_cuda.pt2'
  save(ROOT/'progress.json',dict(phase='gpu_export',job_id=cfg['job_id'],checkpoint_sha256=cfg['checkpoint']['sha256'],device=torch.cuda.get_device_name(0),unix=time.time()))
  from deepmd.pt.entrypoints.freeze_pt2 import freeze_sezm_to_pt2
  freeze_sezm_to_pt2(cfg['checkpoint']['path'],str(target),device=torch.device('cuda:0'),atomic_virial=True)
  from deepmd.infer import DeepPot
  from ase.io import read
  import numpy as np
  before=DeepPot(cfg['checkpoint']['path']);after=DeepPot(str(target));assert before.get_type_map()==after.get_type_map();idx=before.get_type_map().index('Ti');checks=[]
  for sid in cfg['states']:
   atoms=read(ROOT/sid/'initial.data',format='lammps-data',atom_style='atomic',Z_of_type={1:13,2:22});assert len(atoms)==512 and set(atoms.get_chemical_symbols())=={'Ti'}
   args=(atoms.positions.reshape(1,-1),atoms.cell.array.reshape(1,9),[idx]*512)
   old=before.eval(*args)[:3];new=after.eval(*args)[:3]
   e=float(np.max(abs(old[0]-new[0])))/512;f=float(np.max(abs(old[1]-new[1])));v=float(np.max(abs(old[2]-new[2])))
   assert e<1e-5 and f<1e-4 and v<.01,(e,f,v)
   checks.append(dict(id=sid,E_max_eV_atom=e,F_max_eV_A=f,virial_max_eV=v))
  export=dict(status='passed',unix=time.time(),job_id=cfg['job_id'],device=torch.cuda.get_device_name(0),path=str(target),sha256=sha(target),checkpoint_sha256=cfg['checkpoint']['sha256'],checkpoint_internal_step=23199,atomic_virial=True,equivalence=checks,method='Same freeze_sezm_to_pt2(cuda:0,atomic_virial=True) as previously validated PBE15k A100 workflow; unchanged native weights')
  save(record,export)
 frozen=dict(path=export['path'],sha256=export['sha256'])
 if cfg['frozen_model']!=frozen:
  h=json.loads((ROOT/'staging-manifest.json').read_text());assert all(sha(ROOT/n)==expected for n,expected in h.items())
  reg=json.loads((ROOT/'registry.json').read_text());changes=[]
  for sid in cfg['states']:
   assert not list((ROOT/sid/'states'/sid/'replicas').glob('replica_*/summary.json'))
   path=ROOT/sid/'protocol.json';old_sha=sha(path);private=json.loads(path.read_text());old_frozen=private['frozen_model'];private['frozen_model']=frozen
   save(path,private);new_sha=sha(path);cfg['states'][sid]['protocol_sha256']=new_sha;reg['runs'][sid]['protocol_sha256']=new_sha;h[sid+'/protocol.json']=new_sha
   changes.append(dict(id=sid,old_protocol_sha256=old_sha,new_protocol_sha256=new_sha,old_frozen=old_frozen,new_frozen=frozen))
  cfg['frozen_model']=frozen;reg['frozen_model']=frozen;save(ROOT/'protocol.json',cfg);save(ROOT/'registry.json',reg)
  h['protocol.json']=sha(ROOT/'protocol.json');h['registry.json']=sha(ROOT/'registry.json');h[Path(export['path']).relative_to(ROOT).as_posix()]=export['sha256']
  save(ROOT/'staging-manifest.json',h);save(ROOT/'gpu-model-binding.json',dict(unix=time.time(),native_checkpoint_unchanged=True,physical_protocol_unchanged=True,changes=changes,freeze_device='cuda:0',gpu_export_sha256=sha(record)))
 os.execv(sys.executable,[sys.executable,'-B','-u',str(ROOT/'workflow.py')])
if __name__=='__main__':
 try:main()
 except Exception:
  save(ROOT/'failure.json',dict(kind='gpu_export_or_binding',unix=time.time(),job_id=os.environ.get('SLURM_JOB_ID'),traceback=traceback.format_exc()));raise
