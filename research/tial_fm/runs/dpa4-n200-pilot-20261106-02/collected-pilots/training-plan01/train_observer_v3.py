"""Observe installed DeepMD training without changing its optimization algorithm."""
from pathlib import Path
import hashlib,json,os,time
import numpy as np
import torch
from deepmd.pt.train.training import Trainer
from deepmd.pt.model.descriptor.sezm_nn.lora import has_lora
from deepmd.infer import DeepPot

ROOT=Path(__file__).resolve().parent.parent;WORK=Path.cwd()
CFG=json.loads((WORK/'case.json').read_text());PROBES=json.loads((ROOT/'probes.json').read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def tensor_sha(p):return hashlib.sha256(p.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def adapter(n):return any(s.startswith(('A_','B_')) for s in n.split('.'))
def live(model,p):
    mode=model.training;model.eval();device=next(model.parameters()).device
    try:
        with torch.enable_grad():
            x=model(coord=torch.tensor([p['positions']],dtype=torch.float64,device=device),atype=torch.tensor([[model.get_type_map().index(s) for s in p['symbols']]],dtype=torch.int64,device=device),box=torch.tensor(np.array(p['cell']).reshape(1,9),dtype=torch.float64,device=device))
        return float(x['energy'].detach().cpu().ravel()[0]),x['force'].detach().cpu().numpy().reshape(-1,3),x['virial'].detach().cpu().numpy().reshape(3,3)
    finally:model.train(mode)
def pred(dp,p):
    e,f,v=dp.eval(np.array(p['positions']).reshape(1,-1),np.array(p['cell']).reshape(1,9),[dp.get_type_map().index(s) for s in p['symbols']])[:3]
    return float(e.ravel()[0]),f.reshape(-1,3),v.reshape(3,3)
def difference(a,b,n):return dict(energy_delta_eV_atom=(a[0]-b[0])/n,force_max_eV_A=float(np.max(np.abs(a[1]-b[1]))),virial_max_eV=float(np.max(np.abs(a[2]-b[2]))))
def assert_equivalent(d,energy=True):
    assert d['force_max_eV_A']<2e-4 and d['virial_max_eV']<.02,d
    if energy:assert abs(d['energy_delta_eV_atom'])<1e-5,d

original_init=Trainer.__init__;original_save=Trainer.save_model;original_batch=Trainer._next_training_batch
holders=[];inventory={};initial={};start=time.monotonic();training_start=None
exposures=dict(steps=0,frames=0,atoms=0);trace=None
def frozen_check(model):
    changed=[n for n,p in model.named_parameters() if not inventory[n]['trainable'] and tensor_sha(p)!=initial[n]]
    assert not changed,('Frozen parameters changed',changed)

def observed_init(self,*a,**kw):
    global training_start,trace
    original_init(self,*a,**kw);holders.append(self)
    m=self.wrapper.model['Default'];assert self._lora_enabled==(CFG['method']=='lora16')
    assert has_lora(m)==self._lora_enabled
    for n,p in m.named_parameters():
        initial[n]=tensor_sha(p);inventory[n]=dict(trainable=p.requires_grad,numel=p.numel(),shape=list(p.shape),initial_sha256=initial[n],adapter=adapter(n))
    expected=json.loads((ROOT/'runtime-mask-preflight'/(CFG['method']+'.json')).read_text())['parameters']
    assert {n:{k:v[k] for k in ['trainable','numel','shape','adapter']} for n,v in inventory.items()}==expected,'Actual Trainer mask changed from audited preflight'
    if self._lora_enabled:
        bs=[p for n,p in m.named_parameters() if any(x.startswith('B_') for x in n.split('.'))]
        assert bs and all(torch.count_nonzero(p).item()==0 for p in bs)
    dump(WORK/'parameter-inventory.json',dict(parameters=inventory,total=sum(v['numel'] for v in inventory.values()),trainable=sum(v['numel'] for v in inventory.values() if v['trainable']),adapter_trainable=sum(v['numel'] for v in inventory.values() if v['trainable'] and v['adapter']),frozen=sum(v['numel'] for v in inventory.values() if not v['trainable']),num_steps=self.num_steps,optimizer_type=type(self.optimizer).__name__,seed=10))
    (WORK/'merged').mkdir();(WORK/'checks').mkdir()
    self.save_model_merged(WORK/'initial-merged.pt')
    source=DeepPot(CFG['source']['path']);merged=DeepPot(str(WORK/'initial-merged.pt'));zero=[];composition=[];deltas=[]
    for p in PROBES:
        a=live(m,p);b=pred(merged,p);c=pred(source,p);export=difference(a,b,len(p['symbols']));assert_equivalent(export)
        restore=difference(b,c,len(p['symbols']));assert_equivalent(restore,energy=False)
        frac=p['symbols'].count('Al')/len(p['symbols']);composition.append([frac,1-frac]);deltas.append(restore['energy_delta_eV_atom'])
        zero.append(dict(id=p['id'],live_vs_merged=export,initialized_vs_source=restore))
    # The backend deliberately fits species energy biases on the N200 train only.
    coeff=np.linalg.lstsq(composition,deltas,rcond=None)[0];residual=float(np.max(np.abs(np.array(composition)@coeff-deltas)));assert residual<1e-5,residual
    dump(WORK/'zero-step.json',dict(status='PASS',checks=zero,energy_shift_Al_Ti_eV=coeff.tolist(),energy_shift_noncomposition_residual_eV_atom=residual,source_sha256=sha(CFG['source']['path']),initial_merged_sha256=sha(WORK/'initial-merged.pt'),test_access=False))
    del source,merged;frozen_check(m)
    trace=(WORK/'training-exposures.jsonl').open('w');training_start=time.monotonic();torch.cuda.reset_peak_memory_stats()

def observed_batch(self,*a,**kw):
    result=original_batch(self,*a,**kw);inputs,labels,log=result
    frames=int(inputs['coord'].shape[0]);atoms=int(inputs['atype'].shape[-1])
    exposures['steps']+=1;exposures['frames']+=frames;exposures['atoms']+=frames*atoms
    def simple(v):
        if hasattr(v,'detach'):return v.detach().cpu().tolist()
        if hasattr(v,'tolist'):return v.tolist()
        return v
    trace.write(json.dumps(dict(step=exposures['steps'],frames=frames,natoms=atoms,log={k:simple(v) for k,v in log.items()}))+'\n')
    if exposures['steps']%100==0:
        trace.flush();dump(WORK/'progress.json',dict(**exposures,training_seconds=time.monotonic()-training_start,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),job_id=os.environ['SLURM_JOB_ID']))
    return result

def observed_save(self,path,lr=0.,step=0,**kw):
    m=self.wrapper.model['Default'];frozen_check(m);original_save(self,path,lr=lr,step=step,**kw)
    merged=WORK/'merged'/f'{step+1:05d}.pt';self.save_model_merged(merged,lr=lr,step=step)
    dp=DeepPot(str(merged));checks=[]
    for p in PROBES:
        d=difference(live(m,p),pred(dp,p),len(p['symbols']));assert_equivalent(d);checks.append(dict(id=p['id'],**d))
    dump(WORK/'checks'/f'{step+1:05d}.json',dict(status='PASS',step=step+1,checkpoint=str(path),checkpoint_sha256=sha(path),merged=str(merged),merged_sha256=sha(merged),live_vs_merged=checks,exposures=dict(exposures),seconds=time.monotonic()-start))

def main():
    assert os.environ.get('SLURM_JOB_ID') and torch.cuda.is_available()
    assert sha(WORK/'input.json')==CFG['configuration_sha256']
    assert sha(CFG['source']['path'])==CFG['source']['sha256']
    import importlib
    assert sha(importlib.import_module('deepmd.pt.train.training').__file__)=='2ee355877da3ba10bf213b5653b3b858caa7795d846528c47ab1d7881dc3d090'
    assert sha(importlib.import_module('deepmd.pt.model.descriptor.sezm_nn.lora').__file__)=='03591f701e266b9db0e395c53765391183dda38e9c8786d1d0a304a14c5a1644'
    assert json.loads((ROOT/'trainer-zero-preflight/summary.json').read_text())['status']=='PASS'
    for gate in ['baseline','harmonics']:assert json.loads((ROOT/gate/'summary.json').read_text())['status']=='PASS'
    Trainer.__init__=observed_init;Trainer.save_model=observed_save;Trainer._next_training_batch=observed_batch
    from deepmd.main import main as dp_main
    try:dp_main()
    except SystemExit as e:
        if e.code not in (None,0):raise
    assert len(holders)==1;self=holders[0];m=self.wrapper.model['Default'];frozen_check(m);trace.close()
    assert exposures['steps']==self.num_steps
    changes=[n for n,p in m.named_parameters() if tensor_sha(p)!=initial[n]]
    assert changes
    if self._lora_enabled:assert any(inventory[n]['adapter'] for n in changes)
    dump(WORK/'parameter-update-check.json',dict(status='PASS',frozen_changed=[],changed_parameters=changes,final_sha256={n:tensor_sha(p) for n,p in m.named_parameters()}))
    dump(WORK/'training-summary.json',dict(status='PASS',case=CFG,steps=self.num_steps,exposures=exposures,total_seconds=time.monotonic()-start,training_seconds=time.monotonic()-training_start,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),gpu=torch.cuda.get_device_name(),job_id=os.environ['SLURM_JOB_ID'],final_checkpoint=str(WORK/'merged'/f'{self.num_steps:05d}.pt'),test_access=False))
if __name__=='__main__':main()
