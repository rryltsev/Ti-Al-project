"""Initialize each exact N200 Trainer on CPU and stop before the optimizer loop."""
from pathlib import Path
import copy,hashlib,importlib.util,json,os,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parent
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def child():
    import torch
    from deepmd.pt.train.training import Trainer
    from deepmd.infer import DeepPot
    spec=importlib.util.spec_from_file_location('pilot_observer',ROOT/'training-plan01/train_observer_v2.py');obs=importlib.util.module_from_spec(spec);spec.loader.exec_module(obs)
    work=Path.cwd();case=json.loads((work/'case.json').read_text());original_init=Trainer.__init__
    class Initialized(Exception):pass
    def check_init(self,*args,**kwargs):
        original_init(self,*args,**kwargs);m=self.wrapper.model['Default']
        expected=json.loads((ROOT/'runtime-mask-preflight'/(case['method']+'.json')).read_text())
        entries={n:dict(shape=list(p.shape),numel=p.numel(),trainable=p.requires_grad,adapter=obs.adapter(n)) for n,p in m.named_parameters()}
        assert entries==expected['parameters'],'Trainer mask differs from architecture preflight'
        self.save_model_merged(work/'initial-merged.pt')
        source=DeepPot(case['source']['path']);merged=DeepPot(str(work/'initial-merged.pt'));checks=[];x=[];y=[]
        for p in json.loads((ROOT/'probes.json').read_text()):
            a=obs.live(m,p);b=obs.pred(merged,p);c=obs.pred(source,p);d=obs.difference(a,b,len(p['symbols']));obs.assert_equivalent(d)
            restore=obs.difference(b,c,len(p['symbols']));obs.assert_equivalent(restore,energy=False)
            frac=p['symbols'].count('Al')/len(p['symbols']);x.append([frac,1-frac]);y.append(restore['energy_delta_eV_atom']);checks.append(dict(id=p['id'],live_vs_merged=d,source_restoration=restore))
        offset=np.linalg.lstsq(x,y,rcond=None)[0];residual=float(np.max(np.abs(np.array(x)@offset-y)));assert residual<1e-5
        dump(work/'summary.json',dict(status='PASS',method=case['method'],XC=case['XC'],checks=checks,mask_equals_preflight=True,total=expected['total'],trainable=expected['trainable'],energy_offset_Al_Ti_eV=offset.tolist(),composition_offset_residual_eV_atom=residual,optimizer_steps_executed=0,qualification='Exact training configuration except one-step schedule used for initialization; execution stopped immediately after Trainer initialization. GPU runs start anew with original50epoch schedule and independent statistics.'))
        raise Initialized()
    Trainer.__init__=check_init
    from deepmd.main import main
    try:main()
    except Initialized:print('INITIALIZATION_PASS_NO_OPTIMIZER_STEPS',flush=True)
    else:raise RuntimeError('Expected initialization-only stop')
def main():
    assert os.environ.get('SLURM_JOB_ID') and not os.environ.get('CUDA_VISIBLE_DEVICES')
    if '--pt' in sys.argv:return child()
    out=ROOT/'trainer-zero-preflight';out.mkdir(exist_ok=False);summaries=[]
    for name in ['PBE-lora16','PBE-full','r2SCAN-lora16','r2SCAN-full']:
        src=ROOT/'training-plan01/cases'/name;dest=out/name;dest.mkdir();cfg=json.loads((src/'input.json').read_text());cfg['training'].pop('num_epochs');cfg['training']['numb_steps']=1
        dump(dest/'input.json',cfg);(dest/'case.json').write_bytes((src/'case.json').read_bytes())
        with (dest/'initialization.log').open('w') as f:subprocess.run([sys.executable,__file__,'--pt','train','input.json','--finetune',json.loads((dest/'case.json').read_text())['source']['path']],cwd=dest,stdout=f,stderr=subprocess.STDOUT,check=True)
        summaries.append(json.loads((dest/'summary.json').read_text()));print('COMPLETE',name,flush=True)
    dump(out/'summary.json',dict(status='PASS',cases=summaries,optimizer_steps_executed=0,job_id=os.environ['SLURM_JOB_ID']))
if __name__=='__main__':main()
