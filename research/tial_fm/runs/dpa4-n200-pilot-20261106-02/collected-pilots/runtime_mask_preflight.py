"""CPU construction of the exact architecture and installed adapter policy; no updates."""
from pathlib import Path
import copy,hashlib,importlib.metadata,json,os
import torch
import deepmd.pt.train.training as training
import deepmd.pt.model.descriptor.sezm_nn.lora as lora
from deepmd.pt.model.model import get_model
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert os.environ.get('SLURM_JOB_ID') and not torch.cuda.is_available()
    out=ROOT/'runtime-mask-preflight';out.mkdir(exist_ok=False)
    source=json.loads((ROOT/'training-plan01/cases/PBE-lora16/input.json').read_text())['model'];source=copy.deepcopy(source);source.pop('lora')
    results={}
    for method in ['full','lora16']:
        m=get_model(copy.deepcopy(source))
        if method=='lora16':lora.apply_lora_to_sezm(m,rank=16,alpha=16)
        entries={n:dict(shape=list(p.shape),numel=p.numel(),trainable=p.requires_grad,adapter=any(x.startswith(('A_','B_')) for x in n.split('.'))) for n,p in m.named_parameters()}
        result=dict(total=sum(e['numel'] for e in entries.values()),trainable=sum(e['numel'] for e in entries.values() if e['trainable']),adapter_trainable=sum(e['numel'] for e in entries.values() if e['trainable'] and e['adapter']),parameters=entries)
        (out/(method+'.json')).write_text(json.dumps(result,indent=2)+'\n');results[method]={k:v for k,v in result.items() if k!='parameters'}
        del m
    h={'training.py':sha(training.__file__),'lora.py':sha(lora.__file__)}
    assert h['training.py']=='2ee355877da3ba10bf213b5653b3b858caa7795d846528c47ab1d7881dc3d090'
    assert h['lora.py']=='03591f701e266b9db0e395c53765391183dda38e9c8786d1d0a304a14c5a1644'
    summary=dict(status='PASS',results=results,backend_hashes=h,versions={n:importlib.metadata.version(n) for n in ['torch','deepmd-kit']},qualification='Constructed architecture from exact pilot model config using installed backend; zero optimizer steps. Actual Trainer mask is separately asserted and stored inside every GPU job.',job_id=os.environ['SLURM_JOB_ID'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
