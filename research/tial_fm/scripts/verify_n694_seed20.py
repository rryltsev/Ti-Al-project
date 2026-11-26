"""Verify actual first-batch order and LoRA initialization differ from seed10."""
import json
import manage_expansion as m
from prepare_n694_repeats import RUN,REMOTE

def main():
    code='''from pathlib import Path
import hashlib,itertools,json
root=Path(REMOTE)
old=Path('/home6/u3129/tial-fm-research/dpa4-replicates-full-20261106-02/training-plan01/cases/PBE-lora16-N694-seed10')
new=root/'training-plan01/cases/PBE-lora16-N694-seed20'
def inventory(p):return json.loads((p/'parameter-inventory.json').read_text())['parameters']
a=inventory(old);b=inventory(new);assert set(a)==set(b)
adapters=[k for k in a if a[k]['adapter']]
changed=[k for k in adapters if a[k]['initial_sha256']!=b[k]['initial_sha256']]
base=[k for k in a if not a[k]['adapter']]
assert changed and all(a[k]['initial_sha256']==b[k]['initial_sha256'] for k in base)
def first(p):
    with (p/'training-exposures.jsonl').open() as f:return list(itertools.islice(f,100))
x=first(old);y=first(new);assert len(x)==len(y)==100 and x!=y
assert json.loads((new/'zero-step.json').read_text())['status']=='PASS'
print(json.dumps(dict(status='PASS',case='PBE-lora16-N694-seed20',initial_changed_adapter_tensors=changed,original_tensor_hashes_identical=True,first100_batch_stream_different=True,first100_seed10_sha256=hashlib.sha256(''.join(x).encode()).hexdigest(),first100_seed20_sha256=hashlib.sha256(''.join(y).encode()).hexdigest(),zero_step='PASS')))
'''.replace('REMOTE',repr(REMOTE))
    result=json.loads(m.remote_python(code));(RUN/'seed20-initial-check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:(len(v) if isinstance(v,list) else v) for k,v in result.items()}))

if __name__=='__main__':main()
