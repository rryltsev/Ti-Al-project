"""CPU-only checkpoint restoration, exact parameter mask and data-loader audit."""
from pathlib import Path
import hashlib,json,os,time
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
    start=time.monotonic();out=ROOT/'preflight';out.mkdir(exist_ok=False)
    import tensorflow as tf
    from tensorpotential import TensorPotential
    from tensorpotential.instructions.base import load_instructions
    from tensorpotential.cli.data import load_extxyz
    assert not tf.config.list_physical_devices('GPU')
    hashes=json.loads((ROOT/'FILES_SHA256.json').read_text())
    for name,digest in hashes.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    plan=json.loads((ROOT/'preparation-plan.json').read_text());source=ROOT/'checkpoint'
    listed=tf.train.list_variables(str(source/'checkpoint'))
    (out/'checkpoint-objects.json').write_text(json.dumps(listed,indent=2)+'\n')
    model=TensorPotential(potential=load_instructions(str(source/'model.yaml')),param_dtype=tf.float64,jit_compile=False)
    model.load_checkpoint(checkpoint_name=str(source/'checkpoint'),expect_partial=True,assert_existing_objects_matched=True,model_only=True)
    variables=list(model.model.variables);full={id(v) for v in model.model.variables_to_train}
    assert variables and full
    records=[]
    for i,v in enumerate(variables):
        records.append(dict(index=i,name=v.name,shape=v.shape.as_list(),dtype=v.dtype.name,elements=int(np.prod(v.shape)),full_trainable=id(v) in full,initial_sha256=hashlib.sha256(v.numpy().tobytes()).hexdigest()))
    patterns=plan['restricted_patterns'];model.set_trainable_variables(patterns,verbose=True)
    selected={id(v) for v in model.model.variables_to_train}
    expected={id(v) for v in variables if any(s in v.name for s in patterns)}
    assert selected==expected and selected and selected<full
    for p in patterns:assert any(p in v.name for v in variables),p
    for r,v in zip(records,variables):
        r['restricted_trainable']=id(v) in selected
        assert hashlib.sha256(v.numpy().tobytes()).hexdigest()==r['initial_sha256']
    flat=model.model.get_flat_trainable_variables()
    assert int(flat.shape[0])==sum(r['elements'] for r in records if r['restricted_trainable'])
    data=json.loads((ROOT/'data-preflight.json').read_text());loads={}
    for xc in ['PBE','r2SCAN']:
        for part in ['train','validation']:
            df=load_extxyz(str(ROOT/'data'/xc/(part+'.extxyz')))
            assert len(df)==data['data'][xc][part]['structures']
            assert [at.info['geometry_id'] for at in df['ase_atoms']]==data['data'][xc][part]['ids']
            assert all(np.isfinite(e) for e in df['energy'])
            assert all(np.isfinite(f).all() for f in df['forces'])
            assert all(np.isfinite(s).all() and np.shape(s)==(6,) for s in df['stress'])
            loads[f'{xc}/{part}']=len(df)
    result=dict(status='PASS',job_id=os.environ.get('SLURM_JOB_ID'),elapsed_seconds=time.monotonic()-start,
        full_trainable=sum(r['elements'] for r in records if r['full_trainable']),restricted_trainable=sum(r['elements'] for r in records if r['restricted_trainable']),
        total_variables_elements=sum(r['elements'] for r in records),restricted_patterns=patterns,parameters=records,datasets=loads,
        checkpoint_restore='model_only=True, expect_partial=True, assert_existing_objects_matched=True; all existing model tensors matched.',
        checkpoint_optimizer_keys=[n for n,s in listed if 'optimizer' in n.lower()],mask_preserves_all_parameter_values=True,
        training_steps=0,test_access=False,
        remaining_gates=['GRACE B0: derivatives/symmetry/type map/export consistency on actual probes','GRACE H0: original-model phonons on matched DFT cells','first optimizer-step frozen-parameter and fresh-optimizer checks; short GPU timing/memory profile'])
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['parameters','checkpoint_optimizer_keys']}))

if __name__=='__main__':main()
