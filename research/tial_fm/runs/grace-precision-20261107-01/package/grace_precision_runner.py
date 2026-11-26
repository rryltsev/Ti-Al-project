"""Paired ABBA order, independent model processes on the same allocated hardware."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent

def audit():
    import tensorflow as tf
    from tensorpotential import TensorPotential
    from tensorpotential.instructions.base import load_instructions
    plan=json.loads((ROOT/'plan.json').read_text())
    if plan['checkpoints']['fp32'].get('source_kind')=='controlled_cast':
        source=Path(plan['checkpoints']['fp64']['path']);dest=Path(plan['checkpoints']['fp32']['path'])
        assert not (dest/'checkpoint.index').exists()
        model=TensorPotential(potential=load_instructions(str(dest/'model.yaml')),param_dtype=tf.float32,jit_compile=False)
        template_dir=ROOT/'cast-template';template_dir.mkdir(exist_ok=False)
        template_prefix=tf.train.Checkpoint(model=model.model).write(str(template_dir/'checkpoint'))
        template=tf.train.load_checkpoint(template_prefix);original=tf.train.load_checkpoint(str(source/'checkpoint'))
        names=sorted(template.get_variable_to_shape_map());dtypes=template.get_variable_to_dtype_map();tensors=[]
        for name in names:
            if name.startswith('model/'):
                assert name in original.get_variable_to_shape_map(),name
                value=original.get_tensor(name)
                if dtypes[name]!=tf.string:value=np.asarray(value).astype(dtypes[name].as_numpy_dtype)
            else:value=template.get_tensor(name)
            tensors.append(tf.convert_to_tensor(value,dtype=dtypes[name]))
        tf.raw_ops.SaveV2(prefix=str(dest/'checkpoint'),tensor_names=names,shape_and_slices=['']*len(names),tensors=tensors)
        model.load_checkpoint(checkpoint_name=str(dest/'checkpoint'),expect_partial=True,assert_existing_objects_matched=True,model_only=True)
        assert {v.dtype for v in model.model.variables_to_train}=={tf.float32}
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.glob('checkpoint.*')}
        (ROOT/'controlled-cast.json').write_text(json.dumps(dict(status='PASS',source=plan['checkpoints']['fp64'],derived_checkpoint_sha256=hashes,method='SaveV2 using native fp32 template object graph/dtypes; model tensors cast from source; no optimizer or training.'),indent=2)+'\n')
    readers={p:tf.train.load_checkpoint(str(Path(c['path'])/'checkpoint')) for p,c in plan['checkpoints'].items()}
    names={p:{n for n in r.get_variable_to_shape_map() if n.startswith('model/')} for p,r in readers.items()};assert names['fp32']==names['fp64']
    result=[]
    for name in sorted(names['fp64']):
        if not name.startswith('model/'):continue
        x=np.asarray(readers['fp64'].get_tensor(name));y=np.asarray(readers['fp32'].get_tensor(name));assert x.shape==y.shape
        matched=np.array_equal(x.astype(y.dtype),y);assert matched,name
        result.append(dict(name=name,shape=list(x.shape),fp64_dtype=str(x.dtype),fp32_dtype=str(y.dtype),cast_exact=True))
    assert result
    output=dict(status='PASS',source_kind=plan['checkpoints']['fp32'].get('source_kind','official_fp32'),same_model_tensors_after_dtype_cast=True,model_tensor_count=len(result),tensors=result)
    (ROOT/'weight-equivalence.json').write_text(json.dumps(output,indent=2)+'\n');print('SOURCE_WEIGHTS_CAST_EXACT',len(result),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',choices=['cpu','gpu'],required=True);a=ap.parse_args()
    if a.device=='cpu':audit()
    else:assert json.loads((ROOT/'weight-equivalence.json').read_text())['status']=='PASS'
    folder=ROOT/'results'/a.device;folder.mkdir(parents=True,exist_ok=True)
    info=subprocess.run(['lscpu'],capture_output=True,text=True,check=True).stdout
    (folder/'cpu-hardware.txt').write_text(info)
    if a.device=='gpu':
        (folder/'gpu-hardware.txt').write_text(subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv'],text=True))
    jobs=[]
    for precision,block in [('fp32',0),('fp64',0),('fp64',1),('fp32',1)]:
        cmd=[sys.executable,str(ROOT/'grace_precision_worker.py'),'--device',a.device,'--precision',precision,'--block',str(block)]
        begin=time.monotonic()
        with (folder/f'{precision}-block{block}.log').open('w') as log:
            try:p=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=600 if a.device=='cpu' else 900);code=p.returncode
            except subprocess.TimeoutExpired:code=-999
        row=dict(precision=precision,block=block,returncode=code,elapsed_seconds=time.monotonic()-begin);jobs.append(row)
        (folder/'processes.json').write_text(json.dumps(jobs,indent=2)+'\n');print(json.dumps(row),flush=True)
    if any(j['returncode'] for j in jobs):raise SystemExit(1)

if __name__=='__main__':main()
