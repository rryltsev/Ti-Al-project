from pathlib import Path
import json,os,time
import numpy as np
from grace_pilot_common import ROOT,load_model,calculator,predict,numerical_qc,evaluate_dataset,metrics,dump,sha,atoms
def main():
    import tensorflow as tf
    from ase.io import read
    from tensorpotential.instructions.base import load_instructions,save_instructions_dict
    from tensorpotential.cli.data import inject_or_update_atomic_shift_in_model
    from tensorpotential.tpmodel import extract_cutoff_and_elements
    assert len(tf.config.list_physical_devices('GPU'))==1;tf.config.experimental.enable_tensor_float_32_execution(False)
    for d in tf.config.list_physical_devices('GPU'):tf.config.experimental.set_memory_growth(d,True)
    out=ROOT/'baseline';out.mkdir(exist_ok=False);plan=json.loads((ROOT/'plan.json').read_text());source=Path(plan['source_checkpoint'])
    for name,h in plan['source_hashes'].items():assert sha(source/name)==h
    for name,h in json.loads((ROOT/'FILES_SHA256.json').read_text()).items():assert sha(ROOT/name)==h
    model=load_model(source/'model.yaml',source/'checkpoint');calc=calculator(model)
    model.save_model(exact_path=str(out/'original-export'),jit_compile=True)
    qc=numerical_qc(calc,calculator(out/'original-export',export=True));dump(out/'numerical-QC.json',qc);assert qc['status']=='PASS'
    for xc in plan.get('active_XC',['PBE','r2SCAN']):
        rows=evaluate_dataset(calc,xc);m=metrics(rows);dump(out/(xc+'-original-metrics.json'),m);dump(out/(xc+'-original-rows.json'),rows)
        aligned=ROOT/'aligned'/xc;aligned.mkdir(parents=True,exist_ok=False)
        pot=load_instructions(str(source/'model.yaml'));_,symbols,indices=extract_cutoff_and_elements(model.model.instructions)
        emap=dict(zip(symbols.tolist(),[int(i) for i in indices]));assert 'Al' in emap and 'Ti' in emap
        assert sorted(emap.values())==list(range(len(emap)))
        # ConstantScaleShiftTarget packs sorted dict VALUES into a dense tensor;
        # sparse keys {Al_index, Ti_index} lose their positions. Keep all89 slots.
        dense_shifts={symbol:m['offset_eV_atom'].get(symbol,0.0) for symbol in emap}
        inject_or_update_atomic_shift_in_model(pot,dense_shifts,emap);save_instructions_dict(str(aligned/'model.yaml'),pot,param_dtype=tf.float64)
        shifted=load_model(aligned/'model.yaml',source/'checkpoint');cc=calculator(shifted);checks=[]
        for probe in json.loads((ROOT/'probes.json').read_text()):
            at=atoms(probe);e,f,s=predict(calc,at);ee,ff,ss=predict(cc,at);delta=sum(at.get_chemical_symbols().count(k)*v for k,v in m['offset_eV_atom'].items())
            z=dict(id=probe['id'],E_shift_error=abs(ee-e-delta),F_max_difference=float(np.max(abs(f-ff))),stress_max_difference=float(np.max(abs(s-ss))))
            checks.append(z);dump(aligned/'alignment-checks.json',checks)
            assert z['E_shift_error']<1e-7 and z['F_max_difference']<1e-8 and z['stress_max_difference']<1e-9,z
        tf.train.Checkpoint(model=shifted.model).write(str(aligned/'checkpoint'))
        dump(aligned/'alignment.json',dict(status='PASS',**m,checks=checks,source=f"train{m['metrics']['train']['structures']} only",checkpoint_hashes={p.name:sha(p) for p in aligned.glob('checkpoint.*')}))
        for key in ['force000','force001','force002']:
            at=read(ROOT/'harmonic_inputs'/xc/key/'POSCAR',format='vasp');_,forces,_=predict(calc,at);p=out/'harmonic_forces'/xc;p.mkdir(parents=True,exist_ok=True);np.save(p/(key+'.npy'),forces)
    dump(out/'summary.json',dict(status='PASS',B0='PASS',alignment='PASS',source_hashes=plan['source_hashes'],job_id=os.environ['SLURM_JOB_ID'],test_access=False))
if __name__=='__main__':main()
