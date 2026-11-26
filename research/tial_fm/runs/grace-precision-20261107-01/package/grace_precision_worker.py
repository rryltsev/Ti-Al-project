"""Fresh-process GRACE inference: cache-bypassed E/F/stress, latency and diagnostics."""
from pathlib import Path
import argparse,hashlib,json,os,resource,time,traceback
import numpy as np

ROOT=Path(__file__).resolve().parent
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',choices=['cpu','gpu'],required=True);ap.add_argument('--precision',choices=['fp32','fp64'],required=True);ap.add_argument('--block',type=int,required=True);a=ap.parse_args()
    out=ROOT/'results'/a.device/f'{a.precision}-block{a.block}';out.mkdir(parents=True,exist_ok=False)
    try:
        import tensorflow as tf
        from ase import Atoms
        from ase.io import read
        from ase.calculators.calculator import all_changes
        from tensorpotential import TensorPotential
        from tensorpotential.instructions.base import load_instructions
        from tensorpotential.calculator import TPCalculator
        import importlib.metadata as md
        plan=json.loads((ROOT/'plan.json').read_text());model_path=Path(plan['checkpoints'][a.precision]['path'])
        visible=tf.config.list_physical_devices('GPU');assert (len(visible)==1) if a.device=='gpu' else not visible
        for d in visible:tf.config.experimental.set_memory_growth(d,True)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        dtype=tf.float32 if a.precision=='fp32' else tf.float64
        started=time.perf_counter();model=TensorPotential(potential=load_instructions(str(model_path/'model.yaml')),param_dtype=dtype,jit_compile=False)
        model.load_checkpoint(checkpoint_name=str(model_path/'checkpoint'),expect_partial=True,assert_existing_objects_matched=True,model_only=True)
        model.model.decorate_compute_function(jit_compile=(a.device=='gpu'))
        load_seconds=time.perf_counter()-started
        parameter_bytes=sum(np.asarray(v.numpy()).nbytes for v in model.model.variables if v.dtype!=tf.string)
        parameter_dtypes={v.dtype.name for v in model.model.variables_to_train};assert parameter_dtypes=={dtype.name}
        calc=TPCalculator(model.model,enable_uq_if_available=False,mode='diverse',adaptive_padding=False,max_number_reduction_recompilation=0)
        def evaluate(at):
            start=time.perf_counter();calc.calculate(at,properties=['energy','forces','stress'],system_changes=all_changes)
            r=calc.results;val=dict(energy=float(r['energy']),forces=np.asarray(r['forces']).copy(),stress=np.asarray(r['stress']).copy())
            elapsed=time.perf_counter()-start
            assert np.isfinite(val['energy']) and all(np.isfinite(val[k]).all() for k in ['forces','stress'])
            return val,elapsed,float(calc.eval_time)
        timing=[];rng=np.random.default_rng(715)
        sizes=plan['timing_cases'][a.device]
        for name in sizes:
            # Independent calculator padding state for every size, same policy for both precisions.
            calc=TPCalculator(model.model,enable_uq_if_available=False,mode='diverse',adaptive_padding=False,max_number_reduction_recompilation=0)
            at=read(ROOT/'inputs/timing'/f'{name}.extxyz');frames=[]
            reps=6 if a.device=='cpu' else 30;warm=2 if a.device=='cpu' else 5
            for i in range(reps+warm+1):
                frame=at.copy();frame.positions+=rng.normal(0,.002,frame.positions.shape);frames.append(frame)
            first,cold,kernel=evaluate(frames[0])
            for frame in frames[1:warm+1]:evaluate(frame)
            if visible:tf.config.experimental.reset_memory_stats('GPU:0')
            samples=[];kernel_samples=[]
            for frame in frames[warm+1:]:
                _,dt,kt=evaluate(frame);samples.append(dt);kernel_samples.append(kt)
            memory=tf.config.experimental.get_memory_info('GPU:0') if visible else None
            timing.append(dict(name=name,natoms=len(at),cold_seconds=cold,warmup_calls=warm,measured_calls=reps,wall_seconds=samples,kernel_sync_seconds=kernel_samples,
                median_seconds=float(np.median(samples)),p10_seconds=float(np.quantile(samples,.1)),p90_seconds=float(np.quantile(samples,.9)),GPU_memory=memory))
            dump(out/'progress.json',dict(phase='timing',completed=name,timings=timing))
        diagnostics={};pred={}
        if a.block==0:
            panel=read(ROOT/'inputs/panel.extxyz',index=':')
            for at in panel:
                value,dt,kt=evaluate(at);sid=at.info['geometry_id'];pred[sid]=dict(energy=value['energy'],forces=value['forces'].tolist(),stress=value['stress'].tolist(),natoms=len(at))
            dump(out/'panel-predictions.json',pred)
            # Fixed force component chosen from DFT labels, independent of precision.
            fdplan=plan['finite_difference'];at=next(at for at in panel if at.info['geometry_id']==fdplan['geometry_id'])
            ref,_,_=evaluate(at);atom,axis=fdplan['component'];force=[];stress=[]
            for h in [1e-2,1e-3,1e-4,1e-5]:
                plus=at.copy();minus=at.copy();plus.positions[atom,axis]+=h;minus.positions[atom,axis]-=h
                ep=evaluate(plus)[0]['energy'];em=evaluate(minus)[0]['energy'];numeric=-(ep-em)/(2*h)
                force.append(dict(h_A=h,analytic=float(ref['forces'][atom,axis]),finite_difference=numeric,absolute_error_eV_A=abs(numeric-ref['forces'][atom,axis])))
            for h in [1e-3,1e-4,1e-5]:
                plus=at.copy();minus=at.copy();plus.set_cell(at.cell.array*(1+h),scale_atoms=True);minus.set_cell(at.cell.array*(1-h),scale_atoms=True)
                numeric=(evaluate(plus)[0]['energy']-evaluate(minus)[0]['energy'])/(2*h*at.get_volume())
                analytic=float(np.sum(ref['stress'][:3]));stress.append(dict(h=h,analytic_trace_eV_A3=analytic,finite_difference_trace_eV_A3=numeric,absolute_error_GPa=abs(numeric-analytic)*160.21766208))
            diagnostics=dict(force_finite_difference=force,hydrostatic_stress_finite_difference=stress,component=fdplan)
            dump(out/'finite-differences.json',diagnostics)
            for xc in ['PBE','r2SCAN']:
                dest=out/'harmonic_forces'/xc;dest.mkdir(parents=True)
                for key in ['force000','force001','force002']:
                    at=read(ROOT/'harmonic_inputs'/xc/key/'POSCAR',format='vasp');v,dt,kt=evaluate(at)
                    np.save(dest/(key+'.npy'),v['forces'])
        summary=dict(status='PASS',device=a.device,precision=a.precision,block=a.block,load_seconds=load_seconds,parameter_bytes=parameter_bytes,
            parameter_dtypes=sorted(parameter_dtypes),main_compute_jit_compile=(a.device=='gpu'),tf32=False,UQ=False,engine='TensorPotential TPModel / TPCalculator diverse(segment); no ASE result cache',
            timings=timing,peak_process_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,versions={x:md.version(x) for x in ['tensorpotential','tensorflow','numpy','ase']},
            job_id=os.environ['SLURM_JOB_ID'],host=os.uname().nodename,checkpoint=plan['checkpoints'][a.precision],test_access=False)
        dump(out/'summary.json',summary);print(json.dumps(dict(status='PASS',device=a.device,precision=a.precision,block=a.block,timing_medians={x['name']:x['median_seconds'] for x in timing})),flush=True)
    except Exception as e:
        dump(out/'failure.json',dict(status='FAILED',error=repr(e),traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
