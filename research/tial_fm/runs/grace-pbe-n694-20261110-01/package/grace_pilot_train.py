"""Native GRACE Adam with source-only restore, fixed-final selection and mask evidence."""
from pathlib import Path
import argparse,hashlib,json,os,sys,time
import numpy as np
from grace_pilot_common import ROOT,dump,sha,tensor_sha,calculator,predict,atoms
class SmokeDone(BaseException):pass
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    storage_policy=None
    if (ROOT/'STORAGE_POLICY.json').exists():
        from grace_storage_guard import quota_snapshot
        storage_policy=json.loads((ROOT/'STORAGE_POLICY.json').read_text())
        quota_snapshot(storage_policy['minimum_start_bytes'])
    import tensorflow as tf
    from tensorpotential import TensorPotential
    from tensorpotential.cli import gracemaker as gm
    from tensorpotential.cli.train import get_dataset_num_batches
    assert len(tf.config.list_physical_devices('GPU'))==1
    for d in tf.config.list_physical_devices('GPU'):tf.config.experimental.set_memory_growth(d,True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    plan=json.loads((ROOT/'plan.json').read_text());case=next(c for c in plan['cases'] if c['name']==a.case);xc=case['XC']
    for manifest in ['FILES_SHA256.json','TRAINING_FILES_SHA256.json']:
        for name,digest in json.loads((ROOT/manifest).read_text()).items():assert sha(ROOT/name)==digest,name
    if (ROOT/'CHECKPOINT_FILES_SHA256.json').exists():
        for name,digest in json.loads((ROOT/'CHECKPOINT_FILES_SHA256.json').read_text()).items():assert sha(ROOT/name)==digest,name
    backend=json.loads((ROOT/'backend-manifest.json').read_text())
    for name,record in backend['files'].items():assert sha(Path(backend['root'])/name)==record['sha256'],name
    assert json.loads((ROOT/'baseline/summary.json').read_text())['status']=='PASS'
    assert json.loads((ROOT/'baseline/harmonics/summary.json').read_text())['status']=='PASS'
    work=ROOT/('smoke' if a.smoke else 'cases')/a.case;work.mkdir(parents=True,exist_ok=True);os.chdir(work)
    if a.smoke:
        assert not (work/'input.yaml').exists();(work/'input.yaml').write_bytes((ROOT/'cases'/a.case/'input.yaml').read_bytes())
    else:assert json.loads((ROOT/'smoke'/a.case/'optimizer-check.json').read_text())['status']=='PASS'
    assert not (work/'attempt.json').exists();dump(work/'attempt.json',dict(job=os.environ['SLURM_JOB_ID'],smoke=a.smoke))
    cfg=json.loads((work/'input.yaml').read_text());assert cfg['potential']['param_dtype']=='float64' and cfg['fit']['maxiter']==50 and cfg['fit']['batch_size']==1
    prefix=str(ROOT/'aligned'/xc/'checkpoint');alignment=json.loads((ROOT/'aligned'/xc/'alignment.json').read_text())
    for name,h in alignment['checkpoint_hashes'].items():assert sha(ROOT/'aligned'/xc/name)==h
    original_load=TensorPotential.load_checkpoint
    def restore(self,checkpoint_name=None,**kw):
        if checkpoint_name is None:
            # Native CLI attempts a best-validation reload after the last epoch.
            # Preserve and export the declared final epoch instead.
            assert (work/'training-summary.json').exists();dump(work/'validation-selection.json',dict(best_checkpoint_reload_skipped=True,selection='fixed final epoch50'));return
        # /home6 is a symlink to /misc/home6 on UMT. Compare canonical
        # prefixes while retaining the verified checkpoint identity.
        actual_prefix=Path(checkpoint_name).resolve()
        expected_prefix=Path(prefix).resolve()
        assert actual_prefix==expected_prefix, (str(actual_prefix),str(expected_prefix))
        assert Path(str(actual_prefix)+'.index').is_file()
        dump(work/'checkpoint-path-check.json',dict(status='PASS',requested=str(checkpoint_name),expected=prefix,canonical=str(actual_prefix)))
        original_load(self,checkpoint_name=prefix,model_only=True,expect_partial=True,assert_existing_objects_matched=True)
        assert int(self.optimizer.iterations.numpy())==0 and self.step==0 and self.epoch==0
    TensorPotential.load_checkpoint=restore
    if storage_policy:
        original_save=TensorPotential.save_checkpoint
        def save_with_quota(self,suffix='',**kw):
            if suffix=='.best_test_loss' and storage_policy['skip_unused_best_checkpoint']:
                dump(work/'checkpoint-storage-policy.json',dict(skip_unused_best_checkpoint=True,selection='fixed final epoch50',last_skipped_epoch=int(self.epoch)))
                return
            dump(work/'last-checkpoint-quota.json',quota_snapshot(storage_policy['minimum_checkpoint_bytes']))
            return original_save(self,suffix=suffix,**kw)
        TensorPotential.save_checkpoint=save_with_quota
    native_train=gm.train_adam
    def observe(tp,**kw):
        assert get_dataset_num_batches(kw['train_ds'])==case['N'] and get_dataset_num_batches(kw['test_ds'])==case['validation']
        assert {v.dtype for v in tp.model.variables_to_train}=={tf.float64}
        selected={id(v) for v in tp.model.variables_to_train};variables=list(tp.model.variables)
        # TensorFlow permits equal names for distinct variables. Keep each
        # object separate, including non-trainable constants introduced by alignment.
        assert len({id(v) for v in variables})==len(variables)
        keys={id(v):f'{i}:{v.name}' for i,v in enumerate(variables)}
        initial={keys[id(v)]:tensor_sha(v) for v in variables}
        trainable=sum(int(np.prod(v.shape)) for v in variables if id(v) in selected)
        assert trainable==case.get('expected_trainable',21111690 if case['method']=='restricted' else 21764956)
        patterns=case.get('trainable_patterns',plan['restricted_patterns'] if case['method']=='restricted' else None)
        if patterns:
            assert cfg['fit']['trainable_variable_names']==patterns
            assert all((id(v) in selected)==any(s in v.name for s in patterns) for v in variables)
            assert all(any(s in v.name for v in variables if id(v) in selected) for s in patterns)
        if 'initial_parameter_signatures' in plan:
            signatures=sorted([[v.name,v.shape.as_list(),tensor_sha(v)] for v in variables])
            assert signatures==plan['initial_parameter_signatures'][xc]
        dump(work/'parameter-inventory.json',dict(trainable=trainable,parameters=[dict(key=keys[id(v)],name=v.name,shape=v.shape.as_list(),dtype=v.dtype.name,trainable=id(v) in selected,initial_sha256=initial[keys[id(v)]]) for v in variables]))
        calc=calculator(tp);zero=[]
        baseline=json.loads((ROOT/'baseline/numerical-QC.json').read_text())['records']
        for p,b in zip(json.loads((ROOT/'probes.json').read_text()),baseline):
            at=atoms(p);e,f,s=predict(calc,at);expected=b['base']['energy']+sum(at.get_chemical_symbols().count(k)*v for k,v in alignment['offset_eV_atom'].items())
            assert abs(e-expected)<1e-7 and np.max(abs(f-b['base']['forces']))<1e-8 and np.max(abs(s-b['base']['stress']))<1e-9
            zero.append(dict(id=p['id'],energy_shift_check=abs(e-expected)))
        dump(work/'zero-step.json',dict(status='PASS',checks=zero,train_only_alignment=True,optimizer_iterations=int(tp.optimizer.iterations.numpy())))
        step_fn=tp.distributed_train_step;state=dict(steps=0,seconds=[],batch_hashes=[]);started=time.perf_counter()
        def step(batch):
            if state['steps']==0:
                assert int(tp.optimizer.iterations.numpy())==0 and tp.step==0
                optimizer_variables=tp.optimizer.variables
                for v in (optimizer_variables() if callable(optimizer_variables) else optimizer_variables):
                    if 'learning_rate' not in v.name:assert np.max(abs(v.numpy()))==0
            if state['steps']<5:
                h=hashlib.sha256()
                for v in tf.nest.flatten(batch):
                    if hasattr(v,'numpy'):h.update(tensor_sha(v).encode())
                state['batch_hashes'].append(h.hexdigest())
            begin=time.perf_counter();result=step_fn(batch)
            for v in tf.nest.flatten(result):
                if hasattr(v,'numpy'):assert np.isfinite(v.numpy()).all()
            state['seconds'].append(time.perf_counter()-begin);state['steps']+=1
            assert int(tp.optimizer.iterations.numpy())==state['steps']
            if state['steps']==5 and 'reference_first5_batch_hashes' in plan:
                assert state['batch_hashes']==plan['reference_first5_batch_hashes'][xc]
            if state['steps']==5 and not a.smoke:
                assert state['batch_hashes']==json.loads((ROOT/'smoke'/a.case/'optimizer-check.json').read_text())['first5_batch_hashes']
            if state['steps']==1 or state['steps']%1000==0 or state['steps']==case['steps'] or (a.smoke and state['steps']==5):
                changes=[keys[id(v)] for v in variables if tensor_sha(v)!=initial[keys[id(v)]]];frozen=[keys[id(v)] for v in variables if id(v) not in selected and keys[id(v)] in changes]
                assert changes and not frozen
                check=dict(status='PASS',steps=state['steps'],trainable=trainable,changed_tensors=changes,frozen_changed=frozen,optimizer_iterations=int(tp.optimizer.iterations.numpy()),fresh_optimizer=True,
                    recent_step_seconds=state['seconds'][-4:],first5_batch_hashes=state['batch_hashes'],GPU_memory=tf.config.experimental.get_memory_info('GPU:0'))
                dump(work/'checks'/f'{state["steps"]:05d}.json',check)
                if state['steps']==1:dump(work/'first-optimizer-step.json',check)
                if a.smoke and state['steps']==5:dump(work/'optimizer-check.json',check);raise SmokeDone()
            if state['steps']%100==0:dump(work/'progress.json',dict(status='RUNNING',steps=state['steps'],elapsed_seconds=time.perf_counter()-started,GPU_memory=tf.config.experimental.get_memory_info('GPU:0')))
            return result
        tp.distributed_train_step=step
        native_train(tp,**kw)
        assert state['steps']==case['steps'] and tp.epoch==case['epochs'] and tp.step==case['steps']
        if storage_policy:dump(work/'final-export-quota.json',quota_snapshot(storage_policy['minimum_final_export_bytes']))
        final=work/'final-checkpoint';final.mkdir(exist_ok=False);tf.train.Checkpoint(model=tp.model).write(str(final/'checkpoint'))
        dump(work/'training-summary.json',dict(status='PASS',case=case,steps=case['steps'],epochs=case['epochs'],training_seconds=time.perf_counter()-started,final_checkpoint=str(final/'checkpoint'),hashes={p.name:sha(p) for p in final.glob('*')},test_access=False,selection='fixed final epoch50',GPU_memory=tf.config.experimental.get_memory_info('GPU:0')))
    gm.train_adam=observe
    try:gm.main(['input.yaml'])
    except SmokeDone:print('OPTIMIZER_PREFLIGHT_PASS',a.case,flush=True);return
    assert not a.smoke and (work/'training-summary.json').exists()
    print('TRAINING_PASS',a.case,flush=True)
if __name__=='__main__':main()
