"""Compare fixed final N200 checkpoints, with paired validation group bootstrap."""
from pathlib import Path
import datetime, hashlib, json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/dpa4-n200-pilot-20261106-02'
OUT=RUN/'comparison-v1'
CASES=['PBE-lora16','PBE-full','r2SCAN-lora16','r2SCAN-full']
KEYS=['E_raw_meV_atom','E_aligned_meV_atom','F_component_eV_A','stress_component_GPa']

def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def equal(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        assert np.isclose(a,b,rtol=1e-11,atol=1e-12),(a,b)
    else:assert a==b,(a,b)

def paired(a,b):
    aa={r['id']:r for r in a if r['split']=='validation'}
    bb={r['id']:r for r in b if r['split']=='validation'}
    assert aa.keys()==bb.keys() and len(aa)==149
    groups=sorted({r['group'] for r in aa.values()}); assert len(groups)==23
    sums=np.zeros((2,len(groups),6))
    for sid in sorted(aa):
        x,y=aa[sid],bb[sid]
        assert all(x[k]==y[k] for k in ['group','natoms','family','Al_fraction'])
        gi=groups.index(x['group'])
        for mi,r in enumerate([x,y]):
            sums[mi,gi]+=[(1000*r['energy_aligned_error_eV_atom'])**2,1,
                          r['force_sq_sum'],3*r['natoms'],r['stress_sq_sum'],6]
    rng=np.random.default_rng(61106); indices=rng.integers(len(groups),size=(2000,len(groups)))
    sampled=sums[:,indices,:].sum(axis=2)
    values=np.sqrt(sampled[:,:,[0,2,4]]/sampled[:,:,[1,3,5]])
    totals=sums.sum(axis=1); point=np.sqrt(totals[:,[0,2,4]]/totals[:,[1,3,5]])
    return dict(groups=23,replicates=2000,seed=61106,
        definition='Delta RMSE = LoRA minus full. Identical conservative composition groups resampled for both models. Energy offsets held at their N200 train estimates. Conditional on this dataset and training seed; not training-seed uncertainty.',
        delta={k:dict(point=float(point[0,i]-point[1,i]),CI95=np.quantile(values[0,:,i]-values[1,:,i],[.025,.975]).tolist())
               for i,k in enumerate(KEYS[1:])})

def main():
    OUT.mkdir(exist_ok=True)
    collected=RUN/'collected-pilots'; manifest=read(collected/'COLLECTION_MANIFEST.json')
    baseline=read(RUN/'baseline-analysis/metrics.json')
    physics=read(RUN/'pilot-physical-diagnostics-v1/summary.json'); assert physics['status']=='PASS'
    results={}; rows={}; exposures=[]; train_ids=[]; val_ids=[]; hashes={}
    for i,name in enumerate(CASES):
        p=collected/'training-plan01/cases'/name; t=read(p/'training-summary.json'); xc=t['case']['XC']
        ev=read(p/'evaluation/summary.json'); qc=read(p/'evaluation/numerical-QC/summary.json')
        update=read(p/'parameter-update-check.json'); inventory=read(p/'parameter-inventory.json')
        checkpoints=[read(f) for f in sorted((p/'checks').glob('*.json'))]
        for v in [t,ev,qc,update,read(p/'zero-step.json')]+checkpoints:assert v['status']=='PASS'
        assert len(checkpoints)==10 and checkpoints[-1]['step']==10000
        assert not update['frozen_changed']
        assert not t['test_access'] and not ev['test_access']
        assert t['steps']==10000 and t['case']['N']==200 and t['case']['validation']==149
        merged_hash=sha(p/'merged/10000.pt')
        assert merged_hash==ev['model_sha256']==qc['sha256']==checkpoints[-1]['merged_sha256']
        assert merged_hash==manifest['final_artifacts'][name]['merged/10000.pt']['sha256']
        remote_metrics=read(p/'evaluation/metrics/metrics.json')
        metrics=read(RUN/'local-metric-verification'/name/'metrics.json')
        equal(metrics,remote_metrics)
        rr=list(map(json.loads,(RUN/'local-metric-verification'/name/'per_structure.jsonl').read_text().splitlines()))
        assert {r['split'] for r in rr}=={'train','validation'} and len(rr)==349
        rows[name]=rr
        train_ids.append(sorted(r['id'] for r in rr if r['split']=='train'))
        val_ids.append(sorted(r['id'] for r in rr if r['split']=='validation'))
        assert len(train_ids[-1])==len(set(train_ids[-1]))==200
        assert len(val_ids[-1])==len(set(val_ids[-1]))==149
        assert set(train_ids[-1]).isdisjoint(val_ids[-1])
        assert {r['group'] for r in rr if r['split']=='train'}.isdisjoint({r['group'] for r in rr if r['split']=='validation'})
        exposures.append(sha(p/'training-exposures.jsonl'))
        assert len((p/'training-exposures.jsonl').read_text().splitlines())==10000
        ac=next(line.split('|') for line in manifest['accounting'].splitlines() if line.startswith('20988454_'+str(i)+'|'))
        h=physics['results'][xc][t['case']['method']]
        equal(h['sorted_RMSE_THz'],read(p/'evaluation/harmonics/summary.json')['sorted_band_RMSE_THz'])
        results[name]=dict(XC=xc,method=t['case']['method'],status='COMPLETED',steps=10000,
            metrics=metrics[xc],harmonics=h,physical_job_id=physics['job_id'],
            training=t,parameter_counts={k:inventory[k] for k in ['total','trainable','adapter_trainable','frozen']},
            model_sha256=merged_hash,export_sha256=ev['export_sha256'],
            slurm_task=ac[0],allocated_GPU_seconds=int(ac[2]),allocated_CPU_hours=int(ac[2])*int(ac[3])/3600,
            numerical_QC='PASS',local_metric_reproduction='PASS',checkpoint_checks=10,
            merged_weights=str(p/'merged/10000.pt'))
        for f in [p/'training-summary.json',p/'evaluation/summary.json',p/'evaluation/harmonics/summary.json',
                  p/'evaluation/metrics/metrics.json',p/'evaluation/numerical-QC/summary.json']:
            hashes[str(f)]=sha(f)
    assert len(set(exposures))==1, 'Optimizer exposure order differs.'
    assert all(x==train_ids[0] for x in train_ids) and all(x==val_ids[0] for x in val_ids)
    boot={xc:paired(rows[xc+'-lora16'],rows[xc+'-full']) for xc in ['PBE','r2SCAN']}
    output=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        cases=results,paired_validation=boot,baseline={xc:baseline[xc] for xc in ['PBE','r2SCAN']},
        baseline_harmonics=physics['results'],checks=dict(identical_train_validation_ids=True,
        disjoint_train_validation_groups=True,identical_batch_exposure_order=True,
        exposure_log_sha256=exposures[0],local_remote_metrics_match=True,all_final_models_hash_verified=True,
        all_ten_checkpoints_per_case_verified=True,test_labels_used_for_comparison=False),
        input_sha256=hashes,collection=str(RUN/'completed-pilot-collection.json'))
    (OUT/'comparison.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',paired_validation=boot),indent=2))

if __name__=='__main__':main()
