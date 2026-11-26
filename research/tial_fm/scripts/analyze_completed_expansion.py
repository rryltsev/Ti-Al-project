"""Verify all final artifacts and compare methods, seeds and training-set size."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import copy,datetime,hashlib,json,os,subprocess,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/dpa4-replicates-full-20261106-02'
OLD=ROOT/'runs/dpa4-n200-pilot-20261106-02';OUT=RUN/'comparison-20261107'
METRICS=['E_raw_meV_atom','E_aligned_meV_atom','F_component_eV_A','stress_component_GPa']

def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def equal(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    elif isinstance(a,(int,float)) and not isinstance(a,bool):assert np.isclose(a,b,rtol=1e-10,atol=1e-11),(a,b)
    else:assert a==b,(a,b)
def stats(values):
    a=np.asarray(values,dtype=float)
    return dict(n=len(a),mean=float(a.mean()),sample_SD=float(a.std(ddof=1)) if len(a)>1 else None,
                min=float(a.min()),max=float(a.max()),values=a.tolist())

def verify_metrics(case):
    name=case['name'];p=RUN/'collected/training-plan01/cases'/name;out=RUN/'local-metric-verification'/name
    if not out.exists():
        args=[sys.executable,str(RUN/'collected/analyze_predictions.py'),'--package',str(RUN/'package'),
              '--data-root',str(RUN/'package/data'/f"N{case['N']}"),'--train-size',str(case['N']),
              '--predictions',str(p/'evaluation/predictions'),'--output',str(out)]
        env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        result=subprocess.run(args,capture_output=True,text=True,env=env)
        assert result.returncode==0,(name,result.stderr)
    equal(load(out/'metrics.json'),load(p/'evaluation/metrics/metrics.json'))
    return name

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'comparison.json').exists()
    plan=load(RUN/'collected/plan.json');manifest=load(RUN/'collected/COLLECTION_MANIFEST.json')
    with ThreadPoolExecutor(max_workers=4) as pool:
        verified=list(pool.map(verify_metrics,plan['cases']))
    print('Verified metrics',len(verified),flush=True)
    prior_physics=load(OLD/'pilot-physical-diagnostics-v1/summary.json')['results']
    previous=load(OLD/'comparison-v1/comparison.json');entries=[]
    for name,expected in plan['existing_seed10'].items():
        entries.append((OLD/'collected-pilots/training-plan01/cases'/name,200,10,expected['model_sha256']))
    for c in plan['cases']:
        entries.append((RUN/'collected/training-plan01/cases'/c['name'],c['N'],c['seed'],
                        manifest['final_artifacts'][c['name']][f"merged/{50*c['N']:05d}.pt"]['sha256']))
    cases={};rows={};evidence={};checkpoints=0
    for p,n,seed,expected_hash in entries:
        t=load(p/'training-summary.json');c=t['case'];xc=c['XC'];method=c['method'];key=f'{xc}-{method}-N{n}-seed{seed}'
        ev=load(p/'evaluation/summary.json');qc=load(p/'evaluation/numerical-QC/summary.json');update=load(p/'parameter-update-check.json')
        inventory=load(p/'parameter-inventory.json');h=load(p/'evaluation/harmonics/summary.json')
        for v in [t,ev,qc,update,h,load(p/'zero-step.json')]:assert v['status']=='PASS'
        assert t['steps']==n*50 and t['exposures']['steps']==n*50
        assert not t['test_access'] and not ev['test_access'] and not update['frozen_changed']
        mh=sha(p/'merged'/f'{n*50:05d}.pt');assert mh==expected_hash==ev['model_sha256']==qc['sha256']
        saved=[load(f) for f in sorted((p/'checks').glob('*.json'))]
        expected_steps=list(range(1000,n*50+1,1000))
        if expected_steps[-1]!=n*50:expected_steps.append(n*50)
        assert [s['step'] for s in saved]==expected_steps
        assert all(s['status']=='PASS' for s in saved) and saved[-1]['merged_sha256']==mh
        checkpoints+=len(saved)
        m=load(p/'evaluation/metrics/metrics.json')[xc]
        rr=list(map(json.loads,(p/'evaluation/metrics/per_structure.jsonl').read_text().splitlines()))
        assert len(rr)==n+149 and {r['split'] for r in rr}=={'train','validation'}
        assert {r['group'] for r in rr if r['split']=='train'}.isdisjoint({r['group'] for r in rr if r['split']=='validation'})
        assert m['metrics']['train']['structures']==n and m['metrics']['validation']['structures']==149
        if n==200 and seed==10:
            phy=copy.deepcopy(prior_physics[xc][method]);df=prior_physics[xc]['DFT']
        else:
            extra=load(p/'evaluation/physical-diagnostics/summary.json');assert extra['status']=='PASS'
            phy=copy.deepcopy(extra['model']);phy.update(extra['comparison']);df=extra['reference']
        gamma=phy['Gamma'];dg=df['Gamma'];f=m['metrics']['validation']
        values={k:f[k]['RMSE'] for k in METRICS}
        values.update(phonon_sorted_RMSE_THz=h['sorted_band_RMSE_THz'],phonon_overlap_RMSE_THz=phy['overlap_matched_RMSE_THz'],
            Gamma_singlet_THz=gamma['singlet_c_THz'],Gamma_doublet_THz=float(np.mean(gamma['doublet_basal_THz'])),
            Gamma_singlet_abs_error_THz=abs(gamma['singlet_c_THz']-dg['singlet_c_THz']),
            Gamma_doublet_abs_error_THz=abs(np.mean(gamma['doublet_basal_THz'])-np.mean(dg['doublet_basal_THz'])),
            relative_Dmat_error=phy['relative_dynamical_matrix_Frobenius_error'],relative_FC_error=phy['relative_FC_Frobenius_error'],
            imaginary_modes=float(h['model']['imaginary_modes_weighted']),mesh_min_THz=h['model']['mesh_min_THz'],
            training_seconds=t['training_seconds'],peak_GPU_GiB=t['peak_cuda_allocated_bytes']/2**30)
        for direction in ['Gamma-M','Gamma-K','Gamma-A']:
            v=np.array(phy['acoustics'][direction]['phase_velocity_m_s'][0]);dv=np.array(df['acoustics'][direction]['phase_velocity_m_s'][0])
            for i in range(3):
                values[f'{direction}_v{i+1}_m_s']=float(v[i]);values[f'{direction}_v{i+1}_error_pct']=float(100*(v[i]/dv[i]-1))
        cases[key]=dict(N=n,seed=seed,XC=xc,method=method,values=values,metrics=m,harmonics=h,physics=phy,
            reference_Gamma=dg,model_sha256=mh,export_sha256=ev['export_sha256'],training=t,
            parameter_counts={k:inventory[k] for k in ['total','trainable','adapter_trainable','frozen']},
            exposure_log_sha256=sha(p/'training-exposures.jsonl'),case_folder=str(p),checkpoint_checks=len(saved))
        rows[key]=rr
        evidence[str(p/'evaluation/metrics/metrics.json')]=sha(p/'evaluation/metrics/metrics.json')
    assert len(cases)==16 and checkpoints==260
    all_val=[sorted(r['id'] for r in rr if r['split']=='validation') for rr in rows.values()]
    assert all(x==all_val[0] for x in all_val)
    grouped={};paired={};size={}
    for n,seeds in [(200,[10,20,30]),(694,[10])]:
        for seed in seeds:
            same=[c for c in cases.values() if c['N']==n and c['seed']==seed]
            assert len(same)==4 and len({c['exposure_log_sha256'] for c in same})==1
        for xc in ['PBE','r2SCAN']:
            for method in ['lora16','full']:
                keys=[f'{xc}-{method}-N{n}-seed{s}' for s in seeds];cc=[cases[k] for k in keys]
                assert len({c['exposure_log_sha256'] for c in cc})==len(seeds)
                ids=[sorted(r['id'] for r in rows[k] if r['split']=='train') for k in keys]
                assert all(x==ids[0] for x in ids)
                grouped[f'{xc}-{method}-N{n}']={k:stats([c['values'][k] for c in cc]) for k in cc[0]['values']}
            pair={}
            for metric in cc[0]['values']:
                delta=[cases[f'{xc}-lora16-N{n}-seed{s}']['values'][metric]-cases[f'{xc}-full-N{n}-seed{s}']['values'][metric] for s in seeds]
                pair[metric]=stats(delta)|dict(LoRA_lower_count=int(sum(v<0 for v in delta)),full_lower_count=int(sum(v>0 for v in delta)))
            paired[f'{xc}-N{n}']=pair
    for xc in ['PBE','r2SCAN']:
        for method in ['lora16','full']:
            a=cases[f'{xc}-{method}-N200-seed10'];b=cases[f'{xc}-{method}-N694-seed10']
            assert set(r['id'] for r in rows[f'{xc}-{method}-N200-seed10'] if r['split']=='train')<set(r['id'] for r in rows[f'{xc}-{method}-N694-seed10'] if r['split']=='train')
            size[f'{xc}-{method}']={k:dict(N200=a['values'][k],N694=b['values'][k],delta=b['values'][k]-a['values'][k],
                    percent_change=100*(b['values'][k]/a['values'][k]-1) if abs(a['values'][k])>1e-12 else None) for k in a['values']}
    for group in ['N200','N694']:
        remote=load(RUN/'collected/aggregates'/group/'summary.json');assert remote['status']=='PASS' and not remote['failures']
        for name,metrics in remote['seed_summary'].items():
            if group=='N694' and name.endswith('N200'):continue
            for metric,value in metrics.items():equal(grouped[name][metric],value)
    acct=[line.split('|') for line in manifest['accounting'].splitlines()[1:] if line and '.' not in line.split('|')[0]]
    gpu=[a for a in acct if a[0].startswith(('20988583_','20988587_'))]
    result=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=cases,seed_summary=grouped,
        paired_LoRA_minus_full=paired,paired_size_seed10=size,
        seed_policy='N200: mean and sample SD across seeds10,20,30; never pooled as independent validation observations. N694: seed10 only, SD unavailable. Paired differences use same seed; no best-seed selection.',
        checks=dict(local_metric_reproduction_12_cases=True,model_export_QC_hashes=True,checkpoint_checks=checkpoints,
                    identical_validation_ids=True,matched_exposure_order_within_seed=True,independent_exposure_order_between_seeds=True,
                    train200_subset_train694=True,no_test_predictions=True,all_numerical_QC='PASS'),
        costs=dict(new_GPU_hours=sum(int(a[2]) for a in gpu)/3600,
                   N200_new_GPU_hours=sum(int(a[2]) for a in gpu if a[0].startswith('20988583_'))/3600,
                   N694_GPU_hours=sum(int(a[2]) for a in gpu if a[0].startswith('20988587_'))/3600,
                   prior_N200_seed10_GPU_hours=sum(c['allocated_GPU_seconds'] for c in previous['cases'].values())/3600,
                   all_new_allocated_CPU_hours=sum(int(a[2])*int(a[3]) for a in acct)/3600),
        input_sha256=evidence,collection=load(RUN/'completed-collection.json'))
    (OUT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',cases=len(cases),checkpoints=checkpoints,costs=result['costs']),indent=2))

if __name__=='__main__':main()
