"""Reconcile all sixteen completed models and preserve the launch-time state."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/dpa4-replicates-full-20261106-02';OLD=ROOT/'runs/dpa4-n200-pilot-20261106-02'

def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')

def main():
    result=read(RUN/'comparison-20261107/comparison.json');assert result['status']=='PASS'
    for p,name in [(ROOT/'state.json','state-before-completed-comparison.json'),(ROOT/'protocol/experiment_matrix.yaml','matrix-before-completed-comparison.json')]:
        if not (RUN/name).exists():(RUN/name).write_bytes(p.read_bytes())
    state=read(ROOT/'state.json');matrix=read(ROOT/'protocol/experiment_matrix.yaml')
    plan=read(RUN/'collected/plan.json');jobs=read(RUN/'submission-summary.json');manifest=read(RUN/'collected/COLLECTION_MANIFEST.json')
    acct={row.split('|')[0]:row.split('|') for row in manifest['accounting'].splitlines()[1:] if row}
    old=read(OLD/'comparison-v1/comparison.json');updated=[]
    for key,c in result['cases'].items():
        ds=[d for d in matrix['training_designs'] if d.get('architecture')=='DPA4-Neo' and d.get('functional')==c['XC'] and d.get('adaptation')==c['method'] and d.get('train_size')==c['N'] and d.get('training_seed')==c['seed']]
        assert len(ds)==1;d=ds[0];p=Path(c['case_folder']);n=c['N'];seed=c['seed']
        if n==200 and seed==10:
            allocated=old['cases'][f"{c['XC']}-{c['method']}"]['allocated_GPU_seconds'];task=old['cases'][f"{c['XC']}-{c['method']}"]['slurm_task']
        else:
            task=jobs[f'N{n}_training']+'_'+str(plan['groups'][f'N{n}'].index(key));row=acct[task]
            assert row[1]=='COMPLETED' and row[6]=='0:0';allocated=int(row[2])
        d.update(state='completed',blocked_by=[],launch_ready=True,numerical_QC='PASS',training_steps=50*n,
                 actual_cost=dict(GPU_hours=allocated/3600,CPU_hours=allocated*4/3600,scope='GPU allocation including evaluation; CPU-only QC is separate.',
                    training_wall_seconds=c['training']['training_seconds'],peak_cuda_allocated_bytes=c['training']['peak_cuda_allocated_bytes']),
                 result=str(RUN/'comparison-20261107/comparison.json'),model_sha256=c['model_sha256'],
                 executed_configuration=str(p/'input.json'),executed_normalized_configuration=str(p/'out.json'),
                 trainable_mask=str(p/'parameter-inventory.json'),slurm_task=task,
                 next_step='N200 three-seed comparison complete: retain the target/harmonic tradeoff. N694 has seed10 only; additional seeds are needed before claiming a robust full-train ranking.')
        updated.append(key)
    assert len(updated)==16
    state.update(stage='DPA4_N200_three_seed_and_N694_comparison_completed',updated_local_date='2026-11-07',updated_utc=result['utc'],
        completed_training_jobs=16,new_training_jobs=16,latest_expansion_snapshot=read(RUN/'latest-progress.json'),
        expansion_comparison=str(RUN/'comparison-20261107/comparison.json'),expansion_resource_accounting=result['costs'],
        expansion_status={c['name']:dict(state='completed',steps=c['expected_steps'],expected_steps=c['expected_steps']) for c in plan['cases']},
        expansion_validation=dict(N200_seeds=[10,20,30],N694_seeds=[10],numerical_QC='PASS',checkpoints_verified=260,case_count=16,test_access=False),
        next_actions=['If continuing full-train method selection, repeat N694 with seeds20/30; not submitted in the status/collection request.',
                      'Choose the required target/harmonic tradeoff explicitly: no single method dominates all validated quantities.',
                      'Keep test149 and archive49 outside method tuning; do not infer long MD stability from harmonic mesh checks.'])
    state['current_gates']['FT']='N200_three_seed_reproducibility_PASS; N694_single_seed; no_unique_method_choice'
    report='reports/14_multiseed_fulltrain_comparison.md'
    if report not in state['reports']:state['reports'].append(report)
    matrix['screening_summary'].update(completed_DPA4_N200_cases=12,completed_DPA4_N694_cases=4,
        measured_expansion_GPU_hours=result['costs']['new_GPU_hours'],latest_result=report,
        conclusion='N200: LoRA wins aligned validation E/F and Gamma singlet, full FT wins global phonon errors across all three seeds. N694: one seed only; size improves forces/stress but not every energy/phonon observable.')
    write(ROOT/'state.json',state);write(ROOT/'protocol/experiment_matrix.yaml',matrix)
    text='''# Ti–Al FM: текущее состояние

7 ноября 2026: все16 моделей завершены и собраны —12 N200 (seeds10/20/30) и4 N694 (seed10). Численные проверки PASS,260 checkpoint-проверок; новые12 validation-наборов воспроизведены локально. Новых DFT0, test149 не использован.

На N200 LoRA выигрывает aligned E/F и Γ c-singlet во всех трёх seeds, full FT — общую ошибку фононного спектра. На N694 full FT лучше по спектру и точечным E/stress, LoRA немного лучше по F; E/stress-различия неубедительны по group-bootstrap. Все16 моделей без мнимых мод ниже−0.001THz на mesh50³, включая raw FC.

Увеличение train улучшает F/stress, но PBE spectral RMSE ухудшается, а синглет r²SCAN может ухудшиться при улучшении спектра. Универсального победителя нет. N694 имеет один seed; дополнительные повторы в данном статусном запросе не запускались. Последнее продолжение потребовало17.708GPU·h.

[Полное сравнение с разбросом и графиками](14_multiseed_fulltrain_comparison.md) · [История запуска](13_multiseed_fulltrain_launch.md) · [Первоначальный seed10](12_n200_pilot_comparison.md) · [Разбиение](07_grouped_split.md).
'''
    (ROOT/'reports/summary.md').write_text(text,encoding='utf8')
    print('RECORDED',len(updated),'completed models')

if __name__=='__main__':main()
