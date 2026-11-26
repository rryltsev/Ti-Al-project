"""Reconcile finished pilot evidence into the project state and experiment registry."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/dpa4-n200-pilot-20261106-02'

def main():
    data=json.loads((RUN/'comparison-v1/comparison.json').read_text());assert data['status']=='PASS'
    manifest=json.loads((RUN/'collected-pilots/COLLECTION_MANIFEST.json').read_text())
    statepath=ROOT/'state.json';matrixpath=ROOT/'protocol/experiment_matrix.yaml'
    for path,name in [(statepath,'state-before-comparison.json'),(matrixpath,'matrix-before-comparison.json')]:
        backup=RUN/name
        if not backup.exists():backup.write_bytes(path.read_bytes())
    state=json.loads(statepath.read_text());matrix=json.loads(matrixpath.read_text())
    state.update(stage='DPA4_N200_pilot_completed',completed_training_jobs=4,updated_utc=data['utc'],
        latest_progress_snapshot=json.loads((RUN/'latest-progress.json').read_text()),
        first_pilot_progress=json.loads((RUN/'collected-pilots/training-plan01/cases/PBE-lora16/progress.json').read_text()),
        pilot_comparison=str(RUN/'comparison-v1/comparison.json'),
        final_pilot_numerical_QC={name:c['numerical_QC'] for name,c in data['cases'].items()},
        completed_pilots={name:dict(status='completed',steps=c['steps'],slurm_task=c['slurm_task'],model_sha256=c['model_sha256'],merged_weights=c['merged_weights']) for name,c in data['cases'].items()},
        next_actions=['Repeat both DPA4 adaptation candidates with at least three training seeds before selecting a winner.',
                      'Run the matched full eligible train694 control and compare with N200; currently planned, not submitted.',
                      'Keep new test149 excluded from pilot tuning and keep archived benchmark49 separate.'])
    state['current_gates']['FT']='pending_multiseed_and_full_train_confirmation'
    state['jobs']['pilot-physical-diagnostics-v1']=json.loads((RUN/'pilot-physical-diagnostics-v1-submission.json').read_text())
    report='reports/12_n200_pilot_comparison.md'
    if report not in state['reports']:state['reports'].append(report)
    acct=[line.split('|') for line in manifest['accounting'].splitlines()[1:] if line and '.' not in line.split('|')[0]]
    state['pilot_resource_accounting']=dict(GPU_hours=sum(c['allocated_GPU_seconds'] for c in data['cases'].values())/3600,
        allocated_CPU_hours_training_evaluation_QC=sum(int(x[2])*int(x[3]) for x in acct)/3600,
        extra_saved_FC_diagnostic_CPU_hours=29/3600,new_DFT_jobs=0,
        scope='Four GPU array tasks include training/export/predictions/harmonics, plus four metrics and four numerical-QC tasks. Earlier B0/H0/preflight costs are separate.')
    updated=[]
    for design in matrix['training_designs']:
        if design.get('architecture')!='DPA4-Neo' or design.get('train_size')!=200:continue
        name=design['functional']+'-'+design['adaptation']
        if name not in data['cases']:continue
        c=data['cases'][name];p=RUN/'collected-pilots/training-plan01/cases'/name
        rows=list(map(json.loads,(p/'evaluation/metrics/per_structure.jsonl').read_text().splitlines()))
        design.update(state='completed',launch_ready=True,blocked_by=[],
            executed_configuration=str(p/'input.json'),executed_normalized_configuration=str(p/'out.json'),
            training_geometry_ids=sorted(r['id'] for r in rows if r['split']=='train'),
            validation_geometry_ids=sorted(r['id'] for r in rows if r['split']=='validation'),
            trainable_mask=str(p/'parameter-inventory.json'),
            actual_cost=dict(GPU_hours=c['allocated_GPU_seconds']/3600,CPU_hours=c['allocated_CPU_hours'],
                GPU='NVIDIA A100-PCIE-40GB',training_wall_seconds=c['training']['training_seconds'],
                peak_cuda_allocated_bytes=c['training']['peak_cuda_allocated_bytes'],
                scope='This GPU allocation only, including evaluation; additional CPU-only analysis/QC accounted in state.'),
            result=str(RUN/'comparison-v1/comparison.json'),model_sha256=c['model_sha256'],
            numerical_QC='PASS',training_steps=10000,training_seed_executed=10,
            next_step='Keep as a candidate: repeat seeds and matched train694 control. Neither adaptation dominates all target/harmonic criteria; Gate FT remains open.')
        updated.append(name)
    assert set(updated)==set(data['cases'])
    matrix['screening_summary']['completed_DPA4_N200_cases']=4
    matrix['screening_summary']['measured_four_case_GPU_hours']=state['pilot_resource_accounting']['GPU_hours']
    matrix['screening_summary']['latest_result']=report
    statepath.write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    matrixpath.write_text(json.dumps(matrix,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
    summary='''# Ti–Al FM: текущее состояние

Все четыре DPA4 N=200 пилота завершены: PBE/r²SCAN × LoRA16/full FT, по 10 000 шагов, seed 10. Slurm20988454_0–3, метрики и финальные численные проверки PASS. Результаты и финальные merged weights собраны локально с проверкой SHA256. Новых DFT: 0.

На validation149 LoRA лучше по энергии и силам, full FT — по точечным stress RMSE; для PBE различие stress неубедительно. Full FT точнее по спектру и Γ-doublet, LoRA — по Γ c-singlet и поперечной акустической скорости вдоль c. У всех четырёх пилотов нет мнимых мод ниже −0.001 THz на mesh50³, включая raw FC. Исходная FM имела мнимые ветви.

Единого победителя нет. Следующие запланированные шаги: повторы обеих адаптаций на нескольких seeds и контроль на train694. В этом статусном запросе они не запускались; Gate FT остаётся открытым. Новый test149 не использован для сравнения/выбора кандидатов. Benchmark49 архивный.

[Полное сравнение, интервалы и фононы](12_n200_pilot_comparison.md) · [Предыдущий статус B0/H0](11_baseline_pilot_status.md) · [N200](08_n200_pilot.md) · [DFT audit](09_energy_field_audit.md) · [Исходные фононы](10_original_harmonics.md) · [Разбиение](07_grouped_split.md).
'''
    (ROOT/'reports/summary.md').write_text(summary,encoding='utf8')
    print(json.dumps(dict(updated_cases=updated,completed_training_jobs=4,GPU_hours=state['pilot_resource_accounting']['GPU_hours']),indent=2))

if __name__=='__main__':main()
