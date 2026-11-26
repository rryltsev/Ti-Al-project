"""Record the authorized mask-only ablation without replacing parent results."""
import json,datetime
from pathlib import Path
from prepare_grace_mask_ablation import ROOT,RUN,REMOTE,OUT

def load(p):return json.loads(p.read_text(encoding='utf8'))
def dump(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf8')

def main():
    plan=load(OUT/'plan.json');jobs={p.stem:load(p)['job_id'] for p in (RUN/'submissions').glob('*.json')};status=load(RUN/'latest-status.json')
    state=load(ROOT/'state.json');matrix=load(ROOT/'protocol/experiment_matrix.yaml')
    for p in [ROOT/'state.json',ROOT/'protocol/experiment_matrix.yaml']:
        backup=RUN/(p.name+'.before-ablation')
        if not backup.exists():backup.write_bytes(p.read_bytes())
    state.update(status='GRACE_N200_mask_ablation_queued',stage='GRACE_restricted_plus_MLP_and_radial',updated_utc=status['utc'],new_training_jobs=32,active_training_jobs=4)
    state['grace_mask_ablation']=dict(run=str(RUN),remote_root=REMOTE,jobs=jobs,cases=plan['cases'],parent_run=plan['parent_run'],configuration_audit='PASS',remote_preflight='PASS',optimizer=status['files'].get('optimizer-gate.json',{}).get('status','PENDING'),comparison=status['files'].get('comparison/summary.json',{}).get('status','PENDING'))
    state['current_gates']['GRACE_mask_optimizer']=state['grace_mask_ablation']['optimizer']
    state['next_actions']=['Check smoke20990673 and optimizer-gate20990674; pilot20990675 and phonons20990676 follow by afterok.','Collect comparison20990677 when complete; independently reproduce validation and phonon metrics against the full/restricted parent models.','Keep test149 and archival49 outside tuning; only seed10 is authorized for this mask ablation.']
    report='reports/23_grace_mask_ablation_launch.md'
    if report not in state['reports']:state['reports'].append(report)
    for i,c in enumerate(plan['cases']):
        if any(d.get('architecture')=='GRACE-2L' and d.get('adaptation')==c['method'] and d.get('functional')==c['XC'] and d.get('train_size')==200 for d in matrix['training_designs']):continue
        matrix['training_designs'].append(dict(architecture='GRACE-2L',train_size=200,functional=c['XC'],adaptation=c['method'],state='submitted_with_optimizer_gate',training_seed=10,training_precision='float64',epochs=50,expected_steps=10000,base_lr=.001,trainable_patterns=c['trainable_patterns'],expected_trainable=c['expected_trainable'],slurm_array_job=jobs['train'],slurm_array_index=i,submitted_configuration=str(OUT/'cases'/c['name']/'input.yaml'),blocked_by=['GRACE_mask_optimizer'],test_access=False))
    dump(ROOT/'state.json',state);dump(ROOT/'protocol/experiment_matrix.yaml',matrix)
    lines=['# GRACE N200: restricted + MLP / restricted + radial','',f'Снимок: {status["utc"]}. Четыре новых теста поставлены в Slurm.','',
        '| Этап | Job | Зависимость |','|---|---|---|',f'| Проверки optimizer, 5 шагов | {jobs["smoke"]} | Доступная A100 |',f'| Общий optimizer gate | {jobs["optimizer-gate"]} | Все smoke afterok |',f'| Обучения | {jobs["train"]} | Gate afterok |',f'| Фононы | {jobs["trained-harmonics"]} | Все обучения afterok |',f'| Сравнение 8 моделей | {jobs["comparison"]} | Все фононы afterok |','',
        '```',status['queue'].strip(),'```','',
        '| Индекс | XC | Маска | Дополнительные параметры | Всего trainable scalars |','|---:|---|---|---:|---:|']
    for i,c in enumerate(plan['cases']):lines.append(f'| {i} | {c["XC"]} | {c["method"]} | {c["expected_trainable"]-21111690} | {c["expected_trainable"]} |')
    lines+=['','Базовая restricted-маска: I2/reducing_, rho/reducing_, I1/reducing_. Вариант MLP добавляет только LinMLPOut2ScalarTarget_; radial добавляет только R_Linear_ и R1_Linear_. Варианты независимы; совместного размораживания MLP+radial здесь нет.','',
        'Прочие настройки неизменны: тот же исходный GRACE-2L-OMAT-medium-base-fp64 checkpoint; train200/validation149; seed10; batch1; 50 эпох /10000 steps; Adam LR1e-3→1e-4 cosine; Huber E/F/stress16/32/128; clipnorm1; noEMA; fixed final checkpoint. Изменение только маски подтверждено configuration-audit.json.','',
        'Baseline B0/H0 и train-only alignment повторно используются с проверкой хешей. Aligned checkpoint скопированы внутри UMT в новый каталог; parent-модели и завершённые результаты не менялись. До первого шага проверяется полное совпадение initial parameter signatures, затем совпадение первых5 batch с parent-пилотами внутри XC. Frozen tensors контролируются на первом шаге, каждые1000 и на финальном шаге. Smoke-веса отбрасываются; научные пилоты начинают заново.','',
        'Сравнение выполняется CPU-job после фононов: raw/aligned validation E, F, stress; sorted-band RMSE; Γ doublet/singlet; мнимые моды mesh50³; время и peak TensorFlow allocator. Для каждой модели проверяются финальные хеши, 10000 шагов, export/numerical QC и 11 проверок frozen weights. Все восемь моделей должны иметь одинаковые train/validation IDs и первые5 batch внутри XC.','',
        'Автоматический результат будет сохранён на UMT в '+REMOTE+'/comparison/summary.json и comparison/comparison.md. Сбор: scripts/manage_grace_pilot.py collect --run '+RUN.name+' --collection-name collected-final. Статус: та же команда с action status --brief.','',
        'На момент записи новые модели ещё не оценены. Текущее сравнение full/restricted seed10: [отчёт22](22_grace_n200_results.md). Новые тесты тоже используют только seed10; разброс по seeds этим запуском не измеряется. Test149 и архивные49 не используются.']
    (ROOT/report).write_text('\n'.join(lines)+'\n',encoding='utf8')
    (ROOT/'reports/summary.md').write_text('# Ti–Al FM: текущее состояние\n\nЧетыре GRACE N200 mask ablations поставлены в очередь: restricted+MLP и restricted+radial для PBE/r²SCAN. Jobs '+jobs['smoke']+' → '+jobs['optimizer-gate']+' → '+jobs['train']+' → '+jobs['trained-harmonics']+' → '+jobs['comparison']+'.\n\n[Новая серия](23_grace_mask_ablation_launch.md) · [Завершённые full/restricted](22_grace_n200_results.md).\n',encoding='utf8')
    print(json.dumps(dict(jobs=jobs,report=report)))

if __name__=='__main__':main()
