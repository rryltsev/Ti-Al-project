"""Record actual PBE-only submissions; do not report uncompleted training as results."""
from datetime import datetime,timezone
from prepare_grace_pbe_seeds import ROOT,RUN,REMOTE,load,dump
def main():
    plan=load(RUN/'package/plan.json');jobs=load(RUN/'submission-summary.json');gate=load(RUN/'launch-gates.json');assert gate['gate']['status']=='PASS'
    recovery=load(RUN/'storage-recovery/released.json');now=datetime.now(timezone.utc).isoformat()
    report='reports/27_grace_pbe_three_seed_launch.md'
    lines=['# GRACE N200 PBE: повторы seeds20/30','',f'Запуск {now}. Восемь новых обучений: full, restricted, restricted + MLP, restricted + radial × seeds20/30. Seed10 взят из четырёх завершённых моделей. r²SCAN не включён в новые входные данные и запуски.','',
           f'Обучение: массив {jobs["train"]}, максимум одна A100 одновременно. Постобработка: {jobs["postprocess"]}, afterany всего массива. Если какие-либо результаты отсутствуют, итоговый summary будет INCOMPLETE с перечислением ошибок; постоянной зависимости afterok здесь нет.','',
           '## Зафиксированные настройки','',
           'GRACE-2L-OMAT-medium-base-fp64; прежний PBE checkpoint с тем же train-only выравниванием; train200/validation149, неизменный grouped split; Adam, LR1e-3→1e-4 cosine, Huber E/F/stress16/32/128, batch1, 50эпох/10000updates. Выбор checkpoint: фиксированная финальная эпоха50. Test149 и архивные49 не используются. Единственное научное изменение относительно соответствующего seed10 — seed обучения.','',
           'Все восемь 5-step smoke прошли: исходные тензоры совпадают с source, optimizer свежий, frozen tensors неизменны. Первые пять batches совпадают между четырьмя масками одного seed и различаются между seeds20/30. Production сверяет первые batches со своим smoke; smoke-веса отбрасываются.','',
           '| Index | Case | Обучаемых параметров |','|---:|---|---:|']
    for i,c in enumerate(plan['cases']):lines.append(f'| {i} | {c["name"]} | {c["expected_trainable"]} |')
    lines+=['','## Автоматическое сравнение','',
            'После обучения: numerical/export QC, validation149 с raw и train-aligned energy, F/stress; фононы TiAl в тех же DFT-ячейках; Ti₃Al в той же архивной Cmcm-геометрии, включая total/Al/Ti DOS, δ0.005/0.01/0.02Å, mesh14/20, raw/ASR. Скрипт агрегирования объединяет восемь новых результатов с четырьмя seed10, проверяет хеши моделей и одинаковые IDs, пересчитывает метрики, формирует среднее±sampleSD и парные различия с full по seeds10/20/30.','',
            'Ti₃Al: сравнение с архивной PBE-DOS; FORCE_SETS отсутствует, поэтому DFT-ошибка ветвей не вычисляется. Все физические проверки — development diagnostics. Они не доказывают сходимость DFT, независимость от pretraining или устойчивость длительной MD.','',
            '## Хранение','',f'До запуска освобождено {recovery["bytes_released"]/2**30:.3f}GiB: промежуточные checkpoints завершённых работ перенесены в15 локальных архивов с проверкой каждого SHA-256. Финальные модели и экспорты сохранены. Перед отправкой production подтверждено {gate["quota"]["available_bytes"]/2**30:.3f}GiB свободной квоты.','',
            'Best-validation checkpoints не записываются, поскольку они не участвуют в выборе. Регулярный полный Adam checkpoint сохраняется во время обучения; после успешной оценки он сжимается в optimizer-final.tar.gz, содержимое проверяется по хешам до удаления несжатых дубликатов. Архив остаётся на UMT и позволяет восстановить optimizer state. Контроль квоты выполняется перед стартом и записью.','',
            'Результаты на момент отправки ещё не готовы; новых численных выводов по seeds нет. Итог на UMT: `'+REMOTE+'/comparison/summary.json` и `comparison/comparison.md`. Локальный статус/сбор: `manage_grace_pbe_seeds.py status` / `collect`.','',
            'Артефакты: `runs/'+RUN.name+'/configuration-audit.json`, `launch-gates.json`, `submission-summary.json`, `storage-recovery/verified.json`.']
    (ROOT/report).write_text('\n'.join(lines)+'\n',encoding='utf8')
    state=load(ROOT/'state.json');state.update(status='GRACE_PBE_three_seed_replicates_submitted',stage='GRACE_PBE_N200_replicates',updated_utc=now,active_training_jobs=8)
    state['grace_pbe_repeats']=dict(run=str(RUN),remote_root=REMOTE,jobs=jobs,status='submitted',optimizer='PASS',cases=plan['cases'],PBE_only=True,comparison='PENDING',report=report)
    state['current_gates'].update(GRACE_PBE_seed_optimizer='PASS',GRACE_PBE_three_seed_comparison='PENDING')
    state['next_actions']=['Collect eight PBE GRACE repeats and verify the combined three-seed validation/TiAl/Ti3Al comparison.','Keep r2SCAN out of new tests; do not access held-out test149 during selection.']
    if report not in state['reports']:state['reports'].append(report)
    dump(ROOT/'state.json',state)
    matrix=load(ROOT/'protocol/experiment_matrix.yaml')
    for i,c in enumerate(plan['cases']):
        row=dict(architecture='GRACE-2L',functional='PBE',adaptation=c['method'],train_size=200,training_seed=c['seed'],training_precision='float64',epochs=50,expected_steps=10000,state='submitted',slurm_array_job=jobs['train'],slurm_array_index=i,expected_trainable=c['expected_trainable'],submitted_configuration=str(RUN/'package/cases'/c['name']/'input.yaml'),test_access=False,blocked_by=[])
        old=next((d for d in matrix['training_designs'] if all(d.get(k)==row[k] for k in ['architecture','functional','adaptation','train_size','training_seed'])),None)
        if old:old.update(row)
        else:matrix['training_designs'].append(row)
    dump(ROOT/'protocol/experiment_matrix.yaml',matrix)
    (ROOT/'reports/summary.md').write_text('# Ti–Al FM: текущее состояние\n\nОтправлены8 повторов GRACE N200 PBE seeds20/30 для четырёх масок. После завершения автоматически сравниваются validation, TiAl-фононы и Ti₃Al-DOS с seed10. r²SCAN не включён.\n\n[Запуск трёх seeds](27_grace_pbe_three_seed_launch.md) · [Ti₃Al seed10](26_ti3al_pbe_phonons.md) · [Маски GRACE seed10](25_grace_mask_ablation_results.md).\n',encoding='utf8')
    print(ROOT/report)
if __name__=='__main__':main()
