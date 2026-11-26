"""Record submitted PBE full-train controls and their queued comparison."""
from datetime import datetime,timezone
from prepare_grace_n694 import ROOT,RUN,REMOTE,load,dump

def main():
    plan=load(RUN/'package/plan.json');jobs=load(RUN/'submission-summary.json');gate=load(RUN/'launch-gates.json');assert gate['gate']['status'] in ['PASS','PENDING']
    now=datetime.now(timezone.utc).isoformat();report='reports/29_grace_pbe_n694_launch.md'
    recovery=load(RUN/'storage-recovery/released.json');extra=load(RUN/'storage-baseline-exports/released.json')
    recovery['bytes_released']+=extra['bytes_released']
    lines=['# GRACE PBE: N694 full и restricted + radial, три seeds','',f'Отправлено {now}. Обучение {jobs["train"]}, сравнение {jobs["postprocess"]}. Шесть моделей, максимум одна новая A100 одновременно. r²SCAN не включён в обучение и оценку.','',
      '## Сопоставимость с N200','',
      'Train694 — полный разрешённый grouped train; N200 строго вложен. Все200 общих структур и их метки совпадают точно. Validation149 совпадает побайтно. Группы train/validation не пересекаются; условность реконструкции родителей при утраченных жидкостных траекториях сохраняется. Test149 и архивные49 не используются.','',
      'GRACE-2L-OMAT-medium-base-fp64, прежние исходные FM weights и backend. Каждый запуск начинается с исходной модели и нового Adam, без переноса весов N200. Энергетические сдвиги Al/Ti оценены заново только по train694. Baseline, numerical/export QC и фононы исходной модели пройдены.','',
      f'При submission optimizer gate: {gate["gate"]["status"]}; завершены smoke: '+', '.join(gate['completed_smoke'])+'. Все завершённые smoke PASS, исходные веса идентичны. Production ждёт итоговой проверки20991845 и до обучения требует PASS всех6 smoke. Проверяются frozen tensors, свежий Adam и одинаковые первые5 batches внутри seed. Smoke-веса отбрасываются.','',
      'Adam LR1e-3→1e-4 cosine; Huber E/F/stress16/32/128, delta0.01; batch1, clipnorm1, noEMA, fp64. 50 эпох =34700 updates; N200 имел50 эпох/10000 updates. Это сравнение при одинаковом числе эпох, а не при одинаковом числе обновлений или GPU-времени. Checkpoint фиксирован заранее: финальная эпоха50.','',
      '| Index | Модель | Обучаемых параметров |','|---:|---|---:|']
    for i,c in enumerate(plan['cases']):lines.append(f'| {i} | {c["name"]} | {c["expected_trainable"]} |')
    lines+=['','## Последующая оценка','',
      'Для каждого нового финального checkpoint: numerical/export QC; train694/validation149 E raw и train-aligned, F, stress; фононы TiAl в тех же DFT-ячейках и на том же пути; Ti₃Al total/partial DOS в неизменной архивной Cmcm-геометрии, δ0.005/0.01/0.02Å, mesh14/20 и ASR sensitivity. Ti₃Al сравнивается с сохранённой PBE-DOS, DFT FORCE_SETS отсутствует.','',
      'CPU-сравнение afterany всего массива объединяет6 новых N694 и6 завершённых N200 (full/radial × seeds10/20/30). Проверяет хеши checkpoint и геометрии, membership, frozen tensors, optimizer counters, пересчитывает метрики и выдаёт mean±sampleSD, парные различия N694−N200 и radial−full. При пропусках/ошибках сохраняет INCOMPLETE с причинами. afterany предотвращает вечное ожидание DependencyNeverSatisfied, но не скрывает ошибки.','',
      'Результаты: `'+REMOTE+'/comparison/summary.json` и `comparison/comparison.md`. Локальный статус/сбор: `manage_grace_n694.py status` / `collect`. Численных результатов N694 на момент отправки ещё нет.','',
      'Первый smoke20991836 и gate20991837 завершились ошибкой: baseline успел завершиться раньше архивирования, и quota guard остановил smoke до загрузки модели/создания optimizer. После освобождения квоты smoke повторён; исходные логи сохранены в recovery/smoke-quota-20991836. Порог защиты не снижался.','',
      '## Ресурсы и хранение','',
      'Production:6 × максимум90min,1A100,4CPU,32GiB RAM. Это верхний предел9 GPU-hours, не прогноз фактического расхода; baseline/smoke учитываются отдельно. Финальные модели/экспорты остаются на UMT, Adam checkpoints после успешной оценки сжимаются с проверкой SHA-256.','',
      f'Перед запуском перенесены в локальные проверенные архивы только завершённые промежуточные checkpoints и optimizer archives: освобождено {recovery["bytes_released"]/2**30:.3f}GiB. Перенос исторических r²SCAN-файлов относится только к хранению; новых r²SCAN-расчётов нет. Свободная квота при launch: {gate["quota"]["available_bytes"]/2**30:.3f}GiB.','',
      'Артефакты: `configuration-audit.json`, `local-verification.json`, `remote-preflight.json`, `launch-gates.json`, `submission-summary.json`, `storage-recovery/verified.json` в каталоге запуска.']
    (ROOT/report).write_text('\n'.join(lines)+'\n',encoding='utf8')
    state=load(ROOT/'state.json');state.update(status='GRACE_PBE_N694_three_seed_training_submitted',stage='GRACE_PBE_N694',updated_utc=now,active_training_jobs=6)
    state['grace_pbe_n694']=dict(run=str(RUN),remote_root=REMOTE,jobs=jobs,status='submitted',optimizer=gate['gate']['status'],cases=plan['cases'],PBE_only=True,comparison='PENDING',report=report)
    state['current_gates'].update(GRACE_PBE_N694_optimizer=gate['gate']['status'],GRACE_PBE_N694_comparison='PENDING')
    state['next_actions']=['Collect six GRACE PBE N694 runs and compare with N200 by three seeds: validation, TiAl bands, Ti3Al DOS.','Inspect full stress seed sensitivity and mean/sample SD; do not select a lucky seed.','Keep r2SCAN paused and held-out test149/archival49 closed.']
    if report not in state['reports']:state['reports'].append(report)
    dump(ROOT/'state.json',state)
    matrix=load(ROOT/'protocol/experiment_matrix.yaml')
    for i,c in enumerate(plan['cases']):
        row=dict(architecture='GRACE-2L',functional='PBE',adaptation=c['method'],train_size=694,training_seed=c['seed'],training_precision='float64',epochs=50,expected_steps=34700,state='submitted',slurm_array_job=jobs['train'],slurm_array_index=i,expected_trainable=c['expected_trainable'],submitted_configuration=str(RUN/'package/cases'/c['name']/'input.yaml'),test_access=False,blocked_by=[])
        old=next((d for d in matrix['training_designs'] if all(d.get(k)==row[k] for k in ['architecture','functional','adaptation','train_size','training_seed'])),None)
        if old:old.update(row)
        else:matrix['training_designs'].append(row)
    dump(ROOT/'protocol/experiment_matrix.yaml',matrix)
    (ROOT/'reports/summary.md').write_text('# Ti–Al FM: текущее состояние\n\nОтправлены6 обучений GRACE PBE N694: full и restricted + radial, seeds10/20/30. После завершения автоматически сравниваются с N200 по validation, TiAl-фононам и Ti₃Al-DOS. r²SCAN не включён.\n\n[Запуск N694](29_grace_pbe_n694_launch.md) · [Результаты N200](28_grace_pbe_three_seed_results.md).\n',encoding='utf8')
    print(ROOT/report)
if __name__=='__main__':main()
