"""Record actual queued/running expansion tasks without marking them completed."""
from pathlib import Path
import copy,datetime,json

ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/dpa4-replicates-full-20261106-02'

def load(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')

def main():
    plan=load(RUN/'package/plan.json');jobs=load(RUN/'submission-summary.json')
    pointer=load(RUN/'latest-progress.json');snapshot=load(Path(pointer['snapshot']))
    for path,name in [(ROOT/'state.json','state-before-expansion.json'),(ROOT/'protocol/experiment_matrix.yaml','matrix-before-expansion.json')]:
        if not (RUN/name).exists():(RUN/name).write_bytes(path.read_bytes())
    state=load(ROOT/'state.json');matrix=load(ROOT/'protocol/experiment_matrix.yaml');designs=matrix['training_designs']
    statuses={}
    scheduler={line.split('|')[0]:line.split('|')[1] for line in snapshot['accounting'].splitlines()[1:] if line}
    for case in plan['cases']:
        name=case['name'];n=case['N'];seed=case['seed'];xc=case['XC'];method=case['method'];group=f'N{n}'
        result=snapshot['cases'][name]
        task=jobs[group+'_training']+'_'+str(plan['groups'][group].index(name));slurm=scheduler.get(task,'UNKNOWN')
        status='completed' if result.get('evaluation/summary.json',{}).get('status')=='PASS' and result.get('evaluation/numerical-QC/summary.json',{}).get('status')=='PASS' else ('failed' if slurm in ['FAILED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY'] or slurm.startswith('CANCELLED') else ('evaluating' if slurm=='COMPLETED' else ('running' if slurm=='RUNNING' else 'queued')))
        statuses[name]=dict(state=status,steps=result.get('progress.json',{}).get('steps',0),expected_steps=case['expected_steps'])
        matching=[d for d in designs if d.get('architecture')=='DPA4-Neo' and d.get('functional')==xc and d.get('adaptation')==method and d.get('train_size')==n and d.get('training_seed')==seed]
        if matching:assert len(matching)==1;d=matching[0]
        else:
            template=next(d for d in designs if d.get('architecture')=='DPA4-Neo' and d.get('functional')==xc and d.get('adaptation')==method and d.get('train_size')==200 and d.get('training_seed')==10)
            d=copy.deepcopy(template)
            for k in ['actual_cost','result','model_sha256','numerical_QC','training_steps','training_seed_executed','executed_configuration','executed_normalized_configuration']:d.pop(k,None)
            d['run_id']=f'dpa4-neo-{method}-{xc.lower()}-n{n}-seed{seed}';designs.append(d)
        cfg=load(RUN/'package/training-plan01/cases'/name/'input.json')
        d.update(state=status,launch_ready=True,blocked_by=[],training_seed=seed,train_size=n,
            target_geometry_ids=load(RUN/'package/subsets.json')['subsets'][str(n)],
            training_geometry_ids=sorted(load(RUN/'package/subsets.json')['subsets'][str(n)]),
            validation_geometry_ids=sorted(load(RUN/'package/splits.json')['split_ids']['validation']),
            source_checkpoint=case['source'],submitted_configuration=str(RUN/'package/training-plan01/cases'/name/'input.json'),
            configuration_sha256=case['configuration_sha256'],epochs=50,expected_steps=case['expected_steps'],
            training_systems=cfg['training']['training_data']['systems'],validation_systems=cfg['training']['validation_data']['systems'],
            trainable_mask=str(RUN/'package/runtime-mask-preflight'/f'{method}.json'),
            slurm_array_job=jobs[group+'_training'],slurm_array_index=plan['groups'][group].index(name),
            expected_cost=dict(GPU_hours=(plan['estimated_GPU_hours']['N200_8_additional']/8 if n==200 else plan['estimated_GPU_hours']['N694_4_controls']/4),kind='Measured pilot scaling estimate; queue delay excluded'),
            next_step='Await fixed-final training, train/validation metrics, matched harmonics, final numerical/physical QC and automatic group aggregation. No test-driven checkpoint selection.')
    state.update(stage='DPA4_N200_three_seeds_and_N694_controls_running',updated_utc=snapshot['utc'],
        new_training_jobs=16,completed_training_jobs=4+sum(s['state']=='completed' for s in statuses.values()),
        expansion_run=str(RUN),expansion_status=statuses,latest_expansion_snapshot=pointer,
        next_actions=['Complete eight N200 seed20/30 repetitions and aggregate together with the four completed seed10 pilots.',
                      'Complete four N694 seed10 controls after the N200 GPU array; compare size at seed10.',
                      'Inspect per-case numerical/physical QC and aggregate seed variation before deciding the next experiment.'])
    state['jobs']['replicates_and_full_train']=jobs
    state['engineering_failures']=state.get('engineering_failures',[])
    failure=str(ROOT/'runs/dpa4-replicates-full-20261106-01/failure-diagnosis.json')
    if failure not in state['engineering_failures']:state['engineering_failures'].append(failure)
    state['current_gates']['FT']='pending_N200_three_seed_results_and_N694_control'
    report='reports/13_multiseed_fulltrain_launch.md'
    if report not in state['reports']:state['reports'].append(report)
    matrix['screening_summary'].update(submitted_DPA4_N200_total_seeds=[10,20,30],submitted_DPA4_N694_seeds=[10],additional_unique_training_cases=12,latest_launch_report=report)
    matrix['screening_summary']['current_registered_designs']=len(designs)
    save(ROOT/'state.json',state);save(ROOT/'protocol/experiment_matrix.yaml',matrix)
    text=f'''# Три seeds N200 и полный train694: запуски

Статус на {snapshot['utc']}. Отправлены **8 повторов N200** (seeds20/30) и **4 контроля N694** (seed10), PBE/r²SCAN × LoRA16/full FT. Четыре исходных N200 seed10 завершены ранее и включены в будущую сводку по трём seeds. N694 пока имеет один seed и служит парным контролем размера данных относительно N200 seed10.

| Этап | Slurm job/array | Задач | Зависимость |
|---|---:|---:|---|
| N200 seeds20/30 | {jobs['N200_training']} | 8 GPU | По одной одновременно |
| N200 numerical + mode/FC/acoustic QC | {jobs['N200_QC']} | 8 CPU | После массива N200 |
| Сводка N200 seeds10/20/30 | {jobs['N200_aggregate']} | 1 CPU | После QC |
| N694 seed10 | {jobs['N694_training']} | 4 GPU | После массива N200; по одной одновременно |
| N694 numerical + physical QC | {jobs['N694_QC']} | 4 CPU | После массива N694 |
| Парное сравнение N694/N200 seed10 | {jobs['N694_aggregate']} | 1 CPU | После QC |

Всего одновременно не более одной новой GPU-задачи. CPU-проверки могут перекрываться с последующим обучением. Оценка новых задач — **около14 GPU·ч**, без ожидания очереди; лимит каждой GPU-задачи4ч. Оценка масштабирует измеренное время пилота, а не задаёт искусственную суммарную квоту. Существующие сторонние задания не изменялись.

## Зафиксированный протокол

Одна исходная DPA4-Neo SHA256 `fd7f34ae28f921201e4a0328ddf56179892752084f8363e7796038201471c989`, никакой инициализации дообученным TiAl checkpoint. N200 вложена в train694; validation149 и разбиение общие. Статистики/energy bias пересчитываются отдельно только на train каждого запуска. Пакет не содержит координат или меток test149.

50 nominal epochs во всех случаях: N200→10 000 шагов, N694→34 700. Общие HybridMuon, cosine LR1e-4→1e-6, loss E/F/virial20/20/5, batch auto:128 (одна структура), clip5, rank16/alpha16 для LoRA. Финальный checkpoint предопределён. Таким образом, контроль N694 сохраняет число эпох, но не число шагов и не вычислительный бюджет; оба учитываются в результатах.

Меняется `training.seed`; descriptor/fitting seeds42 сохранены. В установленном backend training seed передаётся в DpLoaderSet, который задаёт PyTorch/CUDA RNG и dp_random. Проверка разных начальных LoRA A-хешей и разных exposure streams включена в агрегацию. Внутри размера/seed сравнивается совпадение порядка optimizer batches между методами/XC.

Все 3636 файлов проверены при передаче. До повторной отправки реально выполнен input-verifier всех12 случаев; проверены hashes данных/checkpoint, counts/IDs, отсутствие test-путей и отсутствие прежних statistics. Backend training.py и lora.py совпадают с аудитом пилота. После инициализации каждого Trainer, до первого шага, проверяются исходные силы/virial, композиционный сдвиг энергии, нулевые B, маска requires_grad и live/merged. Проверка числа шагов явная для каждого N.

## Метки и ограничения

Дополнительный r²SCAN raw-аудит охватил все843 train694+validation149. Все energy совпали с e_fr_energy; максимальная разница сил с XML5e-7eV/Å. Все расчёты завершены нормально с EDIFF, без достижения NELM. Параметры одни: ENCUT550, EDIFF1e-6, ISMEAR1, SIGMA0.1, METAGGA=R2SCAN. Максимальное отличие XML stress от исходной NPY-конвенции virial0.002906eV на ячейку совпадает с ранее исследованным масштабом; метки не заменялись. PBE raw-архив по-прежнему не найден; его SCF/energy field не объявляются повторно проверенными. Новых DFT0.

Родительские liquid-траектории не восстановлены. Используется согласованная консервативная группировка по nTi/nAl и прежняя условная гарантия отсутствия пересечений. Гармонические расчёты на фиксированной DFT-геометрии — physical development, не blind test. Сходимость DFT cutoff/kmesh/supercell и долговременная MD-устойчивость этими опытами не устанавливаются.

## Автоматическое завершение и артефакты

Каждый GPU-запуск сохраняет финальные веса/export, train/validation-предсказания, метрики и фононный спектр/DOS. CPU-этапы повторяют численный QC, сравнивают eigenvectors/подпространства, динамические матрицы/FC, raw/central/ASR и акустические наклоны. Сводка сохраняет все seeds и ошибки выполнения, mean±sample SD и парные разности LoRA−full; не исключает неудавшийся seed молча. CPU-зависимости afterany позволяют сформировать отчёт о сбое вместо вечного ожидания afterok.

Удалённый каталог: `{plan['remote_root']}`. Автоматические отчёты появятся в `aggregates/N200/report.md` и `aggregates/N694/report.md`, полные данные — в соответствующих `summary.json`. Это запланированные выходы, а не уже полученные результаты.

[План и конфигурации](../runs/{RUN.name}/package/plan.json) · [Текущий snapshot](../runs/{RUN.name}/latest-progress.json) · [Slurm submissions](../runs/{RUN.name}/submission-summary.json) · [Raw-аудит N694](../runs/{RUN.name}/label-audit/summary.json) · [Завершённый seed10](12_n200_pilot_comparison.md).

Первичная попытка `dpa4-replicates-full-20261106-01` остановилась до оптимизации на `AttributeError` в проверке строкового пути checkpoint. Все12 попыток выполнили0 optimizer steps, использовав13 GPU-секунд выделения; журналы и автоматические отчёты о сбое сохранены. Исправление — нормализация через Path(p); текущая попытка02 имеет отдельный неизменяемый пакет. [Диагноз](../runs/dpa4-replicates-full-20261106-01/failure-diagnosis.json).

## Фактический прогресс

| Case | Состояние | Шаги |
|---|---|---:|
'''
    for name,s in statuses.items():text+=f"| {name} | {s['state']} | {s['steps']} / {s['expected_steps']} |\n"
    (ROOT/report).write_text(text,encoding='utf8')
    running=[f"{name}: {s['steps']}/{s['expected_steps']}" for name,s in statuses.items() if s['state']=='running']
    summary=f'''# Ti–Al FM: текущее состояние

Четыре N200 seed10 завершены; проверенные результаты: LoRA лучше по validation E/F, full FT — по общей ошибке спектра. Ни один метод не выигрывает по всем физическим критериям.

Продолжение отправлено: восемь N200 seeds20/30 (array{jobs['N200_training']}) и четыре N694 seed10 (array{jobs['N694_training']}). GPU-задачи последовательные. Текущий прогресс: {'; '.join(running) or 'задачи в очереди'}. После обучения автоматически выполняются QC и сводки по seeds/размеру данных. Новых DFT0; test149 не используется для настройки.

[Новые запуски и зависимости](13_multiseed_fulltrain_launch.md) · [Завершённое сравнение seed10](12_n200_pilot_comparison.md) · [Разбиение](07_grouped_split.md).
'''
    (ROOT/'reports/summary.md').write_text(summary,encoding='utf8')
    print(json.dumps(dict(active=running,queued=sum(s['state']=='queued' for s in statuses.values()),completed_new=sum(s['state']=='completed' for s in statuses.values())),indent=2))

if __name__=='__main__':main()
