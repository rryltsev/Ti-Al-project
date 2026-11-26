from pathlib import Path
import datetime,json
from prepare_grace_pilot import ROOT,RUN,REMOTE
def load(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def main():
    status=load(RUN/'latest-status.json');jobs={p.stem:load(p)['job_id'] for p in (RUN/'submissions').glob('*.json')};plan=load(RUN/'package/plan.json')
    state=load(ROOT/'state.json');matrix=load(ROOT/'protocol/experiment_matrix.yaml')
    for p,name in [(ROOT/'state.json','state-before-pilot.json'),(ROOT/'protocol/experiment_matrix.yaml','matrix-before-pilot.json')]:
        if not (RUN/name).exists():(RUN/name).write_bytes(p.read_bytes())
    b0=status['files'].get('baseline/summary.json',{}).get('status','PENDING');h0=status['files'].get('baseline/harmonics/summary.json',{}).get('status','PENDING');opt=status['files'].get('optimizer-gate.json',{}).get('status','PENDING')
    state.update(status='GRACE_fp64_N200_pipeline_active',stage='GRACE_B0_H0_alignment_optimizer_and_N200_pilot',updated_utc=status['utc'],new_training_jobs=28,active_training_jobs=4)
    state['grace_pilot']=dict(run=str(RUN),jobs=jobs,cases=plan['cases'],source_checkpoint=plan['source_checkpoint'],B0=b0,H0=h0,optimizer=opt,selection='fixed final epoch50 / step10000',test_access=False)
    state['current_gates'].update(GRACE_B0=b0,GRACE_H0=h0,GRACE_optimizer=opt)
    state['next_actions']=['Allow GRACE baseline/H0 -> optimizer smoke gate -> four fp64 N200 pilots to proceed through recorded Slurm dependencies.','Collect baseline/alignment/optimizer evidence and pilot final validation/phonons; investigate any failed dependency before retry.','Keep test149/archival49 outside tuning.']
    for i,c in enumerate(plan['cases']):
        d=next(d for d in matrix['training_designs'] if d['architecture']=='GRACE-2L' and d['train_size']==200 and d['functional']==c['XC'] and d['adaptation']==c['method'])
        d.update(state='submitted_with_prerequisite_gates',launch_ready=False,training_precision='float64',training_seed=10,epochs=50,base_lr=.001,expected_steps=10000,slurm_array_job=jobs['train'],slurm_array_index=i,
            submitted_configuration=str(RUN/'package/cases'/c['name']/'input.yaml'),
            blocked_by=['GRACE_B0','GRACE_H0','GRACE_optimizer_gate'],
            next_step='Scheduler starts only after baseline/alignment/H0 and optimizer checks pass; fresh weights after smoke, final-epoch checkpoint.',
            planned_resources=dict(account='imet-2026-2',partition='v100',GPUs=1,CPUs=4,memory_GiB=40,wall_limit_hours=4,scope='planning cap; optimizer timing profile pending'))
        d['blocked_by']=[name for name,value in [('GRACE_B0',b0),('GRACE_H0',h0),('GRACE_optimizer_gate',opt)] if value!='PASS']
        d['launch_ready']=not d['blocked_by']
        if d['launch_ready']:
            d['state']='running' if any(jobs['train']+'_'+str(i) in line and 'RUNNING' in line for line in status.get('queue','').splitlines()) else 'queued_ready'
            d['planned_resources']['scope']='Submitted cap; optimizer smoke completed, full training wall time not yet measured.'
    recovered='smoke-checkpoint-path-v1' in jobs
    active_smoke=jobs.get('smoke-checkpoint-path-v1',jobs['smoke'])
    if (RUN/'active-recovery.json').exists():active_smoke=load(RUN/'active-recovery.json')['smoke']
    state['grace_pilot']['active_smoke_job']=active_smoke
    report='reports/21_grace_checkpoint_path_repair.md' if recovered else ('reports/20_grace_alignment_repair.md' if '20261109' in RUN.name else 'reports/19_grace_fp64_n200_launch.md')
    if report not in state['reports']:state['reports'].append(report)
    save(ROOT/'state.json',state);save(ROOT/'protocol/experiment_matrix.yaml',matrix);save(RUN/'submission-summary.json',dict(jobs=jobs,max_concurrent_GPUs=1,prerequisite_gates_required=True))
    text=f'''# GRACE N=200 fp64: baseline → optimizer → четыре пилота

Состояние на {status['utc']} UTC. Все этапы отправлены в Slurm. Подтверждённые статусы: B0/alignment={b0}, H0={h0}, optimizer={opt}. Текущая очередь приведена ниже; ещё не выполненные gates не объявляются PASS.

```
{status.get('queue','')}
```

| Этап | Job | Условие запуска |
|---|---|---|
| Исходная GRACE: B0, export и alignment | {jobs['baseline']} | GPU priority/resources |
| Исходные фононы H0 | {jobs['H0']} | baseline COMPLETED |
| Четыре optimizer smoke-проверки | {active_smoke} | H0 COMPLETED |
| Общий optimizer gate | {jobs['optimizer-gate']} | все smoke COMPLETED |
| Четыре пилота fp64 | {jobs['train']} | optimizer gate COMPLETED/PASS |
| Итоговые фононы пилотов | {jobs['trained-harmonics']} | все пилоты COMPLETED |

Массивы GPU выполняются с concurrency1. Имеющиеся задания других исследований не менялись.

## Исходная модель и данные

GRACE-2L-OMAT-medium-base-fp64, тот же checkpoint revision4aff666029967745f7993f348eb87a804d625cff и архив SHA763b7e5679c81ea82d39e595e701d0c9f6565064393f705cbe76663c9583d0aa. Полные89 элементов сохранены. Источники неизменны; локальные и staged входы/код проверены SHA-256. Синтаксис Python, bash и встроенного Python gate проверен отдельно.

PBE и r²SCAN: train200 и validation149 из общего grouped split. Координаты и E/F/stress сохранены с17 значащими цифрами; ранее проверено точное обратное чтение. Backend test_filename означает только validation149. Test149 и архивные49 не передаются обучению.

## Численные gates

B0 включает конечность предсказаний, pure-Al/pure-Ti/alloy probes, перестановки атомов, переносы/PBC, вращения E/F/stress, force finite differences на трёх шагах и stress FD по всем шести Voigt-компонентам. Native/export предсказания сравниваются на тех же probes.

Для каждого XC рассчитываются исходные предсказания всех200 train и149 validation. Две поправки Al/Ti определяются равновесным по структурам least squares в E/atom строго по train200. Validation не участвует в подгонке. Поправки встраиваются штатным ConstantScaleShiftTarget; проверяется равенство E_aligned=E_source+nAl*bAl+nTi*bTi при неизменных F/stress. Сохраняются отдельные aligned model.yaml и model-only checkpoint; исходные веса не перезаписываются.

H0 заново получает силы на force000/±0.01Å для PBE/r²SCAN DFT-ячеек, затем использует общий central/symfc/ASR protocol, Γ, bands, DOS и mesh50³. Это исходная гармоническая baseline; DFT convergence этими проверками не сертифицируется.

## Optimizer и рецепт пилота

Native Adam, fp64, LR1e−3→1e−4 cosine, Huber energy/forces/stress weights16/32/128, batch1,50epochs, безEMA. Единый train/test bucket уменьшает число XLA-компиляций; задано200 update batches на эпоху. Native scheduler key исправлен на minimum_learning_rate. Смещение энергий фиксируется до сравнения full/restricted.

Для каждой из четырёх конфигураций выполняются5 smoke optimizer updates с тем же рецептом. Проверяется model-only restore, нулевые epoch/step/Adam iterations и свежие slots до первого шага; затем конечность результатов, изменение обучаемых тензоров, отсутствие изменения frozen tensors, фактическая fp64-маска и GPU memory/step timings. Дополнительно проверяются одинаковые первые5 batch внутри XC для full/restricted. Smoke-веса отбрасываются: научный пилот стартует заново из aligned source checkpoint.

Restricted использует штатные I2/reducing_, rho/reducing_, I1/reducing_: ожидается21111690 trainable scalars против21764956 full, около97%. Это ограничение tensor groups, а не сильное сжатие модели.

Все пилоты идут10000 шагов/50epochs. Сохраняется фиксированный final checkpoint. Native автоматическая загрузка best-validation checkpoint после обучения отключена наблюдателем; факт фиксируется в validation-selection.json. Во время обучения проверяется frozen mask на первом шаге и каждые1000 шагов. После обучения: final native/export numerical QC, train/validation метрики и фононы. Дополнительное train-only энергетическое выравнивание при отчётной оценке сохраняется отдельно от raw calibrated errors.

| Task index | XC | FT | N | Seed | Точность |
|---:|---|---|---:|---:|---|
| 0 | PBE | full | 200 | 10 | fp64 |
| 1 | PBE | restricted | 200 | 10 | fp64 |
| 2 | r²SCAN | full | 200 | 10 | fp64 |
| 3 | r²SCAN | restricted | 200 | 10 | fp64 |

Лимит каждого пилота4ч/1A100/4CPU/40GiB — planning cap; измеренный training forecast появится после optimizer gate. Нельзя переносить измеренное ускорение inference fp32/fp64 на backprop/Adam.

Отслеживание: scripts/manage_grace_pilot.py status. Evidence сохраняется в baseline/, aligned/, smoke/, optimizer-gate.json и cases/. Следующие вычисления выполнятся автоматически по Slurm afterok-зависимостям, если все предыдущие проверки успешны. При сбое обучение останется заблокировано; неизменённый gate не обходится.
'''
    import re
    text=re.sub(r'(?<=[а-яА-ЯёЁ])(?=\d)', ' ',text)
    if '20261109' in RUN.name:
        text+='\n## Исправление предыдущей попытки\n\nBaseline20990460 завершился FAILED на проверке E_aligned=E_source+composition*offset, после PASS исходного numerical QC. H0 имел afterok-зависимость от failed job и получил DependencyNeverSatisfied. Обучение не начиналось. Старые зависимые20990461/20990464/20990465/20990466/20990467 отменены, логи сохранены.\n\nПричина: в моём коде штатному ConstantScaleShiftTarget передавался разреженный словарь только Al/Ti (индексы2 и78); конструктор упаковывает отсортированные значения без сохранения пропущенных индексов. Исправление передаёт полную89-элементную карту с нулями вне Al/Ti. Добавлено сохранение фактических E/F/stress различий до assertion. Исходный pretrained checkpoint не менялся.\n'
    if recovered:
        text+='\n## Повтор optimizer после исправления путей checkpoint\n\nВсе четыре проверки массива 20990595 остановились до первого шага Adam: строковое сравнение /home6/ и /misc/home6/ ошибочно отвергало один и тот же checkpoint. Теперь оба префикса приводятся к Path.resolve(); проверяются наличие index, исходные SHA-256 и различимость checkpoint PBE/r2SCAN. Старые smoke-каталоги, код, manifest и зависимости сохранены в recovery/checkpoint-path-v1; локальные журналы — в collected/smoke.\n\nНовый массив '+active_smoke+' запускается с ограничением одной GPU одновременно. Зависимость существующего gate 20990596 переназначена через scontrol на afterok:'+active_smoke+'. Массив обучения 20990597 по-прежнему ждёт gate, фононы 20990598 ждут все обучения. Обхода проверок нет.\n'
        if 'smoke-tensor-identity-v2' in jobs:
            text+='\nПовтор 20990621 подтвердил PASS канонических путей, затем выявил коллизию имён различных TensorFlow variables в диагностическом inventory до первого optimizer step. Inventory и контроль frozen weights теперь адресуют каждый объект отдельным ключом index:name; маска trainable по-прежнему использует identity объектов. Предыдущая попытка полностью сохранена в recovery/tensor-identity-v2.\n'
        if opt=='PASS':
            text+='\nВсе четыре optimizer smoke и общий gate завершились PASS: по 5 обновлений Adam, свежий optimizer, неизменность frozen tensors, совпадение первых пяти batches между full/restricted внутри каждого XC. Контроль путей PASS во всех четырёх случаях. Текущее состояние пилотов отражено в снимке очереди выше.\n'
    (ROOT/report).write_text(text,encoding='utf8')
    (ROOT/'reports/summary.md').write_text('# Ti–Al FM: текущее состояние\n\nGRACE N200 fp64 пилоты: массив '+jobs['train']+' после baseline/alignment/H0 и optimizer gate. Подтверждено: B0='+b0+', H0='+h0+', optimizer='+opt+'.\n\n[Текущая цепочка]('+Path(report).name+') · [N694 и A100 сравнение](18_final_n694_grace_a100.md).\n',encoding='utf8')
    print(json.dumps(dict(jobs=jobs,B0=b0,H0=h0,optimizer=opt,report=report)))
if __name__=='__main__':main()
