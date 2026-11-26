"""Reconcile the active N694 repetitions and the separate GRACE preparation stage."""
from pathlib import Path
import copy,datetime,json
from prepare_n694_repeats import RUN,REMOTE,ROOT
from package_grace_preparation import RUN as GRACE

def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')

def main():
    state=read(ROOT/'state.json');matrix=read(ROOT/'protocol/experiment_matrix.yaml')
    for src,name in [(ROOT/'state.json','state-before-launch.json'),(ROOT/'protocol/experiment_matrix.yaml','matrix-before-launch.json'),(ROOT/'reports/summary.md','summary-before-launch.md')]:
        dest=RUN/name
        if not dest.exists():dest.write_bytes(src.read_bytes())
    plan=read(RUN/'package/plan.json');jobs=read(RUN/'submission-summary.json')
    pointer=read(RUN/'latest-progress.json');snapshot=read(Path(pointer['snapshot']))
    for i,c in enumerate(plan['cases']):
        rid=f"dpa4-neo-{c['method']}-{c['XC'].lower()}-n694-seed{c['seed']}"
        matches=[d for d in matrix['training_designs'] if d['run_id']==rid]
        if matches:d=matches[0]
        else:
            parent=next(d for d in matrix['training_designs'] if d['architecture']=='DPA4-Neo' and d['train_size']==694 and d['training_seed']==10 and d['functional']==c['XC'] and d['adaptation']==c['method'])
            d=copy.deepcopy(parent);matrix['training_designs'].append(d)
        for k in ['result','model_sha256','executed_configuration','executed_normalized_configuration','numerical_QC','trainable_mask','estimated_cost']:d.pop(k,None)
        progress=snapshot['cases'][c['name']];steps=progress.get('progress.json',{}).get('steps',0)
        d.update(run_id=rid,state='running' if steps else 'submitted',training_seed=c['seed'],training_steps=steps,
            configuration_sha256=c['configuration_sha256'],submitted_configuration=str(RUN/'package/training-plan01/cases'/c['name']/'input.json'),
            slurm_array_job=jobs['N694_training'],slurm_array_index=i,slurm_task=jobs['N694_training']+'_'+str(i),
            expected_steps=34700,blocked_by=[],launch_ready=True,
            planned_resources=dict(partition='v100',account='imet-2026-2',gpus=1,cpus=4,memory_GiB=40,wall_limit_hours=4),
            actual_cost=dict(GPU_hours=None,CPU_hours=None,status='not_final'),
            expected_cost=dict(GPU_hours=21.7083333333/8,kind='Twice measured N694 seed10 total divided among eight repetitions; queue excluded.'),
            runtime_inventory_remote=REMOTE+'/training-plan01/cases/'+c['name']+'/parameter-inventory.json',
            next_step='Automatic final-checkpoint validation/harmonics/QC, then N694 seeds10/20/30 aggregate; no test selection.')
        write(ROOT/'configs'/RUN.name/(rid+'.json'),d)
    assert len([d for d in matrix['training_designs'] if d['architecture']=='DPA4-Neo'])==24
    prep=read(GRACE/'package/preparation-plan.json');card=read(GRACE/'package/model-card.json')
    qpath=GRACE/'collected-preflight/summary.json';qc=read(qpath) if qpath.exists() else None
    qjob=read(GRACE/'preflight-submission.json') if (GRACE/'preflight-submission.json').exists() else None
    for d in matrix['training_designs']:
        if d['architecture']=='GRACE-2L':
            d.update(state='prepared_pending_gates' if d['train_size']==200 else 'planned_pending_pilot',
                model_name=card['model'],model_hash=card['archive_sha256'],model_revision=card['revision'],model_card=str(GRACE/'package/model-card.json'),
                launch_ready=False,blocked_by=['GRACE_B0','GRACE_H0','first_optimizer_step_mask_and_fresh_state','A100_runtime_profile'],
                trainable_mask=str(qpath) if qc else None,
                trainable_parameter_patterns=prep['restricted_patterns'] if d['adaptation']=='restricted' else None,
                next_step='Qualify original GRACE on the same numerical/phonon protocol, then bounded GPU optimizer/profile check before four N200 pilots.')
            if d['train_size']==200:
                d.update(base_lr=.001,epochs=50,draft_configuration=str(GRACE/'package/draft-configs'/f"{d['functional']}-{d['adaptation']}-N200-seed10/input.yaml"),data_preflight=str(GRACE/'data-preflight.json'))
            else:d['blocked_by']+=['N200_pilot','GRACE_N694_data_conversion']
    for s in matrix['stages']:
        if s['id']=='B0-runtime':s['architecture']='DPA4-Neo'
        if s['id']=='H0-GRACE':s.update(state='prepared_pending_baseline',blocked_by=['GRACE_B0'],checkpoint=card['model'],preparation_run=str(GRACE))
    if not any(s['id']=='B0-GRACE' for s in matrix['stages']):
        matrix['stages'].append(dict(id='B0-GRACE',state='prepared_pending_execution',work='Original GRACE numerical/derivative/export checks; separate from passed DPA4 gate',preparation_run=str(GRACE)))
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
    state.update(status='DPA4_N694_repeats_active_GRACE_prepared',stage='DPA4_N694_three_seed_extension_and_GRACE_preparation',updated_utc=utc,updated_local_date='2026-11-07',
        new_training_jobs=24,completed_training_jobs=16,active_training_jobs=8,
        current_run=str(RUN),n694_repeats_run=str(RUN),n694_repeats_jobs=jobs,n694_repeats_snapshot=pointer,
        grace_preparation=dict(run=str(GRACE),model=card['model'],checkpoint_sha256=card['archive_sha256'],revision=card['revision'],data='N200/validation149 PBE+r2SCAN roundtrip PASS',runtime_preflight=qc['status'] if qc else 'pending',preflight_job=qjob['job_id'] if qjob else None,training_submitted=False),
        next_actions=['Collect N694 seeds20/30 and three-seed aggregate after completion; compare target/phonon tradeoff.',
            'Run GRACE-specific B0/H0 and fresh-optimizer/mask/profile checks; then four N200 full/restricted pilots.',
            'Keep held-out test149 and archival49 outside method selection.'])
    state['current_gates'].update(B0='PASS (DPA4 only)',H0='PASS (DPA4 only)',GRACE_B0='pending',GRACE_H0='pending',GRACE_runtime_mask=qc['status'] if qc else 'pending',FT='N200_three_seed_PASS; N694_seeds20/30_running; no_unique_method_choice')
    report='reports/15_n694_repeats_grace_preparation.md'
    if report not in state['reports']:state['reports'].append(report)
    matrix['screening_summary'].update(active_DPA4_N694_cases=8,planned_DPA4_N694_seeds=[10,20,30],GRACE_preparation=str(GRACE),latest_launch=report)
    write(ROOT/'state.json',state);write(ROOT/'protocol/experiment_matrix.yaml',matrix)
    lines=['# N=694: seeds 20/30; подготовка GRACE-2L','',f'Обновлено UTC: {utc}. Предыдущие16 DPA4 моделей завершены; добавлены8 новых N694 повторов.','',
        f'GPU array **{jobs["N694_training"]}**, CPU QC **{jobs["N694_QC"]}**, итоговая агрегация **{jobs["N694_aggregate"]}**. Максимум одна новая GPU-задача одновременно. Оценка8 повторов21.71 GPU·ч основана на фактических10.854 GPU·ч четырёх seed10; очередь в оценку не входит. Лимит каждого запуска4ч,4CPU,40GB.', '',
        '| Task | XC | Метод | Seed | Последний подтверждённый шаг |','|---|---|---|---:|---:|']
    for i,c in enumerate(plan['cases']):
        p=snapshot['cases'][c['name']];steps=p.get('progress.json',{}).get('steps',0)
        lines.append(f'| {jobs["N694_training"]}_{i} | {c["XC"]} | {c["method"]} | {c["seed"]} | {steps}/34700 |')
    lines += ['', 'Рецепт N694 seed10 сохранён:50 epochs, исходный DPA4-Neo, LR1e-4→1e-6, HybridMuon, LoRA rank/alpha16 или full FT. Отличаются training seed и пути. Все1931 staged-файла и8 input-verifiers проверены; завершённые seed10 checkpoint хеши подтверждены. Начальный PBE–LoRA seed20: zero-step PASS,22 adapter-тензора и порядок первых100 batch отличаются от seed10; исходные pretrained-тензоры совпадают.',
        '', 'После каждого обучения автоматически выполняются validation149, гармоники и метрики. После массива — numerical QC, физические диагностики и агрегация всех12 N694 моделей: mean/sample SD, парные LoRA−full разности, проверка независимых seeds. Отдельно сохраняется парный seed10 контроль размера N200→694.','',
        '## Подготовка GRACE','',f'Выбран **{card["model"]}**, revision `{card["revision"]}`, SHA-256 архива `{card["archive_sha256"]}`. Окружение TensorPotential0.6.1 / TensorFlow2.20.0. Фиксируется medium/base/OMat24/fp64; выводы не переносятся на FS, SMAX или другие размеры.',
        '', 'Старая small GRACE-2L-OMAT была скачана при инвентаризации и сохранена как невыбранный кандидат; она не используется в pilot package. Официальная документация помечает small как deprecated. [Документация foundation models](https://gracemaker.readthedocs.io/en/latest/gracemaker/foundation/).',
        '', 'Подготовлены4 native draft inputs: PBE/r²SCAN × full/restricted, N200,seed10,50epochs. Ограниченная маска использует штатный substring-фильтр `I2/reducing_`, `rho/reducing_`, `I1/reducing_`; собственной схемой LoRA она не называется. Все элементы исходной модели сохранены, energy shift пока не применяется.',
        '', 'NPY→extxyz:200 train и149 validation для каждого XC. 17значащих цифр, обратное чтение ASE даёт нулевую численную разницу E/F/cell/positions/symmetric stress. Stress=−sym(virial)/volume, eV/Å³. GRACE `test_filename` указывает только на validation149; held-out test149 отсутствует в пакете.',
        '',f'CPU preflight job: {qjob["job_id"] if qjob else "не отправлен"}; статус: {qc["status"] if qc else "ожидается"}.']
    if qc:lines += [f'Фактические обучаемые скаляры: full **{qc["full_trainable"]:,}**, restricted **{qc["restricted_trainable"]:,}** ({100*qc["restricted_trainable"]/qc["full_trainable"]:.2f}% full). Маска содержит12 тензоров и замораживает только около3% исходных trainable параметров: это ограничение типов обучаемых тензоров, а не аналог экономии числа параметров DPA4 LoRA. Native mask совпадает с ожидаемым списком и не изменяет значения весов; backend успешно прочитал все4 файла данных.',
        'Первая CPU-проверка20989421 завершилась ошибкой проверочного хеша строковых TensorFlow metadata: numpy object-array.tobytes() содержит адреса объектов. Попытка и лог сохранены. Во второй попытке20989422 хешируются значения строк, все численные веса остаются неизменны; PASS за10с Slurm. Обучение и optimizer steps в обеих попытках отсутствуют. Foundation checkpoint не содержит optimizer slots; при будущей загрузке сохраняется требование fresh Adam и сброса epoch/step.']
    lines += ['', 'GRACE training ещё не отправлен. Следующие проверки: собственные B0/H0, нулевой шаг и frozen tensors после реального optimizer-step, свежий Adam без foundation optimizer state, короткое профилирование A100. Черновой GRACE Huber recipe E/F/stress16/32/128, LR1e-3→1e-4, batch1 требует этого профиля; он не объявляется эквивалентным DPA4 MAE E/F/virial. Выбор фиксированного final epoch50, без best-validation checkpoint.',
        '', 'Новых DFT0. Test149 и архивные49 не используются для настройки. Фононная устойчивость не заменяет длинный MD-тест.',
        '', '[Предыдущее сравнение16 моделей](14_multiseed_fulltrain_comparison.md).']
    text='\n'.join(lines)+'\n'
    for old,new in {
        'Предыдущие16':'Предыдущие 16','добавлены8':'добавлены 8','Оценка8':'Оценка 8','повторов21.71':'повторов 21.71',
        'фактических10.854':'фактических 10.854','seed10':'seed 10','seed20':'seed 20','размер4':'размер 4',
        'запуска4ч,4CPU,40GB':'запуска 4 ч, 4 CPU, 40 GB','сохранён:50':'сохранён: 50','LR1e':'LR 1e',
        'rank/alpha16':'rank/alpha 16','Все1931':'Все 1931','и8 input':'и 8 input','PASS,22':'PASS, 22',
        'первых100':'первых 100','validation149':'validation 149','всех12':'всех 12','seeds10/20/30':'seeds 10/20/30',
        'N200→694':'N=200→694','TensorPotential0.6.1':'TensorPotential 0.6.1','TensorFlow2.20.0':'TensorFlow 2.20.0',
        'Подготовлены4':'Подготовлены 4','N200,seed 10,50epochs':'N=200, seed 10, 50 epochs','extxyz:200':'extxyz: 200',
        '17значащих':'17 значащих','test149':'test 149','содержит12':'содержит 12','около3%':'около 3%',
        'все4':'все 4','CPU-проверка209':'CPU-проверка 209','попытке209':'попытке 209','за10с':'за 10 с',
        'E/F/stress16/32/128':'E/F/stress 16/32/128','batch1':'batch 1','epoch50':'epoch 50','DFT0':'DFT 0',
        'Test149':'Test 149','архивные49':'архивные 49','сравнение16':'сравнение 16'
    }.items():text=text.replace(old,new)
    (ROOT/report).write_text(text,encoding='utf8')
    (ROOT/'reports/summary.md').write_text('# Ti–Al FM: текущее состояние\n\n7 ноября2026:16 DPA4 моделей завершены. Запущены8 повторов N694 seeds20/30; GPU array '+jobs['N694_training']+' с одной GPU одновременно, затем автоматические QC и сравнение трёх seeds.\n\nПодготовлен GRACE-2L-OMAT-medium-base-fp64: pinned checkpoint, N200/validation149 PBE+r²SCAN, четыре draft full/restricted configs, CPU runtime mask audit. GRACE B0/H0 и GPU optimizer/profile ещё впереди; training GRACE не запущен.\n\n[Запуск и подготовка](15_n694_repeats_grace_preparation.md) · [Завершённое сравнение16 DPA4 моделей](14_multiseed_fulltrain_comparison.md).\n',encoding='utf8')
    print(json.dumps(dict(recorded_new_DPA4_cases=8,GRACE_runtime_mask=qc['status'] if qc else 'pending',report=report)))

if __name__=='__main__':main()
