from pathlib import Path
import datetime,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/dpa4-n200-pilot-20261106-02'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def main():
    if not (RUN/'state-before-pilot.json').exists():shutil.copyfile(ROOT/'state.json',RUN/'state-before-pilot.json')
    state=read(ROOT/'state.json');sub=read(ROOT/'datasets/nested-train-v1-20261106/subsets.json')
    state.update(status='N200_ready_baseline_in_progress',stage='original_DPA4_baseline_and_phonons',current_run=str(RUN),N200=sub['N200_counts'],updated_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    state['resource_authorization']=dict(account='imet-2026-2',source='phonons/config/protocol.json',explicit_current_user_request=True,cumulative_agent_imposed_limit=None,concurrent_new_GPU_jobs=1,respect_scheduler_and_storage_limits=True)
    state['jobs']={k:read(RUN/(k+'-submission.json')) for k in ['baseline','harmonic','training'] if (RUN/(k+'-submission.json')).exists()}
    state['current_gates']={k:('PASS' if (RUN/'collected'/folder/'summary.json').exists() and read(RUN/'collected'/folder/'summary.json')['status']=='PASS' else 'in_progress') for k,folder in [('B0','baseline'),('H0','harmonics')]}
    state['new_training_jobs']=4 if 'training' in state['jobs'] else 0
    state['training_launch_ready']=all(x=='PASS' for x in state['current_gates'].values())
    if 'training' in state['jobs']:state.update(status='DPA4_N200_pilot_submitted',stage='DPA4_N200_training')
    state['next_actions']=['Finish original-model B0/H0 and unified metrics','Run PBE/r2SCAN N200 LoRA16/full from the identical original checkpoint','Verify zero-step, actual masks and frozen weights; measure runtime/memory','Evaluate fixed-final pilots on development and matched phonons; test not used for pilot tuning']
    dump(ROOT/'state.json',state)
    matrix=read(ROOT/'protocol/experiment_matrix.yaml');matrix['allocation_id']='imet-2026-2';matrix['resource_authorization']=state['resource_authorization']
    for stage in matrix['stages']:
        if stage['id'] in ['B0-runtime','H0-original-DPA4']:
            gate='B0' if stage['id']=='B0-runtime' else 'H0';stage['state']='completed' if state['current_gates'][gate]=='PASS' else 'running';stage['run']=str(RUN)
    for design in matrix['training_designs']:
        if design['train_size']==200:design['target_geometry_ids']=sub['subsets']['200']
        design['blocked_by']=[x for x in design.get('blocked_by',[]) if x not in ['allocation_budget','nested_N200_train_selection']]
        if design['architecture']=='DPA4-Neo' and design['train_size']==200:
            design['state']='submitted' if 'training' in state['jobs'] else 'prepared_waiting_B0_H0'
            design['launch_ready']=state['training_launch_ready'];design['runtime_mask_policy']='Measured at initialization before any optimizer step; zero-step and frozen-weight checks are mandatory inside each job.'
            design['planned_resources'].update(memory_GiB=40,planning_wall_cap_hours=4)
            design['estimated_cost']=dict(kind='scheduler_reservation_ceiling_not_runtime_forecast',GPU_hours_cap=4,wall_estimate_hours=None)
            design['pilot_package']=str(RUN/'training-plan01')
            if 'training' in state['jobs']:design['Slurm_array_job_id']=state['jobs']['training']['job_id']
    matrix['screening_summary']['scope']='Wider design matrix is not bulk-submitted. Current request executes four DPA4 N200 cases; schedule ceilings are not measured cost or cumulative agent budget.'
    dump(ROOT/'protocol/experiment_matrix.yaml',matrix)
    report=f'''# N=200 и пилот DPA4

Обновлено {state['updated_UTC']}.

Подвыборка N=200 создана только из train694 нового split. В ней 44 034 атома, 83 консервативные группы состава: 176 liquid, 10 gamma, 11 bcc, 3 hcp. Все исходные метки PBE/PBEsol/r²SCAN скопированы без изменения. Подвыборки N100 ⊂ N200 ⊂ N400 ⊂ N694 вложены. Отбор использует семейство, состав и объём, без энергий, сил, предсказаний или test.

На каждый функционал пилот использует 200 train + 149 development = 349 целевых меток. Test149 остаётся отдельным; его можно оценить фиксированной исходной моделью, но он не подаётся обучению и выбору настроек пилота. Старый test49 — архивная диагностика.

Для PBE и r²SCAN подготовлены LoRA rank16/alpha16 и полный FT: одинаковый parent checkpoint `fd7f34ae28f921201e4a0328ddf56179892752084f8363e7796038201471c989`, seed10, 50 эпох, cosine LR 1e-4→1e-6, HybridMuon, MAE E/F/virial с prefactors20/20/5. Single-GPU запуск использует zero_stage=0. Статистики считаются заново только на N200. Checkpoint для оценки заранее фиксирован: последний после 50 эпох; подбор по test исключён.

Runtime observer проверяет нулевые B-адаптеры, восстановление исходных сил/вириала до первого шага, допустимый композиционный сдвиг энергии, live→merged→export, реальные requires_grad и неизменность замороженных параметров. Он записывает фактические предъявления структур/атомов, шаги, время и память GPU. Слово LoRA не означает, что обучаются только A/B: установленная реализация также размораживает отдельные исходные слои.

Текущие gates: B0={state['current_gates']['B0']}; H0={state['current_gates']['H0']}. Scheduler-записи находятся в каталоге текущего запуска. Новых DFT:0. Новых training tasks: {state['new_training_jobs']} (при наличии submission это число поставленных в очередь задач, не завершённых обучений).

Установленное разрешение пользователя из `phonons/config/protocol.json` снимает искусственные суммарные лимиты агента; действуют фактические ограничения Slurm, счёта imet-2026-2 и дисковой квоты. Одновременно допускается один новый GPU-пилот. Четыре часа на задачу — верхняя граница выделения, а не прогноз стоимости. Прогноз обновляется по первым 100 шагам. Существующие расчёты не изменяются.

История инженерных исправлений: попытка20988437 завершилась за69с на сериализации numpy.bool_ в JSON, до обучения. Повтор20988440 сохраняет этот сбой отдельно. Для Phonopy подтверждён venv с `bin/python`; добавлены версионные launcher-файлы, исходный пакет сохранён.

Гарантия разбиения по-прежнему условна на сохранение nTi/nAl в родительской MD и relaxation; точные liquid trajectory IDs не восстановлены. Группировка также создаёт перенос между точными составами. Фононы при использовании для настройки являются физическим development benchmark; численная сходимость DFT по cutoff/kmesh/supercell не доказана.
'''
    (ROOT/'reports/08_n200_pilot.md').write_text(report,encoding='utf8')
    (ROOT/'reports/summary.md').write_text(f'''# Ti–Al FM: текущее состояние

6 ноября 2026: новое разбиение 694 train / 149 validation / 149 test проверено. Подвыборка N=200 готова: 44 034 атома, 83 группы; по 349 train+development меток на функционал.

Текущий этап: {state['stage']}. B0={state['current_gates']['B0']}; H0={state['current_gates']['H0']}. Пилот DPA4: LoRA16 и full FT для PBE/r²SCAN из одного исходного checkpoint. Новых DFT:0; поставленных training tasks:{state['new_training_jobs']}.

[Пилот и протокол](08_n200_pilot.md) · [Разбиение и ограничения provenance](07_grouped_split.md).

Прежний test49 остаётся архивной диагностикой. Старые FT checkpoints видели часть нового test и не являются независимыми baseline на новом split.
''',encoding='utf8')
    print(state['status'])
if __name__=='__main__':main()
