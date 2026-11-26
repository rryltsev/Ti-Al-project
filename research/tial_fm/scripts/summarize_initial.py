"""Publish audit indices, protocol and explicitly non-executable experiment designs."""
from pathlib import Path
import ast, csv, hashlib, json, shutil

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/initial-audit-20261106-01'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def md(name,s):(ROOT/name).write_text(s.strip()+'\n',encoding='utf8')
s=read(RUN/'summary.json');parents=read(RUN/'parent_audit.json')
# The liquid directories contain initial POSCARs, but most do not contain trajectories.
parents['liquid_independence']='Unresolved: 220 initial POSCAR directories, only two trajectory.xyz files locally. Initial structures alone do not recover MD/relaxation frame ancestry.'
write(RUN/'parent_audit.json',parents)
for name in ['data_manifest.jsonl','splits.json','models.json']:
    shutil.copyfile(RUN/name,ROOT/'manifests'/name)
for name in ['source_hashes.json','summary.json','parent_audit.json']:
    shutil.copyfile(RUN/name,ROOT/'manifests'/name)
shutil.copyfile(RUN/'per_structure_residuals.csv',ROOT/'tables/per_structure_residuals.csv')
metrics=read(RUN/'baseline_recomputed.json')['metrics']
rows=[{'reference':xc,'model':role,'quantity':q,**v} for k,props in metrics.items() for xc,role in [k.split('/')] for q,v in props.items()]
with (ROOT/'tables/metrics.csv').open('w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

# Extract policy constants as syntax data: do not import or execute remote code.
lora=ROOT/'runs/backend-source-20261106/1-lora.py';tree=ast.parse(lora.read_text(encoding='utf8'));policy={}
for node in tree.body:
    if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name) and node.target.id.startswith(('_UNFREEZE','_OVERRIDE_FREEZE','_BLOCKS_PATH')):
        val=node.value.args[0] if isinstance(node.value,ast.Call) else node.value
        v=ast.literal_eval(val);policy[node.target.id]=sorted(v) if isinstance(v,set) else v
write(ROOT/'configs/installed_lora_policy.json',{'source_sha256':hashlib.sha256(lora.read_bytes()).hexdigest(),'scope':'Installed UMT implementation as inspected 2026-11-06; not proven byte-identical to original September training code','constants':policy,'extra_rules':['Freeze all first; replace every exact SO3Linear/SO2Linear with adapters','Unfreeze declared entire modules and per-block modules','Unfreeze listed leaves and any leaf containing bias','Override-freeze adam_type_embedding and adam_freqs','Override-freeze all GatedActivation modules'],'rank':16,'alpha':16,'alpha_over_rank':1.0,'runtime_named_parameter_mask':None,'total_parameters':None,'trainable_parameters':None})
configs=[]
for arch,modes in [('DPA4-Neo',['lora16','full']),('GRACE-2L',['full','restricted'])]:
    for mode in modes:
        for xc in ['PBE','r2SCAN']:
            for n in [200,'full_grouped_train']:
                rid=f'{arch.lower()}-{mode}-{xc.lower()}-n{n}-seed10'
                d={'run_id':rid,'state':'blocked','hypothesis':'Compare adaptation freedom at fixed initialization, data IDs and development budget','control_group':f'{arch}/{xc}/{n}','architecture':arch,'functional':xc,'adaptation':mode,'train_size':n,'training_seed':10,'data_selection_seed':20261106,'model_hash':read(RUN/'models.json')['pretrained']['sha256'] if arch=='DPA4-Neo' else None,'split_hash':None,'target_geometry_ids':None,'new_DFT_jobs':0,'max_initial_lr_candidates':3,'base_lr':1e-4 if arch=='DPA4-Neo' else None,'epochs':50 if arch=='DPA4-Neo' else None,'rank':16 if mode=='lora16' else None,'alpha':16 if mode=='lora16' else None,'trainable_mask':None,'planned_resources':{'partition':'v100','account':'imet-2026-2','qos':'normal','gpus':1,'gpu_type':'a100','cpus':4,'memory_GiB':32,'planning_wall_cap_hours':2},'estimated_cost':{'kind':'provisional_reservation_cap_not_measured_forecast','GPU_hours_cap':2,'CPU_hours_cap':8,'wall_estimate_hours':None},'actual_cost':{'GPU_hours':0,'CPU_hours':0},'launch_ready':False,'blocked_by':['grouped_split_and_parent_provenance','Gate_B0','Gate_H0','allocation_budget','runtime_mask']+(['pinned_GRACE_2L_checkpoint'] if arch=='GRACE-2L' else []),'next_step':'Do not submit. Resolve inputs and gates, then profile one short run and replace provisional caps with measured estimates.'}
                write(ROOT/'configs'/(rid+'.json'),d);configs.append(d)
matrix={'format_note':'JSON syntax is valid YAML 1.2. These are experiment designs, not scheduler submission files.','date':'2026-11-06','new_DFT_allowed':False,'allocation_id':None,'max_concurrent_training_jobs':1,'stages':[{'id':'I0','state':'completed','work':'Map 2990 label records, preserve historical splits and hashes','actual_cost':{'local_wall_seconds':s['elapsed_seconds'],'cluster_GPU_hours':0,'new_DFT_jobs':0}},{'id':'B0-cached','state':'completed','work':'Independent raw E/F/stress arithmetic from cached predictions, 49 geometries','actual_cost':{'new_model_evaluations':0},'gate_B0_closed':False},{'id':'H0-cached','state':'completed','work':'Independent spectral RMSE and DOS integrals from saved arrays','gate_H0_closed':False},{'id':'B0-runtime','state':'blocked','work':'Original FM and FT derivative, symmetry, export checks; train-only energy alignment','expected_cost':{'type':'planning_cap','CPUs':4,'wall_minutes':30,'CPU_hours_cap':2},'blocked_by':['allocation_for_new_cycle','evaluator_inputs']},{'id':'H0-original-DPA4','state':'blocked','work':'Original DPA4 on 3 XC cells x force000/001/002 = 9 force evaluations; reuse FT forces only with exact identities','expected_cost':{'type':'planning_cap','CPUs':4,'wall_minutes':15,'CPU_hours_cap':1,'rationale':'Historical matched FT jobs 34-90 s; original-FM runtime not yet measured.'},'blocked_by':['allocation_for_new_cycle','runtime_validation']},{'id':'H0-GRACE','state':'blocked','work':'GRACE-2L on same 9 force geometries','expected_cost':None,'blocked_by':['pinned_GRACE_2L_checkpoint']}],'training_designs':configs,'screening_summary':{'base_configurations':16,'DPA4_configurations':8,'GRACE_configurations':8,'one_seed_one_LR_GPU_hours_cap':32,'one_seed_one_LR_CPU_hours_cap':128,'with_two_extra_LRs_max_GPU_hours_cap':96,'scope':'Provisional caps, not approved quota or measured forecast; no bulk submission.','cost_update':'Measure peak memory, wall seconds per optimizer step, validation time and target exposures on first short training; project each 50-epoch schedule separately.'},'later_controls':{'random_initialization':'Same selected architecture, PBE/r2SCAN x N200/full; 4 designs after architecture selection','finalist_seeds':[10,20,30],'learning_curves':[100,200,400,'full_grouped_train'],'PBEsol':'after screening','KD':'S0/S1/S2 after qualified teacher; not scheduled'}}
write(ROOT/'protocol/experiment_matrix.yaml',matrix)
md('protocol/benchmark_protocol.md', '''# Ti–Al FM: benchmark protocol v0.1 — 6 ноября 2026

Статус: предварительный протокол; новый обучающий split ещё не выпущен.
Источник постановки: CODEX_TiAl_FM_RESEARCH_PLAN.md. Первый цикл: 0 новых DFT.

- Исторические train/validation сохраняются без изменений для воспроизведения.
- Общий пул содержит 992 геометрии с тремя метками; неполные тройки сохраняются отдельно в manifest.
- Новый split строится по родителям и кластерам почти совпадающих геометрий. Сначала исключить родителей внешних структур, затем stratified group split/folds. Не подменять независимость случайным разделением кадров.
- Пары ± смещений одного harmonic parent остаются вместе. Первоначально harmonic labels используются только для контроля. Исторический PBE15k уже выбирали с учётом фононов.
- N=100/200/400 и полный train вводятся после установления допустимого размера. 850 не является обязательным размером нового train. Во всех методах и XC одинаковые ID и вложенные подвыборки.
- Одна исходная FM для каждого N; отдельно считать уникальные геометрии, geometry–XC labels, атомы, development/replay/harmonic labels и teacher evaluations.
- Энергия: полная энергия eV; метрика meV/atom, structure-weighted. Силы: Cartesian components eV/Å, pooled atom-weighted; дополнительно equal-structure и Ti/Al. Stress: ASE tensile-positive, −virial/volume, шесть независимых компонент, 160.21766208 GPa на eV/Å³.
- Исторический external benchmark: raw energy; VASP TOTEN. Train-aligned энергия — отдельный результат; fit только на train данного N, rank/condition elemental-offset matrix обязательно публиковать.
- Test49 — фиксированный ранее изученный benchmark, с адаптивным выбором PBE checkpoint и остановкой на 49/50. Не использовать его для нового подбора.
- Финальные интервалы: bootstrap по независимым родителям. До полного восстановления родителей не представлять component bootstrap как независимые наблюдения.
- Gate B0 требует fresh runtime checks E/F и E/strain, rotations/permutations/PBC, native/export, precision-dependent tolerances и train alignment. Пересчёт сохранённых residuals закрывает только арифметическую часть.
- Gate H0 требует исходную FM и FT на одинаковых DFT cells, 4×4×3 и ±0.01 Å. Raw/drift/symmetry/ASR результаты раздельны; mode tracking не заменять sorted-frequency RMSE.
- Численные пороги качества и устойчивости определяются до выбора финалиста на development; сейчас они не назначены. FT/KD/дорогая MD закрыты до прохождения gates.
- Один screening seed=10; финалисты 10/20/30; до двух дополнительных LR только по development-диагностике. Весь tuning cost учитывается.
''')
md('reports/00_inventory.md', f'''# Первичная инвентаризация — 6 ноября 2026

Выполнен первый локальный аудит по программе. Этапы 1–3 начаты, но data gate пока не пройден.

## Проверенные данные

| XC | Всего меток | Train | Validation |
|---|---:|---:|---:|
| PBE | 999 | 850 | 149 |
| PBEsol | 998 | 849 | 149 |
| r²SCAN | 993 | 845 | 148 |

Выбранный пул: 1000 геометрий; 880×216, 104×250, 16×256 атомов. Все 2990 NPY-меток сопоставлены с выбранными геометриями при допуске 5×10⁻⁵ Å, с перестановками атомов, периодическими образами и общим переносом. Проверены конечность массивов, число записей и отсутствие exact geometry overlap внутри каждого исторического split. Cell axes сохранены; произвольные замены базиса не проверялись. Все исходные E/F/virial массивы прочитаны и хешированы.

Общий пул: **992 тройки labels**, 218346 атомов геометрий, 655038 атомов–меток. Общих исторических train-ID только **658**, validation-ID — **14**. Эти пересечения не являются новым независимым split.

Для 120 геометрий найден parent registry. Хеши всех 46 твёрдотельных траекторий совпали с архивом. Train/validation используют общие траектории: **11 PBE / 14 PBEsol / 11 r²SCAN**. **24 из 25** твёрдотельных структур внешнего benchmark имеют родителя, представленного в исходном пуле. Геометрическая неповторяемость не устраняет эту зависимость.

Для 880 геометрий жидкостного/релаксационного пула parent/frame пока не восстановлены. Целевой поиск проведён в MACE-database, MD-liquid+relax/RESULT, исходных scripts, reference_origins и trajectory_audit; также просмотрено содержимое двух известных ZIP. В 220 каталогах родителей есть initial POSCARs; локально найдены только две trajectory.xyz. ZIP AlTi_liq_configs содержит 11 начальных структур; select-200 ZIP содержит геометрии, а не полный parent registry. Это ограничение локальной доступности, не доказательство утраты данных на всех носителях.

## Артефакты и пути

- Исходный проект: `{str(Path(r'C:/Users/Roman Ryltsev/Desktop/STARTUPS/Ti-Al-X/TiAl-Project'))}`.
- Геометрии: `MACE-dataset-gen/MACE-database/TiAl-liq-relax-ss_select-1000.xyz`.
- NPY labels: `DPA4-finetune/TiAl_{{PBE,PBEsol,scan}}_upd/npy_format/{{train,validate}}`.
- Raw VASP candidates: `VASP-output/r2SCAN/RESULTS`, `VASP-output/VASP_PBEsol/RESULTS`; их SCF и связь каждого energy field с обучающей меткой требуют отдельного аудита. Полный PBE raw archive пока не установлен.
- Актуальные main/SI: `manuscript/main.tex`, `manuscript/supporting_materials.tex`; main имеет дату 03.11, sections содержат последующие правки KRP. Сохранены фактические исходники; PDF 03.11 и 06.11 не смешиваются.
- Родительская FM: `testset/monitor/models/pretrained/DPA4-Neo-OMat24-v20260805.pt`; SHA-256 fd7f34ae28f921201e4a0328ddf56179892752084f8363e7796038201471c989.
- FT checkpoints/configs/logs: `TiAl-Project/DPA4-checkpoints`, `DPA4-finetune`; frozen exports и метаданные: `testset/monitor/models`.
- Фононы: `phonons/runs/alpha-ti-xc-phonons-20261021-01`, `phonons/results/figure2-six-panels-20261101-02`; raw forces, POSCAR, Phonopy YAML, arrays, processing scripts найдены.
- MD: `TiAl-Project/MD-results`, `MD-scripts`, локальные `audit/oct25`, `audit/oct29`, `audit/nov01`, `audit/nov05`, `audit/nov06`. Это установленные locations; полный MD provenance audit ещё не выполнен.
- GRACE: подтверждено существующее TensorPotential 0.6.1 / TensorFlow 2.20.0 окружение; старый registry содержит GRACE-1L, pinned GRACE-2L пока не установлен. Не заменять 2L на 1L молча.
- DPA4C: исторические synthetic training checks в MLIP-research; Ti–Al training/export/MD qualification ещё нет.

## Snapshot и границы проверки

В `runs/initial-audit-20261106-01/snapshot` сохранены копии main/SI, sections, supporting tables/configs и плана. `source_hashes.json` фиксирует **5025** файлов. Weights/большие данные остаются в исходном хранилище; manifest не считается их резервной копией. Идентификатор geometry основан на выбранном source/frame и проверенном структурном сопоставлении, а не на равенстве файловых хешей.

Источник исторических IDs — reference_origins.json; все привязки NPY повторно проверены по структурам. Единый data_manifest и historical splits доступны в `manifests/`. Near-duplicate audit нового grouped split, полный SCF/DFT-settings audit, гармоническая исходная FM, runtime parameter counts и budget нового цикла остаются открытыми.

Изменений численных результатов рукописи, нового DFT и нового обучения: **0**.
''')
table='\n'.join(f"| {xc} | {metrics[xc+'/ft']['energy_meV_atom']['rmse']:.9f} | {metrics[xc+'/ft']['force_eV_A']['rmse']:.9f} | {metrics[xc+'/ft']['stress_GPa']['rmse']:.9f} |" for xc in ['PBE','PBEsol','r2SCAN'])
md('reports/01_baseline_reproduction.md',f'''# Baseline: воспроизведение сохранённых предсказаний

Независимо пересчитаны raw energy, Cartesian force и stress residuals для 49 общих структур, трёх FT и одной неизменённой FM. SHA-256 модели, POSCAR, порядок элементов и SCF-флаг каждой используемой метки проверены. MAE/RMSE/bias совпадают с сохранённой неокруглённой таблицей (rtol 1e-9, atol 1e-10). Хвосты и residuals сохранены в tables.

| XC | Energy RMSE, meV/atom | Force RMSE, eV/Å | Stress RMSE, GPa |
|---|---:|---:|---:|
{table}

PBE здесь **20k**, в harmonic/physical branch **15k**; PBEsol=23100, r²SCAN=23200. Это разные baseline identities. Старые PBE5k/15k не перезаписывались.

Причина 50→49: r²SCAN для TiAl-test-025 отменён 04.11 до начала выполнения. PBE/PBEsol labels для этого ID существуют; они не удаляются. Для тройного сравнения используется общий набор49; для одиночных анализов неполная тройка сохраняется отдельно. Источник: `testset/monitor/runs/early-stop-20261104/closure.json`.

Это повторная арифметическая оценка cached predictions, **0 новых model evaluations**. Новые native/export, finite differences, equivariance и train-only alignment не выполнялись; полный Gate B0 открыт. Исторические интервалы не пересчитывались как independent-parent bootstrap. Полученное совпадение не доказывает отсутствие parent leakage.
''')
ph=read(RUN/'phonon_recomputed.json')['results']
md('reports/02_phonon_baseline.md','''# Harmonic baseline: начальный аудит

По сохранённым массивам bands/DOS независимо воспроизведены sorted-frequency RMSE:

'''+ '\n'.join(f"- {xc}: {v['sorted_band_RMSE_THz']:.12f} THz." for xc,v in ph.items())+'''

Шесть DOS-интегралов равны 3 modes/atom в допуске 1e-8. Для r²SCAN: DFT singlet 5.189323502 THz, FT singlet 4.220688046 THz; отношение квадратов около 0.662. У FT singlet меняет порядок относительно doublet, поэтому сравнивать только последние отсортированные частоты неправильно.

Сохранены найденные ссылки на DFT cells, force000/001/002, ±0.01 Å, 4×4×3, Phonopy YAML и обработку. Это воспроизведение спектральной арифметики, не новый расчёт force constants или проверка mode overlaps.

Гармонические силы исходных DPA4-Neo и GRACE-2L в новом цикле не рассчитаны. Gate H0 открыт; текущий mismatch не называется forgetting. Различия DFT training/phonon protocols и отсутствие доказанной cutoff/kmesh/supercell convergence сохраняются.
''')
md('protocol/decision_log.md','''# Решения — 6 ноября 2026

1. Новый цикл ведётся в research/tial_fm текущего рабочего проекта; Desktop/source и рукопись только читаются.
2. Исторический split сохранён; общий реестр строится по структурным совпадениям. Новый train не выпускается до восстановления родителей и near-duplicate audit.
3. 992 общие тройки не означают 850 независимых обучающих состояний. Исторический common-train658 не объявляется исправленным benchmark.
4. Test49 фиксируется как ранее изученный внешний benchmark с parent overlap; TiAl-test-025 не считается модельным отказом.
5. PBE20k для accuracy и PBE15k для phonons — отдельные модели. Никакой автоматической смены baseline.
6. Копирование полного remote training.py/LoRA сначала отклонено автоматической проверкой как потенциально конфиденциальный перенос. Пользователь явно разрешил копирование; повторная операция прошла, хеши локальных копий совпали. Ограничение снято.
7. В текущей реализации LoRA обучаются также целые группы модулей, biases и scales. Нельзя описывать её как обучение только низкоранговых матриц. Byte identity с оригинальным кодом сентябрьского обучения пока не доказана; runtime named-parameter mask ещё нужен.
8. Установленный DeepMD 3.2.0 поддерживает LoRA/HybridMuon; GRACE окружение существует. Наличие пакета не подтверждает готовность GRACE-2L checkpoint или scientific DPA4C pipeline.
9. Кластерная preflight: существующая Ti-visc-r2SCAN-refine20988268 работает на A100, другие задачи не менялись; свободно около20.98 GiB home6 по quota. Старые Al32 allocation limits не переносятся на новый Ti–Al цикл. Известного численного бюджета нового цикла не найдено.
10. До закрытия data/B0/H0 и измеренного profiling не отправлять весь пилот. Matrix содержит проектные caps, а не обещание времени/разрешённую квоту.
11. Корневой Git не имеет коммитов, remote и настроенного имени/email. Авторская identity не выдумана; отдельный каталог и SHA-256 snapshot обеспечивают сохранение начальной работы. Коммит пока не создан.
''')
md('reports/summary.md',f'''# Начало проекта Ti–Al FM — 6 ноября 2026

План прочитан полностью. Выполнен первый аудит: **2990 labels, 1000 геометрий, 992 общие тройки, 5025 хешей источников**; сохранены исторические splits и snapshot. Новых DFT, обучения и model evaluations — **0**.

Воспроизведены raw E/F/stress статьи и спектральные RMSE по сохранённым массивам. Это закрывает арифметическую часть baseline, но не полные B0/H0.

Основной найденный риск: **11/14/11** родительских траекторий пересекают train/validation PBE/PBEsol/r²SCAN; **24/25** внешних solid states имеют общего родителя с исходным пулом. У **880** геометрий ещё не восстановлены parent/frame. Новый независимый split пока не выпущен.

Зафиксирован источник 50→49: отмена ещё не начатого r²SCAN для TiAl-test-025. Политика одинакова для всех моделей; существующие другие labels сохраняются.

Подготовлены [протокол](../protocol/benchmark_protocol.md), [матрица](../protocol/experiment_matrix.yaml), 16 отдельных design configs и [подробная инвентаризация](00_inventory.md). Первые независимые продолжения: восстановление liquid ancestry/near-duplicate groups; fresh baseline derivative/export checks; 9 force evaluations исходного DPA4 на трёх DFT cells. GRACE-2L checkpoint пока не pinned.

Ресурсы: новые baseline checks проектируются с caps 2 CPUh, исходные DPA4 phonons — 1 CPUh. Пилот16 имеет предварительный cap32 GPUh/128 CPUh на один LR и seed; это резервный предел проектирования, **не измеренная оценка и не выделенная квота**. До первого profiling время обучения неизвестно; после него пересчитать очередь. Численный бюджет нового Ti–Al цикла пока не установлен, Al32 бюджет не используется.

Прогресс и следующий шаг записаны в [state.json](../state.json). Архивы исходников LoRA сохранены после явного разрешения пользователя; блокировка переноса снята.
''')
md('README.md','''# Ti–Al foundation-model research

Entry points: reports/summary.md, state.json, protocol/benchmark_protocol.md.
The scientific programme is preserved in protocol/CODEX_TiAl_FM_RESEARCH_PLAN.md.

Reproduce the initial read-only audit with the existing project Python:

```powershell
& './testset/.venv/Scripts/python.exe' research/tial_fm/scripts/audit_initial.py --run-id UNIQUE_NEW_RUN_ID
```

Run from the TiAl workspace. A fresh run ID is mandatory; the script refuses to overwrite runs.
Expected Python dependencies: NumPy and ASE. Source paths and their SHA-256 values are recorded.
The unified dataset is a manifest view over original labels; no DFT/teacher labels are mixed or overwritten.
Config files are experiment designs with launch_ready=false, not submit-ready jobs.
''')
state=read(ROOT/'state.json');state.update({'updated_local_date':'2026-11-06','stage':'initial_inventory_and_cached_reproduction','new_DFT_jobs':0,'new_training_jobs':0,'external_shared_parent_structures':24,'installed_lora_policy':'configs/installed_lora_policy.json','backend_source_permission':'explicit_user_approval_received_and_transfer_verified','next_actions':['Recover missing liquid trajectory/frame provenance from registered source locations and archives','Construct grouped split with external-parent exclusion and near-duplicate audit','Pin runtime parameter masks and historical-code identity','Resolve TiAl allocation; profile one CPU baseline and original DPA4 harmonic run','Run pilot only after data/B0/H0 gates'],'training_launch_ready':False,'git':{'commit':None,'reason':'unborn repository; author identity not configured; no remote'},'reports':['reports/00_inventory.md','reports/01_baseline_reproduction.md','reports/02_phonon_baseline.md','reports/summary.md']});write(ROOT/'state.json',state)
print('Published protocol, 16 blocked design configs, manifests, metrics and reports.')
