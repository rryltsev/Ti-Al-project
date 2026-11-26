from pathlib import Path
import hashlib,json,re
import numpy as np
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/dpa4-n200-pilot-20261106-02'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def main():
    metrics=read(RUN/'baseline-analysis/metrics.json');base=read(RUN/'collected/baseline/summary.json');assert base['status']=='PASS'
    projection=read(RUN/'collected/harmonics/projection_diagnostics.json');assert projection['status']=='PASS'
    progress=read(Path(read(RUN/'latest-progress.json')['snapshot']));p=progress['cases']['PBE-lora16'];profile=p['progress.json'];checks=p['checkpoint_checks'];assert checks['01000.json']['status']=='PASS'
    assert p['numerical-check-step01000/summary.json']['status']=='PASS'
    # Independent and bulk inference evaluated the same nine input geometries.
    harmonic_repeat={}
    for xc in ['PBE','PBEsol','r2SCAN']:
        harmonic_repeat[xc]={key:float(np.max(np.abs(np.load(RUN/'collected/baseline/harmonic_forces'/xc/key/'forces.npy')-np.load(RUN/'collected/harmonic-early/baseline/harmonic_forces'/xc/key/'forces.npy')))) for key in ['force000','force001','force002']}
    assert max(v for d in harmonic_repeat.values() for v in d.values())<2e-4
    dump(RUN/'harmonic-repeat-check.json',dict(status='PASS',force_max_difference_eV_A=harmonic_repeat))
    rows=progress['completed_CPU_accounting'].splitlines()[1:];allocated=0
    for line in rows:
        a=line.split('|')
        if len(a)>5 and '.' not in a[0]:allocated+=int(a[3])*int(a[4])/3600
    text='''# Baseline исходной DPA4 на новом разбиении

Полностью рассчитаны 200 train,149 validation и149 test геометрий для каждого из PBE/PBEsol/r²SCAN:1494 native model evaluations. В каждом функционале использована его точная сохранённая DFT-геометрия. Исходная модель, hashes, evaluator и все предсказания сохранены в `runs/dpa4-n200-pilot-20261106-02`.

## Test149: фиксированная исходная модель

| XC | E raw RMSE, meV/atom | E train-aligned RMSE, meV/atom | F component RMSE, eV/Å | Stress component RMSE, GPa |
|---|---:|---:|---:|---:|
'''
    for xc,x in metrics.items():
        m=x['metrics']['test'];text+='| '+xc+' | '+' | '.join(f"{m[k]['RMSE']:.6f}" for k in ['E_raw_meV_atom','E_aligned_meV_atom','F_component_eV_A','stress_component_GPa'])+' |\n'
    text+='''
Энергия усредняется по структурам после деления на число атомов. Forces объединены по атомам и декартовым компонентам; отдельно сохранены ошибки норм векторов и равновесовое по структурам F-RMSE. Stress рассчитан как −sym(virial)/volume, в ГПа, по шести независимым компонентам. Статистика включает MAE, RMSE, bias,95/99%-квантили и максимумы, семейства/составы/элементы,25 групп test и23 группы validation. Неудачных предсказаний:0; ни одна структура не исключена из-за ошибки модели.

Энергетические сдвиги E_aligned=E_model+nAl*bAl+nTi*bTi оценены только на N200 train равновесовым по структурам least squares. Они не меняют силы и напряжения. Validation/test не используются для сдвигов.

| XC | bAl, eV | bTi, eV | Rank | Condition number |
|---|---:|---:|---:|---:|
'''
    for xc,x in metrics.items():
        a=x['alignment'];text+=f"| {xc} | {a['offset_eV'][0]:.8f} | {a['offset_eV'][1]:.8f} | {a['rank']} | {a['condition_number']:.4f} |\n"
    text+='''
Сырые и выровненные ошибки существенно различаются для r²SCAN: композиционный сдвиг объясняет большую часть исходного энергетического residual, но после него остаётся конфигурационная ошибка. Сравнение с числами старого test49 некорректно из-за другого набора и истории обучения.

Bootstrap2000 повторов ресэмплирует целые консервативные fixed-composition группы. Это не точный trajectory bootstrap: liquid parent IDs утрачены. Интервалы сохранены в metrics.json; десятки тысяч силовых компонент не трактуются как независимые траектории.

## Численная проверка

Перестановки, PBC-переносы, вращения, E→F и E→virial при трёх шагах конечной разности проверены на pureAl, pureTi и сплаве. Максимальное различие сил при симметрийных преобразованиях4.20e-6эВ/Å. Для finite differences уменьшение шага ниже0.003Å усиливает float32 cancellation; полный ряд шагов сохранён, а не заменён требованием машинного нуля.

Свежий CPU-export `.pt2` согласуется с native checkpoint: E≤1.19e-7эВ/атом, F≤2.51e-6эВ/Å, virial≤1.78e-4эВ на проверенных геометриях. Независимый аналитический тест evaluator восстановил заранее заданные Al/Ti energy offsets, F-component/vector distinction и знак/единицы stress. Архивная арифметика старых FT сохранена в предыдущем аудите.

Bulk inference был продолжен после штатного Slurm TIMEOUT: завершённые файлы не пересчитаны и проверены по ID/shape/finite/hash; второе задание вычислило только119 недостающих кадров. B0 завершён со статусом PASS. H0 завершён отдельным job20988449; повторная native оценка тех же9 геометрий согласуется с ним в пределах установленного допуска.

## Пилот

'''
    text+=f"На момент {progress['utc']} PBE LoRA находится на шаге{profile['steps']}/10000. Checkpoint1000 прошёл проверку live→merged и неизменности frozen weights. Peak CUDA allocated={profile['peak_cuda_allocated_bytes']/2**30:.2f}GiB, reserved={profile['peak_cuda_reserved_bytes']/2**30:.2f}GiB. Наблюдаемая средняя стоимость≈{profile['training_seconds']/profile['steps']:.3f}с/шаг, включая текущую периодическую validation; порядок40мин на10000шагов, затем отдельный export/evaluation. Это оценка данного LoRA-прогона, не измерение fullFT.\n\n"
    text+='''GPU array20988454:0=PBE-LoRA16,1=PBE-full,2=r2SCAN-LoRA16,3=r2SCAN-full; одновременно один новый GPU. Остальные три задачи стоят в очереди. Все четыре реальные CPU-инициализации уже пройдены без optimizer steps. Runtime mask: full1124686/1124686 параметров; LoRA325158/1410382, включая285696 adapter-параметров и39462 исходных параметров. GPU observer дополнительно требует точного совпадения маски и hashes установленного backend.

Каждый pilot начинает с одного исходного pretrained checkpoint, заново считает N200-only statistics и использует50эпох, seed10 и исходный HybridMuon/MAE20/20/5 recipe. После обучения автоматически проверяется export, оцениваются train/development и matched phonons. CPU array20988460 вычисляет единые метрики и group-bootstrap после соответствующей успешной GPU-задачи. Test149 не передаётся этим обучающим/evaluation задачам пилота.

'''
    text+='Checkpoint 1000 также прошёл повторные проверки перестановок, вращений, PBC, E→F и E→virial при нескольких шагах конечной разности (job 20988473). Такие же проверки после каждого финального пилота назначены CPU-массиву 20988469.\n\n'
    text+=f"Завершённые CPU-проверки заняли {allocated:.3f} allocated CPU-hours (включая техническую ошибку первой попытки, четыре zero-step инициализации и численные проверки checkpoint 1000); фактический TotalCPU также сохранён. GPU-обучения ещё не завершены; итоговые веса, их качество и стоимость не заявляются. Новых DFT:0.\n\n"
    text+='[N200 и протокол](08_n200_pilot.md) · [Raw VASP audit](09_energy_field_audit.md) · [Исходные фононы](10_original_harmonics.md).\n'
    (ROOT/'reports/11_baseline_pilot_status.md').write_text(text,encoding='utf8')
    state=read(ROOT/'state.json');state.update(status='B0_H0_passed_PBE_LoRA_running',stage='DPA4_N200_pilot_training',training_launch_ready=True,new_training_jobs=4,completed_training_jobs=0,new_DFT_jobs=0,current_gates={'B0':'PASS','H0':'PASS'},runtime_masks={'full_trainable':1124686,'lora_trainable':325158,'lora_adapters':285696},first_pilot_progress=profile,first_verified_training_checkpoint=1000,completed_CPU_allocated_hours=allocated,latest_progress_snapshot=read(RUN/'latest-progress.json'),label_field_audit=str(ROOT/'runs/label-fields-20261106-02'),baseline_metrics=str(RUN/'baseline-analysis/metrics.json'))
    state['jobs']={p.stem:read(p) for p in RUN.glob('*-submission.json')}
    state['trained_checkpoint_numerical_QC']={'checkpoint':1000,'status':'PASS','job_id':'20988473','evidence_snapshot':read(RUN/'latest-progress.json')['snapshot']}
    state['reports']=[str(p.relative_to(ROOT)) for p in sorted((ROOT/'reports').glob('*.md'))]
    dump(ROOT/'state.json',state)
    matrix=read(ROOT/'protocol/experiment_matrix.yaml')
    for d in matrix['training_designs']:
        if d['architecture']=='DPA4-Neo' and d['train_size']==200:
            d['blocked_by']=[];d['launch_ready']=True;d['state']='running' if d['functional']=='PBE' and d['adaptation']=='lora16' else 'queued'
            d['trainable_parameters']=1124686 if d['adaptation']=='full' else 325158
            d['actual_cost_note']='See current progress snapshot; only PBE-LoRA has executed optimizer steps at this snapshot.'
    dump(ROOT/'protocol/experiment_matrix.yaml',matrix)
    (ROOT/'reports/summary.md').write_text(f'''# Ti–Al FM: текущее состояние

N=200 подготовлена на новом разбиении694/149/149. B0 и H0 выполнены. Исходная DPA4 имеет мнимые ветви в alpha-Ti на фиксированных DFT-ячейках; они присутствуют до ASR.

Пилот начат: PBE-LoRA — шаг{profile['steps']}/10000, checkpoint1000 проверен. PBE-full, r²SCAN-LoRA и r²SCAN-full поставлены в последовательную GPU-очередь20988454. Новых DFT:0. Финальные результаты обучения ещё не получены.

[Основной отчёт и метрики](11_baseline_pilot_status.md) · [N200](08_n200_pilot.md) · [VASP audit](09_energy_field_audit.md) · [Фононы и график](10_original_harmonics.md) · [Разбиение](07_grouped_split.md).
''',encoding='utf8')
    replacements={'равновесовое по структурам F-RMSE':'F-RMSE с одинаковым весом каждой структуры','равновесовым по структурам least squares':'методом наименьших квадратов с одинаковым весом каждой структуры','Forces объединены':'Ошибки сил объединены','Stress рассчитан':'Тензор напряжений рассчитан','энергетического residual':'энергетического отклонения','Bootstrap2000':'Bootstrap: 2000','pureAl, pureTi':'чистых Al, Ti','float32 cancellation':'потерю значащих разрядов при вычитании энергий в float32','job20988449':'заданием 20988449','Checkpoint1000':'Checkpoint 1000','Peak CUDA allocated=':'Пиковая память CUDA: allocated=','порядок40мин на10000шагов':'около 40 мин на 10 000 шагов','GPU array20988454':'GPU-массив 20988454','CPU array20988460':'CPU-массив 20988460','full1124686/1124686':'full FT — 1 124 686 из 1 124 686','LoRA325158/1410382':'LoRA — 325 158 из 1 410 382','включая285696 adapter-параметров и39462 исходных параметров':'включая 285 696 параметров адаптеров и 39 462 исходных параметра','seed10':'seed 10','fullFT':'full FT','Bootstrap2000 повторов':'Bootstrap из 2000 повторов'}
    for name in ['08_n200_pilot.md','09_energy_field_audit.md','10_original_harmonics.md','11_baseline_pilot_status.md','summary.md']:
        path=ROOT/'reports'/name;t=path.read_text(encoding='utf8')
        for old,new in replacements.items():t=t.replace(old,new)
        t=re.sub(r'([А-Яа-яЁё])(\d)',r'\1 \2',t);t=re.sub(r'(\d)([А-Яа-яЁё])',r'\1 \2',t)
        t=re.sub(r'([,:])(\d)',r'\1 \2',t);path.write_text(t,encoding='utf8')
    dump(RUN/'local_artifact_hashes.json',{p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for directory in [ROOT/'scripts',ROOT/'reports',RUN/'baseline-analysis'] for p in directory.rglob('*') if p.is_file() and '__pycache__' not in str(p)})
    print('REPORT_READY',profile['steps'],allocated)
if __name__=='__main__':main()
