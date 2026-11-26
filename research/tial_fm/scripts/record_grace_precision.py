"""Document precision evidence without prematurely choosing GRACE training precision."""
from pathlib import Path
import datetime,json
from prepare_grace_precision import ROOT,RUN
def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')

def main():
    jobs=read(RUN/'submission-summary.json');status=read(RUN/'latest-status.json')
    path=RUN/'collected/comparison/cpu/summary.json';cpu=read(path) if path.exists() else None
    state=read(ROOT/'state.json');matrix=read(ROOT/'protocol/experiment_matrix.yaml')
    for p,name in [(ROOT/'state.json','state-before-precision.json'),(ROOT/'protocol/experiment_matrix.yaml','matrix-before-precision.json')]:
        if not (RUN/name).exists():(RUN/name).write_bytes(p.read_bytes())
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
    state.update(updated_utc=utc,stage='DPA4_N694_active_GRACE_precision_benchmark',status='GRACE_precision_provisional_CPU_'+('complete' if cpu else 'running')+'_GPU_queued')
    state['grace_precision_benchmark']=dict(run=str(RUN),jobs=jobs,CPU='PASS' if cpu else 'running',GPU='queued_after_N694',model_family='GRACE-2L-OMAT-medium-base',selected_precision=None,inference_preference='fp32_pending_GPU_measurements',diagnostic_preference='fp64_for_tight_finite_differences',training_precision='undecided',source_kind='controlled_fp32_cast_of_pinned_fp64',training_submitted=False)
    state['current_gates']['GRACE_precision']='CPU_complete_GPU_pending' if cpu else 'CPU_running_GPU_pending'
    state['grace_preparation']['precision_selection']='provisional; benchmark required'
    state['next_actions']=['Complete A100 fp32/fp64 inference benchmark after N694 array; then choose GRACE training/deployment precision.']+[x for x in state['next_actions'] if 'Complete A100' not in x]
    report='reports/16_grace_precision_comparison.md'
    if report not in state['reports']:state['reports'].append(report)
    for d in matrix['training_designs']:
        if d['architecture']=='GRACE-2L':
            d['precision_selection']='provisional';d['precision_benchmark']=str(RUN)
            if 'GRACE_precision_benchmark' not in d['blocked_by']:d['blocked_by'].append('GRACE_precision_benchmark')
    write(ROOT/'state.json',state);write(ROOT/'protocol/experiment_matrix.yaml',matrix)
    lines=['# GRACE: сравнение fp32 и fp64','',f'Состояние на {utc} UTC. GRACE training ещё не запущен; точность остаётся предварительным выбором.','',
        '## Что сравнивается','',
        'FP32 хранит число в 32 битах (около 7 значащих цифр), FP64 — в 64 битах (около 16). В данном GRACE fp32 означает параметры и часть вычислений нейросети в float32 при геометрии float64; FP64 использует двойную точность параметров. Численная точность влияет на округление и разности близких энергий/сил. Ошибка относительно DFT также зависит от самой модели и обучающих данных.',
        '', 'Официальная документация даёт ориентир около 2× ускорения и снижения памяти для fp32. Этот ориентир не заменяет измерения на выбранном backend, устройстве и размерах ячеек. [Документация GRACE](https://gracemaker.readthedocs.io/en/latest/gracemaker/foundation/).',
        '', 'Контролируемый эксперимент использует GRACE-2L-OMAT-medium-base: исходный fp64 checkpoint revision `4aff666029967745f7993f348eb87a804d625cff`, архив SHA-256 `763b7e5679c81ea82d39e595e701d0c9f6565064393f705cbe76663c9583d0aa`. FP32 checkpoint получен преобразованием тензоров этого источника в типы native fp32-модели. Все 44 model tensors проверены на точное равенство после cast. Обучение не выполнялось.',
        '', 'Официальный fp32-архив той же revision не загружен полностью из-за медленной передачи (~50 КБ/с); partial-файл сохранён и не используется. Преобразованный checkpoint явно отличается по происхождению от скачанного официального fp32-артефакта.',
        '', 'Первая попытка20989425 остановилась до инференса: имена путей aliased TensorFlow objects в старом checkpoint отличаются от сериализации текущего backend. Исправление восстанавливает source через object graph, нормализует его текущим backend и затем выполняет cast. Исходные веса не изменены; неудачная попытка сохранена в grace-precision-20261107-01. Зависимые старые GPU-задачи20989427/20989428 отменены, DPA4 jobs не изменялись.',
        '', '## Протокол','',
        '- TensorPotential 0.6.1 / TensorFlow 2.20.0, native TPModel + ASE TPCalculator, diverse/segment engine, UQ и TF32 выключены.',
        '- CPU: 4 потока, main compute JIT выключен; A100: main compute XLA включён. Сравнение точностей выполняется внутри одного устройства/режима, а не между разными backend.',
        '- Четыре свежих процесса в порядке fp32/fp64/fp64/fp32. Холодный вызов и прогрев учитываются отдельно; ASE result cache принудительно обходится. Результаты синхронизируются переводом выходов TensorFlow в numpy.',
        '- Ti96 и TiAl216/432; A100 дополнительно TiAl864. После прогрева — по 6 CPU или 30 GPU измерений на размер в каждом из двух процессов.',
        '- 8 фиксированных N200-train геометрий × 2 XC, с исходными XC-specific координатами: pure Al/Ti/alloy probes и все четыре семейства. Это численный диагностический panel, не полный validation benchmark.',
        '- Производные: силы при h=1e−2…1e−5 Å; гидростатический stress при strain=1e−3…1e−5. Фононы PBE/r²SCAN на одних DFT cells, ±0.01 Å, тот же central/ASR protocol и mesh50³.',
        '- Это TensorFlow/ASE latency, а не измерение LAMMPS/Kokkos MD throughput. Финальная стоимость MD требует соответствующего deployment backend.',
        '',f'CPU benchmark **{jobs["CPU_benchmark"]}**, анализ **{jobs["CPU_analysis"]}**. A100 benchmark **{jobs["GPU_benchmark"]}**, анализ **{jobs["GPU_analysis"]}**; GPU ждёт окончания N694 array20989407.']
    if cpu:
        assert cpu['status']=='PASS'
        lines += ['', '## Измерения CPU','', '| Структура | fp32, ms | fp64, ms | Ускорение fp32 |','|---|---:|---:|---:|']
        for name,r in cpu['timing'].items():lines.append(f'| {name} | {r["fp32"]["median_ms"]:.2f} | {r["fp64"]["median_ms"]:.2f} | {r["fp32_speedup"]:.2f}× |')
        lines += ['', 'Intel Xeon Gold 6254 (apollo30), выделены 4 CPU-потока. Медиана полного вызова E/F/stress, включая подготовку соседей. Распределения и отдельные process medians сохранены в summary.json; два процесса оценивают технический разброс, а не вариацию физических структур. fp64 показал заметный разброс между процессами; объединённые коэффициенты ускорения следует считать ориентировочными.',
            '', '| Разность fp32−fp64 | RMSE | Максимум |','|---|---:|---:|']
        for name,r in cpu['precision_delta'].items():lines.append(f'| {name} | {r["RMSE"]:.6g} | {r["max_abs"]:.6g} |')
        lines += ['', '| XC | Phonon RMSE fp32−fp64, THz | Максимум, THz | fp32/DFT RMSE | fp64/DFT RMSE |','|---|---:|---:|---:|---:|']
        for xc,r in cpu['harmonics'].items():
            z=r['fp32_minus_fp64'];lines.append(f'| {xc} | {z["band_RMSE_THz"]:.6g} | {z["band_max_abs_THz"]:.6g} | {r["fp32"]["sorted_band_RMSE_to_DFT_THz"]:.6g} | {r["fp64"]["sorted_band_RMSE_to_DFT_THz"]:.6g} |')
        lines += ['', '| Precision | Память численных параметров, MiB | Peak process RSS по двум запускам, MiB |','|---|---:|---|']
        for p,r in cpu['memory'].items():lines.append(f'| {p} | {r["parameter_bytes"]/1024**2:.2f} | '+', '.join(f'{x:.1f}' for x in r['peak_process_RSS_MiB'])+' |')
        lines += ['', 'Пиковая память процесса включает TensorFlow, параметры, промежуточные тензоры и кэши графов; она не обязана уменьшаться ровно вдвое.',
            '', '| h, Å | Force FD error fp32, eV/Å | Force FD error fp64, eV/Å |','|---|---:|---:|']
        for x,y in zip(cpu['finite_differences']['fp32']['force_finite_difference'],cpu['finite_differences']['fp64']['force_finite_difference']):lines.append(f'| {x["h_A"]:.0e} | {x["absolute_error_eV_A"]:.6g} | {y["absolute_error_eV_A"]:.6g} |')
        lines += ['', 'Для малого h видна разница в устойчивости finite differences энергии: при h=1e−4 Å ошибка fp32 составляет0.0300 eV/Å, fp64 —6.14e−9 eV/Å. При этом сами силы autodiff совпадают гораздо лучше: максимальное различие по panel1.32e−6 eV/Å. Поэтому fp64 сохраняет пользу для строгих derivative checks, даже когда fp32 достаточен для самих предсказаний сил.',
            '', 'Фононные различия fp32−fp64 малы по сравнению с исходной модельной ошибкой к DFT: RMSE около1e−4 THz против0.47/0.78 THz. На ASR mesh50³ обе точности не дают мод ниже−0.001 THz. Длительная MD-стабильность здесь не проверялась.',
            '', 'Дополнительное наблюдение для подготовки FT: raw energy mismatch к r²SCAN на этом panel около4.08 eV/atom, почти одинаковый в обеих точностях. Он требует аудита энергетических отсчётов и train-only per-element alignment на всём N200 до настройки GRACE FT; это не эффект fp32. Panel не используется для подгонки offsets.',
            '', '![Timing spread, memory, derivatives and phonons](grace-precision-20261107/cpu-precision.png)', '',
            'CPU результаты позволяют оценить влияние округления на выбранных структурах. Скорость A100 пока не измерена; выбор production precision остаётся открытым до GPU benchmark. Возможная схема после проверки: fp32 для массового inference, fp64 для строгих derivative/phonon diagnostics. Перевод обученных fp32 весов в fp64 повышает точность арифметики, но не восстанавливает информацию, утраченную при fp32 training.']
    else:lines += ['', 'CPU измерения выполняются; численные результаты ещё не объявлены.']
    lines += ['', 'Новых DFT и training jobs нет; test149 и архивные49 в сравнении не использованы.']
    text='\n'.join(lines)+'\n'
    for old,new in {'попытка209':'попытка 209','GPU-задачи209':'GPU-задачи 209','array209':'array 209','составляет0.0300':'составляет 0.0300','—6.14e':'— 6.14e','panel1.32e':'panel 1.32e','около1e':'около 1e','против0.47':'против 0.47','mesh50':'mesh 50','ниже−':'ниже −','около4.08':'около 4.08','test149':'test 149','архивные49':'архивные 49'}.items():text=text.replace(old,new)
    (ROOT/report).write_text(text,encoding='utf8')
    text='# Ti–Al FM: текущее состояние\n\nN=694: восемь DPA4 повторов seeds20/30 продолжаются, array20989407.\n\nGRACE fp32/fp64: '+('CPU-сравнение завершено' if cpu else 'CPU-сравнение выполняется')+f'; A100 benchmark {jobs["GPU_benchmark"]} ожидает окончания DPA4. Выбор precision предварительный; GRACE training не запущен.\n\n[Сравнение точности и скорости GRACE](16_grace_precision_comparison.md) · [Запуск N694 и подготовка GRACE](15_n694_repeats_grace_preparation.md) · [Сравнение завершённых16 DPA4](14_multiseed_fulltrain_comparison.md).\n'
    (ROOT/'reports/summary.md').write_text(text,encoding='utf8')
    print(json.dumps(dict(CPU='PASS' if cpu else 'running',GPU='queued',report=report)))

if __name__=='__main__':main()
