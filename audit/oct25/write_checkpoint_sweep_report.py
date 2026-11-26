from pathlib import Path
import json,statistics
HERE=Path(__file__).resolve().parent;P=HERE/'pbe_checkpoint_sweep'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
a=read(P/'analysis.json');pilot=read(P/'pilot-verification.json');inp=read(P/'input-manifest.json')
lines=['Проверка пяти PBE-checkpoints на CPU, 25.10.2026','',
'Все модели относятся к TiAl_tune_pbe_Neo_DPA4_shard_20260920_140047.',
'Оценены 850 train, 149 validation и фиксированные 20 liquid test frames.',
'Checkpoints: 5000, 10000, 15000, 20000 и 23400; исходные модели не менялись.',
'Использован native DeepPot CPU без GPU, без нового AOT-экспорта, обучения',
'или подгонки энергетических отсчётов. Checkpoint 5k побайтно совпал с',
'ранее переданным pbe_checkpoint.pt, а CPU-предсказания совпали с его',
'ранее проверенным frozen-экспортом в пределах численной точности.',
'','Энергии: meV/атом, знак = модель минус DFT TOTEN.',
'Силы: ошибки Cartesian-компонент, eV/A. Stress: 6 компонент, GPa.',
'Train/validation включают разные исходные семейства; test20 — жидкий пул.',
'','Главные результаты:',
'Финальный checkpoint уменьшает RMSE энергии относительно 5k:',
'train 5.214 -> 1.201; validation 6.071 -> 1.233; test20 6.406 -> 1.341.',
'RMSE сил также уменьшается: train 0.06298 -> 0.05949;',
'validation 0.06669 -> 0.06468; test20 0.06754 -> 0.06540 eV/A.',
'Улучшение сил наблюдается на 850/850 train, 147/149 validation и 20/20 test.',
'Следовательно, эти метрики не подтверждают общее ухудшение обобщения',
'от 5k к финальному checkpoint. Рост train/val-разрыва по силам сам по себе',
'не означает рост validation-ошибки: здесь она продолжает снижаться.',
'Смещение Ti-богатых энергий относится к ранней выбранной модели:',
'при xTi>=0.9 mean train +10.933 -> +0.781; validation +11.347 -> +1.260.',
'Test009: +17.926 -> +3.656. При фиксированном составе Al11Ti205 крупное',
'смещение и его выраженная объёмная тенденция существенно ослабевают.',
'На Ti-богатом train при V/N>=20 mean +9.078 -> +1.330 meV/атом.',
'Финальный PBE имеет меньший energy RMSE, чем pretrained на train',
'(1.201 против 2.786) и test20 (1.341 против 3.398). Предыдущий обратный',
'порядок относился к checkpoint 5k и не характеризует весь fine-tuning.',
'Модель 20k даёт немного меньший energy RMSE на validation/test20, чем',
'23400; силы на test20 почти одинаковы. На текущих 20 геометриях разность',
'RMSE энергий 23400 минус 20000 равна +0.113 meV/атом, описательный парный',
'95% интервал [0.005,0.216]. Это локальное сравнение небольшой выборки,',
'не доказательство универсального превосходства 20k.',
'Наблюдение автора о лучших фононах при 5k остаётся отдельным критерием.',
'Следующая проверка компромисса — сопоставить фононы 5k/20k/23400 при',
'одинаковой ячейке и параметрах конечных смещений, затем при собственных',
'равновесных геометриях, отделив кривизну поверхности от сдвига равновесия.',
'','Шаг | split | N | E MAE | E RMSE | E bias | F MAE | F RMSE | stress RMSE']
for key,groups in a['groups'].items():
    for split,g in groups.items():
        lines.append(f"{key:>5} {split:>10} {g['n']:4d} {g['energy_mae_meV_atom']:8.4f} {g['energy_rmse_meV_atom']:8.4f} {g['energy_bias_meV_atom']:+8.4f} {g['force_mae_eV_A']:.6f} {g['force_rmse_eV_A']:.6f} {g['stress_rmse_GPa']:.5f}")
lines+=['','Ti-богатые подмножества: шаг | split | группа | N | E RMSE | E bias | F RMSE']
for key,groups in a['subgroups'].items():
    for split,gg in groups.items():
        for name,g in gg.items():
            if not g:continue
            lines.append(f"{key} {split} {name} {g['n']} {g['energy_rmse_meV_atom']:.4f} {g['energy_bias_meV_atom']:+.4f} {g['force_rmse_eV_A']:.6f}")
lines+=['','Test009: шаг | ошибка энергии (meV/атом)']
for key,r in a['test009'].items():lines.append(f"{key} {r['energy_error_meV_atom']:+.6f}")
lines+=['','Парное сравнение 23400 против 5000:']
for split,g in a['paired_frame_counts'].items():
    lines.append(f"{split}: из {g['n']} структур абсолютная ошибка энергии меньше у финальной на {g['final_lower_absolute_energy']}; RMSE сил меньше на {g['final_lower_force_rmse_per_frame']}.")
lines+=['','Описательные 95% bootstrap-интервалы по целым 20 тестовым конфигурациям:',
'2000 парных перевыборок; неизвестные корреляции траекторий не учтены.',
'Это не утверждение об оптимальном checkpoint вне текущей выборки.']
for prop,g in a['common20_paired_bootstrap'].items():lines.append(prop+': '+json.dumps(g))
lines.append('Сравнение 23400 минус 20000: '+json.dumps(a['common20_late_checkpoint_bootstrap']))
metas=[read(P/f'result-{i}.json') for i in range(40)]
times=[t for meta in metas for t in meta['inference_seconds']]
acct=(P/'full-accounting.txt').read_text().splitlines()
cpu_seconds=sum(int(r.split('|')[5]) for r in acct if '.' not in r.split('|')[0] and r.split('|')[1]=='COMPLETED')
compute={'jobs':40,'max_parallel_jobs':8,'cpus_per_job':4,'gpu_requested':0,
    'submission_to_final_collection_minutes':(read(P/'full-collection.json')['unix']-read(P/'full-submission.json')['unix'])/60,
    'cpu_hours_including_startup':cpu_seconds/3600,'mean_inference_seconds_per_configuration':statistics.mean(times),
    'median_inference_seconds_per_configuration':statistics.median(times),'nodes':sorted(set(v['node'] for v in metas)),
    'n_predictions':len(times),'max_task_wall_seconds':max(v['total_wall_seconds'] for v in metas)}
(P/'compute_summary.json').write_text(json.dumps(compute,indent=2))
lines+=['','Вычислительная стоимость:',json.dumps(compute,ensure_ascii=False,indent=2),
'','Ограничения и дальнейший контроль:',
'Это диагностическое сравнение после выбора 5k по фононам Ti. Выбранная',
'для статьи модель и архив common20 не заменялись. Новые DFT не запускались.',
'Фононы для двух checkpoints здесь заново не считались; утверждение о',
'преимуществе 5k по фононам остаётся наблюдением автора, а не результатом',
'этого energy/force benchmark. Нельзя выбирать новый потенциал только',
'по минимуму ошибок этих 20 конфигураций и считать эту же выборку слепой.',
'Для смены модели нужно отдельно проверить фононы и согласованность',
'MD-свойств с новым checkpoint. Особенно важны ошибки энергий между',
'фазами и составами; средние RMSE сами по себе их не ограничивают.',
'','Файлы:',
'analysis.json — все агрегаты, подмножества и bootstrap.',
'per_configuration.json — ошибки каждого checkpoint на каждой структуре.',
'checkpoint_comparison.png — эволюция ошибок и смещения.',
'fixed_composition_volume.png — зависимость ошибок от объёма при двух фиксированных составах.',
'input-manifest.json и analysis-source-manifest.json — хеши и происхождение.',
'labels-local-only.npz — исходные DFT-метки; на кластер не передавались.',
'result-*.npz/json — исходные предсказания и сведения о CPU-задачах.']
(P/'REPORT_RU.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(compute,indent=2))
