"""Summarize verified equilibrium-cell DOS without changing the manuscript model."""
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent
OUT=HERE/'pbe_phonons_15k_equilibrium/comparison'
a=json.loads((OUT/'analysis.json').read_text(encoding='utf-8'))
p=a['width_checks']['0.05']
assert all(v==['15000','5000'] for d in a['own_cell_rankings'].values() for v in d.values())
assert all(d['imaginary_below_minus_0p01_THz']==0 for d in a['mesh_frequency_info'].values())
improvement={m:100*(1-p['15000/own/MP40'][m]/p['5000/own/MP40'][m]) for m in ['L2_relative','TV','W1_THz']}
lines=['DOS hcp-Ti: 15k при собственной равновесной ячейке; сравнение с 5k и DFT',
 '25.10.2026','',
 'ГЛАВНЫЙ РЕЗУЛЬТАТ',
 '15k остаётся ближе к DFT по всем трём интегральным метрикам DOS после',
 'независимой релаксации a,c обеих моделей. Это выполняется для трёх q-сеток',
 'и трёх Gaussian-уширений (всего 27 сравнений). Преимущество по L2 меньше,',
 'чем при общей DFT-геометрии: около 10% вместо 20%.',
 'Оно не означает улучшение каждой детали спектра: главный высокочастотный',
 'пик 5k при своей ячейке ближе к DFT, чем пик 15k.','',
 'РАВНОВЕСНЫЕ ЯЧЕЙКИ',
 'checkpoint | a, A | c, A | c/a | V/atom, A^3 | max abs(stress), GPa']
for k,d in a['geometry'].items():
    lines.append(f"{k} | {d['a_A']:.8f} | {d['c_A']:.8f} | {d['c_over_a']:.8f} | {d['V_per_atom_A3']:.8f} | {max(map(abs,d['stress_GPa'])):.8f}")
lines+=['DFT reference | 2.95000000 | 4.68000000 | 1.58644068 | 17.63561142 | not a model equilibrium','',
 'МЕТРИКИ DOS: MP 40x40x30, sigma=0.05 THz, 3 моды/атом',
 'model/geometry | L2 relative | total variation | W1, THz | mean frequency, THz | main peak, THz']
for k in ['5000','15000']:
    for geom in ['DFT-cell','own']:
        d=p[k+'/'+geom+'/MP40']
        lines.append(f"{k}/{geom} | {d['L2_relative']:.6f} | {d['TV']:.6f} | {d['W1_THz']:.6f} | {d['mean_THz']:.6f} | {d['main_peak_THz']:.2f}")
lines+=['DFT | 0 | 0 | 0 | 5.300667 | 6.84','']
for m,v in improvement.items():lines.append(f'Уменьшение {m} для 15k относительно 5k при своих ячейках: {v:.2f}%')
lines+=['',
 'ИНТЕРПРЕТАЦИЯ',
 'Релаксация уменьшает объём относительно DFT-ячейки и увеличивает среднюю',
 'частоту обеих моделей. При переходе к собственным ячейкам расхождения',
 'с архивной DFT-DOS увеличиваются. Несмотря на это, преимущество 15k по',
 'интегральной форме спектра сохраняется. Его выигрыш не обусловлен только',
 'использованием общей DFT-геометрии.',
 'Главные пики: DFT 6.84 THz; 5k own 6.92 THz; 15k own 7.08 THz.',
 'При общей DFT-геометрии главные пики были 6.74 THz для 5k и 6.91 THz для 15k.',
 'Следовательно, интегральное качество DOS и положение одного пика дают',
 'разные частные оценки; вывод о преимуществе 15k относится к полному спектру.',
 'Предыдущая проверка энергий/сил также благоприятствует 15k относительно 5k.',
 'Это укрепляет основание рассматривать 15k как кандидата для проверки',
 'плавления Ti и энтальпии смешения, но не предсказывает результат этих свойств.','',
 'РАСЧЁТ И ПРОВЕРКИ',
 'CPU Slurm 20960437_0, COMPLETED, exit 0:0, 4 CPU, 83 s, 0.0922 CPU-hours.',
 '15k checkpoint SHA256: '+a['metadata']['15000']['model_sha256'],
 'Нативный DeepPot CPU; версии numpy/scipy/ASE/torch/DeepMD совпадают с 5k.',
 'Оптимизировались независимые a,c при фиксированных дробных координатах hcp,',
 'внешнее напряжение ноль. Остаточные компоненты напряжения <3e-5 GPa,',
 'силы <3e-7 eV/A. Кривизна в подпространстве деформаций a,c положительна.',
 'Проверено соответствие энергий и напряжений двухатомной ячейки её 4x4x4',
 'повторению до и после релаксации. Собственные значения strain Hessian:',
 str(a['metadata']['15000']['optimization']['hcp_strain_hessian_eigenvalues_eV'])+' eV.',
 'Фононы: 9x9x9 (1458 атомов), центральные +/-0.01 A, масса 47.867 amu.',
 'ASE Frederiksen, 3 symmetrization iterations, acoustic sum rule; сохранены',
 'сырые силы, исходные/обработанные силовые постоянные и POSCAR-relaxed.',
 'Для 5k использованы проверенные ранее силы и матрицы при собственной ячейке.',
 'DOS обеих моделей заново построены из матриц одинаковым алгоритмом.',
 'Fourier-преобразование воспроизводит сохранённые ASE-дисперсии в пределах 3e-6 THz.',
 'Сетки: MP 40x40x30, MP 50x50x50, Gamma-centered 40x40x30.',
 'Ширины: 0.05, 0.10, 0.15 THz. Частоты и энергии не сдвигались и не подгонялись.',
 'На проверенных сетках нет мод ниже -0.01 THz. Минимум около Gamma у 15k',
 'примерно -0.000267 THz — малый численный остаток акустической моды.','',
 'ОГРАНИЧЕНИЯ',
 'DFT использовал собственную архивную геометрию и сверхъячейку 4x4x3.',
 'MLIP используют 9x9x9; полная сходимость абсолютных спектров не установлена.',
 'Сравнение собственных ячеек включает и ошибку геометрии, и ошибку силовых',
 'постоянных. Сравнение при общей DFT-геометрии отдельно сохраняется в таблице.',
 'Это гармоническая hcp-Ti DOS; она не определяет автоматически устойчивость',
 'bcc-Ti, температуру плавления или свойства расплавов Ti-Al.',
 'Плавление и энтальпия в этой задаче не пересчитывались; модель статьи не заменялась.','',
 'ФАЙЛЫ',
 'dos_5k_15k_equilibrium.png/svg — сравнение собственных ячеек с DFT.',
 'dos_equilibrium_geometry_effect.png/svg — влияние перехода от DFT-ячейки к своей.',
 'analysis.json — все метрики, геометрии, хеши, сетки и уширения.',
 'dos_comparison.npz — численные DOS.',
 '../15000/POSCAR-relaxed — равновесная структура 15k.',
 '../15000/results.npz и ../15000/forces/ — исходные результаты.']
(OUT/'REPORT_RU.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(OUT/'summary.json').write_text(json.dumps({'own_cell_improvement_percent':improvement,'primary':p,
 'geometry':a['geometry'],'ranking_robust':True},indent=2),encoding='utf-8')
print(json.dumps({'own_cell_improvement_percent':improvement,'report':str(OUT/'REPORT_RU.txt')},indent=2))
