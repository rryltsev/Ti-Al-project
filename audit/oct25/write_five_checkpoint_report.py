"""Summarize the completed five-checkpoint comparison without selecting on test data."""
from pathlib import Path
import json

HERE=Path(__file__).resolve().parent
OUT=HERE/'pbe_phonons_five_checkpoints'
a=json.loads((OUT/'analysis.json').read_text(encoding='utf-8'))
p=a['width_checks']['0.05']
keys=['5000','10000','15000','20000','23400']
ranking=['15000','5000','23400','20000','10000']
assert all(v==ranking for d in a['rankings'].values() for v in d.values())
assert all(v['imaginary_below_minus_0p01_THz']==0 for v in a['mesh_frequency_info'].values())
rows=[]
for k in keys:
    m=p[k+'/MP40']; e=a['errors'][k]
    rows.append(f"{int(k)/1000:4g}k | {m['L2_relative']:.6f} | {m['TV']:.6f} | {m['W1_THz']:.6f} | "
                f"{e['train']['energy_rmse_meV_atom']:.4f} | {e['validation']['energy_rmse_meV_atom']:.4f} | "
                f"{e['test20']['energy_rmse_meV_atom']:.4f} | {e['validation']['force_rmse_eV_A']:.6f} | {e['test20']['force_rmse_eV_A']:.6f}")
improve={metric:100*(1-p['15000/MP40'][metric]/p['5000/MP40'][metric]) for metric in ['L2_relative','TV','W1_THz']}
improve['test_energy_RMSE']=100*(1-a['errors']['15000']['test20']['energy_rmse_meV_atom']/a['errors']['5000']['test20']['energy_rmse_meV_atom'])
report='''СРАВНЕНИЕ ПЯТИ PBE CHECKPOINTS ПРИ ТОЧНОЙ DFT-ГЕОМЕТРИИ
25.10.2026

ГЛАВНЫЙ РЕЗУЛЬТАТ
15k лучше 5k одновременно по всем трём метрикам DFT-DOS и по полным train,
validation и текущим test-ошибкам энергий и сил. Он является более сильным
кандидатом для компромисса между гармонической динамикой hcp-Ti и подгонкой E/F.
Это не означает оптимальность для всех MD-свойств Ti-Al.

Порядок DOS во всех 27 сочетаниях метрики, сетки и ширины:
15k < 5k < 23.4k < 20k < 10k (меньше — лучше).

ТАБЛИЦА
Основная сетка: Monkhorst-Pack 40x40x30, Gaussian sigma=0.05 THz.
checkpoint | DOS L2_rel | TV | W1, THz | E_RMSE train | E_RMSE val | E_RMSE test | F_RMSE val | F_RMSE test
E в meV/atom; F — Cartesian component RMSE в eV/A.
'''+ '\n'.join(rows) + '\n\n' + '\n'.join(f'Уменьшение {k} у 15k относительно 5k: {v:.2f}%' for k,v in improve.items()) + '''

ИНТЕРПРЕТАЦИЯ
1. 5k лучше 20k и 23.4k по DOS, как было установлено раньше, но не лучше всех
   сохранённых checkpoints. Расширение сравнения обнаружило преимущество 15k.
2. Изменение DOS немонотонно. Объяснять все результаты одним монотонным
   ухудшением фононов после 5k нельзя: 10k хуже, 15k лучше, затем DOS хуже снова.
3. 20k и 23.4k всё ещё значительно точнее по энергиям, чем 15k. Поэтому нет
   единственного checkpoint, минимизирующего одновременно все ошибки.
4. По паре DOS/E-RMSE недоминируемые варианты — 15k, 20k и 23.4k; это верно
   для validation и текущего test, трёх метрик DOS, всех проверенных сеток/ширин.
   Здесь недоминируемый означает отсутствие другого варианта, который был бы
   не хуже сразу по обеим координатам и строго лучше хотя бы по одной.
5. 10k не даёт полезного компромисса: 20k и 23.4k лучше по DOS, энергиям и силам.
6. Test использован для ретроспективной диагностики; выбор нового checkpoint
   следует обосновывать validation и физическими критериями, не выдавая
   просмотренный test за новый слепой тест.

ПРОТОКОЛ И ПРОВЕРКИ
Точная архивная DFT-геометрия: a=2.950 A, c=4.680 A, V/N=17.635611419 A^3.
Геометрия не релаксировалась. SHA256 POSCAR, ячейка и дробные координаты
совпадают у всех пяти моделей. Проверены идентификаторы моделей и шаги 9999/14999.
Новые CPU-задачи: 20960432_0 и 20960432_1, обе COMPLETED, exit=0:0,
4 CPU на задачу, 90 и 91 s соответственно; 0.2011 выделенных CPU-часов, без GPU.
Сверхъячейка 9x9x9, 1458 атомов; центральные смещения +/-0.01 A;
масса Ti 47.867 amu; ASE Frederiksen, symmetrize=3, acoustic=True.
Код расчёта и версии окружения идентичны прежним 5k/20k/23.4k.
Сохранены 13 наборов сил на модель, исходные и исправленные силовые постоянные.
Новые DOS получены из этих матриц; Fourier-преобразование проверено относительно ASE
на тех же 200 q-точках с допуском 3e-6 THz. Старые три DOS использованы повторно
из проверенного архива; повторный расчёт метрик воспроизводит прежние значения.
Все DOS нормированы к 3 модам/атом, частоты и энергии не сдвигались и не подгонялись.
Сетки: MP 40x40x30, MP 50x50x50, Gamma-centered 40x40x30.
Ширины: 0.05, 0.10, 0.15 THz. На всех сетках нет частот ниже -0.01 THz.
Остаточные отрицательные акустические частоты Gamma у 15k меньше 0.0008 THz
по абсолютной величине и рассматриваются как численный остаток.

ОГРАНИЧЕНИЯ И СЛЕДУЮЩИЙ ШАГ
Это сравнение гармонической DOS hcp-Ti при фиксированной DFT-геометрии.
DFT использовал сверхъячейку 4x4x3, MLIP — 9x9x9; полная сходимость
абсолютного спектра и полная идентичность DFT/MLIP обработки не установлены.
Для 10k/15k отдельные равновесные a,c и DOS в них здесь не вычислялись.
15k следует проверить при собственной равновесной ячейке и на ключевых
MD-свойствах перед заменой модели статьи. Уже полученные MD-результаты относятся
к 5k и не могут быть приписаны 15k. Текущая рукопись в этой задаче не изменялась.

ФАЙЛЫ
dos_five_checkpoints.png/svg — все пять DOS и высокочастотный участок.
checkpoint_tradeoff.png/svg — сравнение DOS с E_RMSE validation/test и метрик DOS.
analysis.json — метрики, идентификаторы, проверки сеток/уширения, Pareto-наборы.
dos_comparison.npz — численные DOS на общей частотной сетке.
../pbe_phonons_dft_geometry_intermediate/10000 и /15000 — исходные новые расчёты.
../pbe_phonons_dft_geometry/ — исходные результаты 5k, 20k, 23.4k и DFT-референс.
'''
assert all(set(v)=={'15000','20000','23400'} for v in a['pareto'].values())
(OUT/'REPORT_RU.txt').write_text(report,encoding='utf-8')
(OUT/'summary.json').write_text(json.dumps({'ranking':ranking,'improvement_15k_vs_5k_percent':improve,
 'checkpoints':{k:{'DOS':p[k+'/MP40'],'errors':a['errors'][k]} for k in keys},
 'interpretation':'15k improves DOS and E/F over 5k. Later models remain more accurate in energies. MD properties not recalculated.'},indent=2),encoding='utf-8')
print(report[:3100])
