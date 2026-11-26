"""Add log provenance and a Russian report; refresh the main figure from CSV."""
from pathlib import Path
import csv, hashlib, json, os, re, sys
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/'oct20/deps'))
os.environ['MPLBACKEND']='Agg'
os.environ['MPLCONFIGDIR']=str(OUT/'.mplconfig')
import matplotlib.pyplot as plt

ROOT=Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project')
logs=ROOT/'results-scripts-logs/logs'
inventory=[]
excerpts=[]
for path in sorted(logs.glob('run_mixing_enthalpy*.log')):
    text=path.read_text(encoding='utf8',errors='replace')
    params={}
    for field in ['DFT method','T measure','Melt steps','Equil steps','Prod steps','MD timestep','Log interval']:
        match=re.search(r'^'+re.escape(field)+r':\s*(.+)$',text,re.M)
        if match: params[field]=match.group(1)
    inventory.append({'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                      'parameters':params,'equil_stage_messages':text.count('--- EQUIL ---'),
                      'production_stage_messages':text.count('--- PROD ---')})
    if params.get('T measure')=='2000 K':
        excerpts.append(str(path)+'\n'+json.dumps(params,ensure_ascii=False))
        lines=text.splitlines()
        for i,line in enumerate(lines):
            if '--- EQUIL ---' in line:
                excerpts.append('\n'.join(f'{k+1}: {lines[k]}' for k in range(max(0,i-1),min(i+9,len(lines)))))
                break
(OUT/'run_log_inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf8')
(OUT/'run_log_evidence.txt').write_text('\n\n'.join(excerpts),encoding='utf8')
rows=list(csv.DictReader((OUT/'enthalpy_2000K.csv').open(encoding='utf-8-sig')))
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(7.6,4.9),layout='constrained')
for model,c,label in [('pbe','#2878B5','PBE'),('sol','#39835B','PBEsol'),('scan','#D55E00',r'r$^2$SCAN')]:
    g=sorted([r for r in rows if r['model']==model],key=lambda r:float(r['x_Al']))
    ax.errorbar([float(r['x_Al']) for r in g],[float(r['dH_mix_kJ_mol']) for r in g],
                yerr=[float(r['dH_sem_5ps']) for r in g],fmt='o-',ms=4,c=c,capsize=3,label=label)
ax.set(xlabel=r'Al atomic fraction, $x_{Al}$',ylabel=r'$\Delta H_{mix}$ (kJ mol$^{-1}$)',
       title='Ti–Al, 2000 K: archived 30 ps production',xlim=(0,1),ylim=(-35,1))
fig.text(.10,-.045,'Error bars: block SEM (6 × 5 ps), including pure-liquid references\nEquilibration samples were not saved; averaging has not been extended.',fontsize=8)
ax.legend(frameon=False)
ax.grid(alpha=.15)
for ext in ['png','pdf']: fig.savefig(OUT/f'mixing_enthalpy_2000K_recomputed.{ext}',dpi=190,bbox_inches='tight')
plt.close(fig)

report='''Анализ энтальпии смешения Ti–Al при 2000 K — 23 октября 2026

Основной вывод
Добавить последние шаги термализации к усреднению по имеющемуся архиву
невозможно: значения энергии, объёма и температуры этой стадии не записаны.
Утверждать, что 30000 шагов термализации избыточны, по этим данным нельзя.
Пересчитаны все 33 сохранённые производственные траектории (3 модели × 11
составов), построены кривые и диагностика. Дополнительных MD-данных не создано.
Числа основного графика воспроизводят уже исправленные результаты статьи.
Исходные архивы и рисунок в рукописи не перезаписаны.

Что сохранилось и где находятся логи
Основные логи лежат в TiAl-Project/results-scripts-logs/logs, а не в каталоге
2000K. Производственным сериям соответствуют:
  PBE:    run_mixing_enthalpy_20261014_090145.log
  PBEsol: run_mixing_enthalpy_20261014_181921.log
  r2SCAN: run_mixing_enthalpy_20261013_143816.log
Все три указывают 200 атомов, шаг 1 fs, 1 bar, нагрев при 3000 K в течение
20000 шагов, EQUIL при 2000 K — 30000, PROD — 30000, интервал записи — 100.
В логах только сообщения о стадиях и итоговые средние, без временных рядов EQUIL.
В каждом *_2000K.traj ровно 301 кадр: начало PROD и 300 записей через 0.1 ps.
Все 33 cache/*_relaxed.traj содержат ровно по одному кадру минимизированной
структуры до нагрева; это не архив термализации при 2000 K.
utils/md_runner.py передаёт trajectory только для production(), logfile=None
во всех NPT-стадиях. Код повторно генерирует скорости и создаёт новый NPT
интегратор на каждой стадии, включая начало PROD. Поэтому даже сохранённые
EQUIL и PROD следовало бы проверять как участки с перезапуском динамики.
run_log_evidence.txt содержит выдержки с номерами строк; run_log_inventory.json
и sources.json фиксируют пути и SHA-256 источников.

Определение энтальпии и воспроизводимость
H = 96.48533212331 * <(Epot + Ekin + p_ext V)/N>, в kJ/mol атомов.
p_ext=1 bar. DeltaH(x)=H(x)-x H(Al)-(1-x) H(Ti).
Состав определяется по атомным номерам в каждом файле. Чистые компоненты
усредняются тем же способом, при той же температуре и для той же модели.
Главное среднее использует все 301 кадр, как в текущей обработке статьи.
Старые summary.json содержат выражение с +Tr(stress)/3 вместо внешнего
давления. В ASE Tr(stress)/3 имеет знак, противоположный давлению. Замена
на p_ext устраняет также шум мгновенного внутреннего давления.
Это отдельная поправка определения H, а не эффект увеличения статистики.
Максимальная поправка DeltaH относительно summary.json:
  PBE 1.91; PBEsol 3.94; r2SCAN 1.35 kJ/mol.
Проверено воспроизведение старой H до точности округления summary.json (0.005)
и совпадение новой H с audit/oct20/thermo_reprocessed.json лучше 1e-8 kJ/mol.

Результаты при x(Al)=0.5 — минимумы на имеющейся сетке
  PBE:    -26.310 ± 0.216 kJ/mol
  PBEsol: -25.440 ± 0.231 kJ/mol
  r2SCAN: -32.039 ± 0.171 kJ/mol
Здесь ± — диагностическая стандартная ошибка, не 95%-ный доверительный интервал.
По внутренним составам диапазон ошибок:
  PBE: 0.105–0.291; PBEsol: 0.130–0.403; r2SCAN: 0.123–0.330 kJ/mol.

Оценка статистики и стационарности
Для ошибок взяты кадры 1–300 (начальный кадр исключён только из блоковой
диагностики), шесть неперекрывающихся блоков по 5 ps. SEM = sd(средних блоков)/sqrt(6).
Для DeltaH складываются дисперсии сплава и обеих чистых компонент с весами
1, x² и (1-x)², предполагая независимость отдельных запусков. Ошибки разных
составов коррелируют через общие чистые компоненты; матрицы сохранены отдельно.
При x=0 и x=1 DeltaH тождественно равна нулю, и её ошибка равна нулю.
block_size_sensitivity.csv содержит блоки 0.1, 0.5, 1, 2, 3, 5 и 10 ps.
Шесть блоков не позволяют надёжно исключить медленные корреляции; ошибки
остаются диагностическими и не включают ошибки потенциала и конечного размера.

Наибольшие различия DeltaH между первыми и последними 15 ps:
  PBE:    0.945 kJ/mol, x(Al)=0.5;
  PBEsol: 0.774 kJ/mol, x(Al)=0.2;
  r2SCAN: 1.081 kJ/mol, x(Al)=0.7.
По H, температуре и плотности выполнены сравнения половин (Welch, по 3 блока)
и линейные тренды по 6 блокам. Из 99 сравнений половин два имеют p<0.05 до
поправки за множественные проверки: H для PBEsol x=0.2 и T для r2SCAN x=0.7.
После BH-коррекции внутри каждого семейства из 99 тестов нет q<0.05 ни у
сравнений половин, ни у трендов. Это не доказательство равновесия: мощность
коротких рядов мала, а независимость блоков гарантировать нельзя.
Для начала PROD проверена чувствительность к удалению 0, 1, 2, 5, 10 и 15 ps.
Эти перекрывающиеся окна не являются независимыми репликами. Они не дают
оснований выбирать отсечение ради гладкости концентрационной кривой.

Что можно заключить
Основная глубина смешения и различие моделей устойчивее, чем мелкие перегибы.
Разброс около 1 kJ/mol между половинами подтверждает ограниченную точность
тонких деталей. Однако он не устанавливает, что все перегибы вызваны шумом.
30 ps EQUIL нельзя признать избыточными, потому что его временного ряда нет.
Сокращать этот интервал по внешнему виду итоговой кривой было бы необоснованно.
Даже гипотетическое увеличение усреднения с 30 до 50 ps уменьшило бы SEM лишь
до sqrt(30/50)=0.775 исходной при одинаковых корреляциях и стационарности;
для двукратного уменьшения SEM потребовалось бы примерно вчетверо больше
эффективной статистики. Это оценка масштаба, не гарантия сходимости.

Практический следующий расчёт
Чтобы реально увеличить статистику, нужны новые шаги MD или ранее не найденные
численные записи EQUIL. В следующем запуске следует сохранять энергию, объём,
температуру, давление и траекторию на всей стадии при целевой температуре;
выбирать начало усреднения по согласованности нескольких временных окон
для H, плотности и температуры, затем проверять химическую/структурную
релаксацию и корреляции. Лучше продолжать тот же интегратор без повторного
задания скоростей; при перезапуске анализировать переходный участок отдельно.
Независимые начальные состояния позволят проверить разброс между запусками.
Новые MD-расчёты в рамках этой обработки не запускались.

Замечание к методике статьи
Логи 2000 K задают 20000 шагов нагрева (20 ps), тогда как текущий текст
methods.tex описывает 10 ps. Это следует уточнить для данной серии.

Файлы
mixing_enthalpy_2000K_recomputed.png/.pdf — основная кривая с SEM.
formula_and_sampling_comparison.png/.pdf — исходная формула, исправленная
энтальпия и две половины производственного интервала.
production_cutoff_sensitivity.png/.pdf — влияние удаления начала PROD.
production_diagnostics_{pbe,sol,scan}.png/.pdf — H, T, плотность для всех составов.
enthalpy_2000K.csv — значения H, DeltaH, ошибки, сравнение половин.
production_timeseries.npz — численные ряды; названия колонок внутри файла.
stationarity_diagnostics.csv — описанные тесты, p и q.
endpoint_covariance.json — ковариации DeltaH из общих чистых компонентов.
analyze.py — воспроизводимый основной анализ; finalize.py — оформление отчёта.
Воспроизведение: Python 3.12 с ASE, NumPy, SciPy, Matplotlib;
выполнить analyze.py, затем finalize.py. В текущем окружении используются
пакеты audit/oct20/deps и Python из bundled runtime Codex.
'''
(OUT/'report_ru.txt').write_text(report,encoding='utf-8-sig')
print('Report, log provenance and final figure saved.')
