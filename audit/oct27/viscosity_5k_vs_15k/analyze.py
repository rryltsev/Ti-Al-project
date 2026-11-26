"""Compare supplied STACIE exports; do not recompute trajectories or edit manuscript."""
from pathlib import Path
import csv
import hashlib
import json
import math
import sys

OUT = Path(__file__).resolve().parent
WORK = OUT.parents[2]
BASE = Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project\results-scripts-logs')
NEW = BASE / 'stacie_pipline_PBE-15k_27.10.2026/results'
OLD = {1873: BASE/'stacie_pipeline_wsl2/results',
       2000: BASE/'stacie_pipline_wsl2_22.10.2026/results'}
MANIFEST = {}

def read(p):
    MANIFEST[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text(encoding='utf-8-sig'))

def key(r):
    return int(r['T']), r['dft'], float(r['c_Ti'])

def flatten(j):
    return [r for groups in j['temperatures'].values()
            for group in groups.values() for r in group['points']]

def detail(root, r):
    return root/f"AlTi/{r['T']}/{r['dft']}/c_Ti_{r['c_Ti']:.2f}/result.json"

current_path = WORK/'audit/oct25/viscosity/viscosity_merged.json'
current = {key(r): r for r in read(current_path)}
incoming = flatten(read(NEW/'summary.json'))
assert len(incoming) == len({key(r) for r in incoming}) == 8
csv_path = NEW/'summary.csv'
MANIFEST[str(csv_path)] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
with csv_path.open(encoding='utf-8-sig', newline='') as f:
    csv_data = {key(r): r for r in csv.DictReader(f)}
assert set(csv_data) == {key(r) for r in incoming}
changed, unchanged = [], []
for r in incoming:
    assert read(detail(NEW, r)) == r
    for f, v in r.items():
        if f in csv_data[key(r)]:
            if isinstance(v, (float, int)):
                assert math.isclose(float(csv_data[key(r)][f]), v, rel_tol=1e-12)
            else:
                assert csv_data[key(r)][f] == v
    assert math.isclose(r['eta_pa_s']*1000, r['eta_mPa_s'])
    assert math.isclose(r['eta_std_pa_s']*1000, r['eta_std_mPa_s'])
    assert r['n_steps'] == 20001 and r['timestep_ps'] == .001 and r['block_size'] == 1
    target = changed if any(current[key(r)].get(f) != v for f, v in r.items()) else unchanged
    target.append(key(r))
assert set(changed) == {(1873, 'pbe', 1.), (2000, 'pbe', 1.)}
results = []
for t in [1873, 2000]:
    new = next(r for r in incoming if key(r) == (t,'pbe',1.))
    old = next(r for r in flatten(read(OLD[t]/'summary.json')) if key(r) == key(new))
    assert read(detail(OLD[t], old)) == old
    assert all(current[key(old)].get(k) == v for k, v in old.items())
    a, b = old['eta_mPa_s'], new['eta_mPa_s']
    sa, sb = old['eta_std_mPa_s'], new['eta_std_mPa_s']
    delta, sd = b-a, math.hypot(sa,sb)
    ratio = b/a
    sr = ratio*math.hypot(sa/a,sb/b)
    scan = current[(t,'scan',1.)]['eta_mPa_s']
    row = dict(T=t, checkpoint_5k=old, checkpoint_15k=new,
        delta_mPa_s=delta, delta_std_mPa_s=sd,
        delta_95CI_mPa_s=[delta-1.96*sd,delta+1.96*sd],
        relative_change_percent=100*(ratio-1),
        relative_change_95CI_percent_delta_method=[100*(ratio-1-1.96*sr),100*(ratio-1+1.96*sr)],
        abs_z=abs(delta)/sd, two_sided_normal_p=math.erfc(abs(delta)/sd/math.sqrt(2)),
        relative_std_5k_percent=100*sa/a, relative_std_15k_percent=100*sb/b,
        aggregate_production_ps_5k=20*old['n_replicas'],
        aggregate_production_ps_15k=20*new['n_replicas'],
        tau_exp_difference_z=(new['tau_exp_ps']-old['tau_exp_ps'])/math.hypot(new['tau_exp_std_ps'],old['tau_exp_std_ps']),
        resolution_guideline_20pi_tau_ps_5k=20*math.pi*old['tau_exp_ps'],
        resolution_guideline_20pi_tau_ps_15k=20*math.pi*new['tau_exp_ps'],
        scan_eta_mPa_s=scan,
        scan_above_pbe_5k_percent=100*(scan/a-1),scan_above_pbe_15k_percent=100*(scan/b-1),
        source_5k=str(detail(OLD[t],old)),source_15k=str(detail(NEW,new)))
    results.append(row)
    for root in [NEW, OLD[t]]:
        for filename in ['spectrum.png','extras.png']:
            p=detail(root,new).with_name(filename)
            MANIFEST[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()

historical = next(r for r in flatten(read(BASE/'stacie_pipeline_wsl2/results/summary.json')) if key(r)==(2000,'pbe',1.))
payload = dict(date='2026-10-27', comparison=results, changed_states=changed,
    unchanged_states=unchanged, incoming_count=len(incoming),
    historical_2000K_5k=historical,
    hypothetical_change_vs_superseded_2000K_percent=100*(results[1]['checkpoint_15k']['eta_mPa_s']/historical['eta_mPa_s']-1),
    assumptions=['Checkpoint labels follow user information and existing project provenance; no model hashes in new export.',
        'Normal approximation and zero covariance between checkpoint estimates; not verified from raw replicas.',
        'Reported errors are STACIE estimator uncertainty, not total model or protocol uncertainty.',
        'No raw stress trajectories, preparation logs, seeds or numeric cutoff history supplied in new folder.'],
    checks=['All 8 result.json files equal summary.json records and agree with summary.csv.',
        'Units and sample metadata checked; only the two pure-Ti PBE states differ from the latest merged 5k data.',
        'Both old estimates verified against their original summary and detail JSON exports.',
        'All four spectra and four cutoff diagnostic images visually inspected.'],
    methodology_urls=['https://molmod.github.io/stacie/apidocs/stacie.estimate.html',
        'https://molmod.github.io/stacie/preparing_inputs/data_sufficiency.html',
        'https://molmod.github.io/stacie/theory/model.html'])
(OUT/'comparison.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'source_manifest.json').write_text(json.dumps(MANIFEST,ensure_ascii=False,indent=2),encoding='utf-8')

lines = ['Сравнение вязкости чистого Ti: PBE 5k и 15k. 27 октября 2026 года', '',
    'Вывод: наблюдаемое изменение мало и статистически не разрешено имеющимися оценками.',
    'Эти две точки не обосновывают немедленный полный пересчёт по причине обнаруженного',
    'изменения вязкости, но и не доказывают нечувствительность всей системы Ti-Al к чекпойнту.',
    'Если итоговая серия должна целиком соответствовать модели PBE 15k, остальные точки',
    'нужно пересчитать; это требование единства модели, а не следствие значимого сдвига в Ti.', '',
    'Сравниваемые значения (мПа·с; ± = одна стандартная ошибка оценщика STACIE):']
for r in results:
    a,b=r['checkpoint_5k'],r['checkpoint_15k']
    lo,hi=r['delta_95CI_mPa_s']; rlo,rhi=r['relative_change_95CI_percent_delta_method']
    lines += [f"{r['T']} K: 5k {a['eta_mPa_s']:.6f} ± {a['eta_std_mPa_s']:.6f} ({a['n_replicas']} реплик);",
        f"        15k {b['eta_mPa_s']:.6f} ± {b['eta_std_mPa_s']:.6f} ({b['n_replicas']} реплик).",
        f"  Изменение {r['relative_change_percent']:+.3f}%; Δ = {r['delta_mPa_s']:+.6f} ± {r['delta_std_mPa_s']:.6f} мПа·с; |z|={r['abs_z']:.3f}.",
        f"  Приближённый 95% ДИ для Δ: [{lo:+.6f}; {hi:+.6f}] мПа·с.",
        f"  Приближённый 95% ДИ относительного изменения (delta method): [{rlo:+.2f}; {rhi:+.2f}]%.",
        f"  τ_exp: {a['tau_exp_ps']:.6f} -> {b['tau_exp_ps']:.6f} ps; τ_int: {a['tau_int_ps']:.6f} -> {b['tau_int_ps']:.6f} ps.",
        f"  neff: {a['neff']:.2f} -> {b['neff']:.2f}; относительная ошибка η: {r['relative_std_5k_percent']:.2f}% -> {r['relative_std_15k_percent']:.2f}%.", '']
lines += [
    'Статистическая интерпретация',
    'σ_Δ = sqrt(σ_5k² + σ_15k²), ДИ = Δ ± 1.96 σ_Δ. Формулы предполагают',
    'нулевую ковариацию двух оценок и нормальное приближение. Сырых реплик нет:',
    'проверить независимость ансамблей, общие исходные конфигурации и ковариацию нельзя.',
    'Отсутствие обнаруженного различия не является тестом эквивалентности.',
    'В частности, точность ±5% по этим двум проверкам не установлена.',
    'Обе центральные оценки стали ниже, но систематическое снижение не доказано.',
    'Падение вязкости с температурой и порядок η(r2SCAN)>η(PBE) сохраняются.',
    'Изменения τ_exp меньше одной объединённой стандартной ошибки; τ_int меняется мало.',
    'Уменьшение ошибки при 1873 K сопровождается ростом числа реплик 15->25;',
    'при 2000 K число реплик меньше (50->20), и ошибка выше. Это не оценка качества модели.', '',
    'Выбор исходных данных и происхождение',
    'При 1873 K использован stacie_pipeline_wsl2/results; при 2000 K — обновлённый',
    'stacie_pipline_wsl2_22.10.2026/results (50 реплик). Оба совпадают с текущим',
    'объединённым набором audit/oct25/viscosity/viscosity_merged.json.',
    f"Устаревшая оценка при 2000 K равна {historical['eta_mPa_s']:.6f} ± {historical['eta_std_mPa_s']:.6f} (15 реплик).",
    f"Сравнение с ней дало бы {payload['hypothetical_change_vs_superseded_2000K_percent']:+.2f}%, т.е. даже другой знак сдвига.",
    'Её нельзя подставлять вместо актуальной оценки 5k или усреднять с ней без знания',
    'перекрытия ансамблей. Это иллюстрирует чувствительность вывода к выбору статистики.',
    'В новом экспорте 8 записей: меняются только PBE c_Ti=1 при 1873 и 2000 K.',
    'Шесть записей полностью совпадают с прежними: четыре r2SCAN и две PBE при',
    '1873 K, c_Ti=0.50 и 0.55. Они не являются новыми проверками 15k для сплавов.',
    'Обозначение двух новых точек как 15k основано на сообщении пользователя;',
    'в JSON нет номера/checksum модели. Точный MD-протокол 15k из экспорта не восстановить.', '',
    'Диагностика сходимости',
    'Все четыре сравниваемых расчёта имеют 20001 отсчёт с интервалом 0.001 ps:',
    '20 ps на реплику. Спектры LorentzModel имеют сходную форму, видимого нового',
    'аномально узкого пика у 15k нет. Оценки устойчивее к cutoff в области его основных',
    'весов, чем при включении высоких частот; при больших cutoff η заметно снижается.',
    'По extras.png у 15k, особенно 2000 K, Z-score критерия cutoff около 2 и выше',
    'в существенной части области весов. Это визуальная оценка, точные Z-score',
    'и история подгонки не экспортированы. Z-score стоимости существенно ниже 2',
    'в основной области весов. У 1873 K распределение весов имеет два локальных пика.',
    'Положительные собственные значения Гессиана на изображениях не заменяют',
    'проверку зависимости результата от длины траекторий и модели спектра.',
    'В документации STACIE рекомендуется neff > 20P; LorentzModel имеет P=3,',
    'т.е. ориентир около 60. Полученные neff=30–40 ниже этого ориентира для обеих',
    'моделей. neff — эффективное число спектральных частот, не число реплик.',
    'Критерий разрешения пика t_sim ≳ 20π τ_exp выполняется: для 15k он даёт',
    'около 9.9 и 7.9 ps, меньше 20 ps. Этот критерий не доказывает полной сходимости.',
    'Рекомендуется проверить 20->40 ps на контрольных состояниях, при необходимости',
    'до 80 ps, и проверить cutoff/альтернативную разумную модель низких частот.',
    'Простое добавление реплик снижает шум, но не улучшает частотный шаг 1/t_sim.', '',
    'Рекомендация по дальнейшим расчётам',
    '1. Не запускать полный пересчёт только на основании этих двух небольших сдвигов.',
    '2. До решения о всей кривой проверить 15k на сплавах при 1873 K:',
    '   x_Al=0.20 (Ti-богатая ветвь), 0.40 или 0.45 (вершина), 0.80 (Al-богатая',
    '   ветвь и чувствительное сравнение с экспериментом). Здесь x_Al=1-c_Ti.',
    '   Практический допуск, например ±5%, нужно выбрать до анализа контрольных точек;',
    '   для заявления эквивалентности нужен соответствующий доверительный интервал,',
    '   а не только |Δ|<2σ. Если интервал слишком широк, сначала увеличить статистику.',
    '3. Если контроль выявит воспроизводимые сдвиги, превышающие выбранный допуск,',
    '   или изменение формы максимума/сравнения с экспериментом — пересчитать серию.',
    '4. Для публикации всех результатов как одного потенциала PBE 15k полный',
    '   пересчёт оставшихся 12 из 14 STACIE-состояний PBE методологически оправдан:',
    '   10 составов сплава при 1873 K и Ti при 1941/2100 K. Новые две точки уже есть.',
    '   Альтернатива — явно маркировать сохранённые данные 5k и отдельные проверки 15k.',
    '   Нельзя просто переименовать прежнюю кривую в 15k или масштабировать её',
    '   коэффициентом, полученным для чистого Ti.',
    '   Отдельная серия 1773 K (TDM) не входит в эти 14 состояний и здесь не проверена.', '',
    'Источники методики:', *payload['methodology_urls'], '',
    'Числа и проверки: comparison.json. Файлы и SHA256: source_manifest.json.',
    'Воспроизведение: python analyze.py. Рукопись и исходные результаты не изменялись.']
(OUT/'report_ru.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps([{k:r[k] for k in ['T','delta_mPa_s','delta_std_mPa_s','relative_change_percent','abs_z','delta_95CI_mPa_s','relative_change_95CI_percent_delta_method']} for r in results],indent=2))
print('Validated: 8 incoming states; 2 changed; 6 unchanged.')

sys.path.insert(0,str(OUT/'python_deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig, axs = plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
colors = {'5k':'#2965a8','15k':'#d55e00'}
for checkpoint, offset in [('5k',-9),('15k',9)]:
    points=[r['checkpoint_'+checkpoint] for r in results]
    axs[0].errorbar([r['T']+offset for r in results],[p['eta_mPa_s'] for p in points],
        yerr=[p['eta_std_mPa_s'] for p in points],fmt='o',capsize=5,ms=7,
        color=colors[checkpoint],label='PBE '+checkpoint)
axs[0].set(title='Чистый Ti: вязкость',xlabel='Температура, K',ylabel='η, мПа·с',
    xticks=[1873,2000],xlim=(1845,2028),ylim=(2.45,3.55))
axs[0].legend(frameon=False)
axs[0].text(.04,.04,'Погрешности: ±1σ STACIE',transform=axs[0].transAxes,fontsize=9,color='.35')
for i,r in enumerate(results):
    lo,hi=r['delta_95CI_mPa_s']; v=r['delta_mPa_s']
    axs[1].errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',capsize=5,ms=7,color='#74498c')
    axs[1].annotate(f"{r['relative_change_percent']:+.2f}%  ·  {r['abs_z']:.2f}σ",(v,i),
        xytext=(0,14),textcoords='offset points',ha='center',fontsize=10)
axs[1].axvline(0,color='.4',lw=1,ls='--')
axs[1].set(title='Разность 15k − 5k',xlabel='Δη, мПа·с',yticks=[0,1],
    yticklabels=['1873 K','2000 K'],ylim=(1.55,-.6),xlim=(-.48,.38))
axs[1].text(.04,.03,'Интервалы: ≈95%, при нулевой ковариации',transform=axs[1].transAxes,fontsize=9,color='.35')
for ax in axs: ax.grid(alpha=.17)
fig.savefig(OUT/'comparison.png',dpi=200)
fig.savefig(OUT/'comparison.svg')
plt.close(fig)
