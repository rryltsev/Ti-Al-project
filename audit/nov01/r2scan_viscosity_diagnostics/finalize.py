from analyze import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
d=read(OUT/'results.json');m=read(OUT/'model_check.json');extra=read(OUT/'supplement.json')
w=read(OUT/'windows.json')
assert len(d['temperatures'])==len(m)==4
temps=np.array([1873,1941,2000,2100]);es=[d['temperatures'][str(t)] for t in temps]
ys=np.array([e['auto']['eta'] for e in es]);ses=np.array([e['auto']['se'] for e in es]);fit=d['arrhenius']
# Repair older run history serialization from the independent model-check pass.
for t,e in zip(temps,es):e['history']=m[str(t)]['lorentz_history']
save('results.json',d)
grid=m['1873']['dense_cutoff'];common=[]
for i,r in enumerate(grid):
    if all(m[str(t)]['dense_cutoff'][i]['criterion_zscore']<2 and m[str(t)]['dense_cutoff'][i]['cost_zscore']<2 for t in temps):common.append(r['fcut_THz'])
fixed125=[next(x for x in e['scan'] if x['fcut_THz']==1.25) for e in es]
summary={'auto':dict(zip(map(str,temps),[e['auto'] for e in es])),'common_cutoff_passing_grid_THz':common,'common_cutoff_note':'Exploratory diagnostics < 2, not independent validation or guarantee of unbiasedness. Finite scan spacing 0.05 THz. neff >=60 is not met in this window.','fixed125':dict(zip(map(str,temps),fixed125)),'arrhenius':fit,'fixed125_arrhenius':arrhenius(temps,[a['eta'] for a in fixed125],[a['se'] for a in fixed125]),'max_relative_change_block20_percent':float(max(abs(e['variants']['block20']['eta']/e['auto']['eta']-1)*100 for e in es)),'max_relative_change_grid_percent':float(max(abs(e['variants']['dense_grid']['eta']/e['auto']['eta']-1)*100 for e in es))}
save('summary.json',summary)

fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
tt=np.linspace(1850,2125,400);X=np.array([np.ones(len(tt)),1000/tt]).T;mu=fit['ln_eta0']+fit['B_K']/tt
err=np.sqrt(np.einsum('ij,jk,ik->i',X,np.array(fit['cov']),X));yy=np.exp(mu)
ax.fill_between(tt,np.exp(mu-1.96*err),np.exp(mu+1.96*err),color='#247c83',alpha=.14,label='95% confidence band for Arrhenius fit')
ax.plot(tt,yy,color='#247c83',lw=2,label='Weighted Arrhenius fit')
ax.errorbar(temps,ys,yerr=ses,fmt='o',ms=7,color='#b74b37',capsize=4,label='Original STACIE estimates, +/- 1 SE')
for t,y,e in zip(temps,ys,es):ax.annotate(f"{e['n_replicas']} x 20 ps",(t,y),xytext=(7,12),textcoords='offset points',fontsize=9)
ax.set(xlabel='Temperature (K)',ylabel='Shear viscosity (mPa s)',title='Liquid Ti, r2SCAN: existing data support a smooth trend')
ax.text(.97,.95,f"chi-square / dof = {fit['chi2']:.3f} / 2\nmaximum residual = {max(abs(z) for z in fit['z']):.2f} SE",transform=ax.transAxes,ha='right',va='top',fontsize=10)
ax.grid(alpha=.2);ax.legend(loc='lower left',fontsize=9)
fig.savefig(OUT/'temperature_smooth.png',dpi=190);fig.savefig(OUT/'temperature_smooth.svg');plt.close(fig)

fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained');colors=['#b74b37','#2264a1','#30835b','#8c59a8']
for t,e,c in zip(temps,es,colors):
    sc=m[str(t)]['dense_cutoff'];ff=[x['fcut_THz'] for x in sc]
    axs[0,0].plot(ff,[100*(x['eta']/e['auto']['eta']-1) for x in sc],color=c,label=f'{t} K')
    axs[0,1].plot(ff,[x['criterion_zscore'] for x in sc],color=c,label=f'{t} K')
axs[0,0].axhline(0,color='gray',lw=.8);axs[0,0].set(xlabel='Fixed cutoff (THz)',ylabel='Change from automatic estimate (%)',title='a  Sensitivity to frequency cutoff')
axs[0,1].axhline(2,color='black',ls='--',lw=1);axs[0,1].set(xlabel='Fixed cutoff (THz)',ylabel='Cutoff criterion Z',title='b  Spectral consistency diagnostic',ylim=(-2,12))
for ax in axs[0]:ax.axvline(1.25,color='gray',ls=':',lw=1);ax.legend(fontsize=9);ax.grid(alpha=.2)
for label,key,marker in [('Block average 20 fs','block20','o'),('Discard first 1 ps','discard1ps','s'),('Three off-diagonal channels','offdiag3','^')]:
    axs[1,0].plot(temps,[100*(e['variants'][key]['eta']/e['auto']['eta']-1) for e in es],marker+'-',label=label)
axs[1,0].plot(temps,[100*(m[str(t)]['models']['ExpPoly024']['eta']/e['auto']['eta']-1) for t,e in zip(temps,es)],'D-',label='ExpPoly (0, 2, 4)')
axs[1,0].axhline(0,color='gray',lw=.8);axs[1,0].set(xlabel='Temperature (K)',ylabel='Change from automatic estimate (%)',title='c  Processing and model checks');axs[1,0].legend(fontsize=8);axs[1,0].grid(alpha=.2)
ix=np.arange(4);width=.25
axs[1,1].bar(ix-width,ses,width,label='Formal STACIE SE',color='#537ba3')
axs[1,1].bar(ix,[e['jackknife']['1']['auto_se'] for e in es],width,label='Leave one replica out',color='#3f977b')
axs[1,1].bar(ix+width,[e['jackknife']['5']['auto_se'] for e in es],width,label='Leave five replicas out',color='#c98b48')
axs[1,1].set(xticks=ix,xticklabels=temps,xlabel='Temperature (K)',ylabel='Standard error (mPa s)',title='d  Uncertainty checks',ylim=(0,.24));axs[1,1].legend(fontsize=8);axs[1,1].grid(axis='y',alpha=.2)
fig.savefig(OUT/'processing_diagnostics.png',dpi=170);fig.savefig(OUT/'processing_diagnostics.svg');plt.close(fig)

lines=['Аудит температурной вязкости чистого Ti, r2SCAN — 1 ноября 2026',
'', 'Вывод: существующие четыре точки не обнаруживают статистически значимого отклонения от гладкого тренда. Рекомендуется сохранить исходные оценки и ошибки STACIE, а гладкую зависимость показать отдельным взвешенным Arrhenius-fit. Подбирать cutoff для красоты кривой не следует.',
'', 'Происхождение данных:',
'В предоставленной папке для чистого Ti: 1773 K — 25 SACF, без pressure; 1873 K — 30 pressure; 1943 K — 50 SACF, без pressure; 2000 K — 15 pressure.',
'1941 и 2100 K найдены в соседнем models_check_cluster. Это именно точки текущего рисунка. 1943 K — другая серия, не опечатка в 1941 K: логи и thermostat задают 1943 K.',
'Копии pressure для 1873 и 2000 K побайтно совпадают с models_check_wsl2. Они не считаются дополнительными репликами. Все четыре опубликованные оценки воспроизведены до < 0.00002 mPa s. Проверены 20001 отсчёт, шаги 0..20000, отсутствие NaN и дубликатов, 512 атомов Ti, объёмы каждой реплики.',
'', 'STACIE и MD:',
'Шаг 0.001 ps; длина 20 ps; block_size=1; 5 ортогональных сдвиговых каналов; давления bar→Pa; V/(kB*T) отдельно для каждой реплики. Спектры усредняются, траектории не склеиваются во времени. Включена нулевая частота.',
'LorentzModel (3 параметра), стандартная маргинализация по cutoff, CV2LCriterion (fcut_factor=1.25); switch_exponent=8; fcut_spacing=0.5; neff_min=15; neff_max=1000; nonlinear_budget=100. Использована локальная копия исходников STACIE из прежнего аудита; совпадение старых чисел проверено напрямую.',
'MD: NVT Nose-Hoover, Tdamp=0.2 ps. Подготовка через NPT-резервуар 5 ps и 1 ps NVT на каждую стартовую конфигурацию, со сменой seed скоростей. Общий исходный резервуар ограничивает доказательство независимости: jackknife не покрывает неизвестную общую систематику подготовки или термостата.',
'', 'Основные числа (ошибки — 1 SE, mPa s):',
'T, N, total_ps, eta, formal_SE, cutoff_THz, neff, cost_Z, cutoff_Z, jackknife_replica_SE, jackknife_group5_SE']
for t,e in zip(temps,es):
    a=e['auto'];lines.append(f"{t}, {e['n_replicas']}, {e['n_replicas']*20}, {a['eta']:.6f}, {a['se']:.6f}, {a['fcut_THz']:.4f}, {a['neff']:.2f}, {a['cost_zscore']:.3f}, {a['criterion_zscore']:.3f}, {e['jackknife']['1']['auto_se']:.6f}, {e['jackknife']['5']['auto_se']:.6f}")
lines+=['',f"Arrhenius: eta [mPa s] = exp({fit['ln_eta0']:.9f} + {fit['B_K']:.6f}/T). chi2={fit['chi2']:.6f}, dof=2, p={fit['p']:.6f}. Наибольший стандартизованный остаток={max(abs(z) for z in fit['z']):.3f}. Это модель температурного тренда по 4 точкам, не новые независимые измерения, не свидетельство уникальности Arrhenius и не основание уменьшать исходные ошибки.",
'Взвешенная регрессия выполнена для ln(eta), sigma_ln≈sigma_eta/eta. Доверительная полоса использует исходные ошибки без уменьшения на sqrt(chi2/dof).',
'', 'Чувствительность к обработке:',
f"Усреднение давления блоками по 20 fs меняет eta максимум на {summary['max_relative_change_block20_percent']:.5f}%; более плотная сетка cutoff — максимум на {summary['max_relative_change_grid_percent']:.5f}%. Это не добавляет низкочастотной информации и не решает проблему гладкости.",
f"Общая сетка cutoff, где для всех 4 температур cost_Z<2 и cutoff_Z<2: {', '.join(f'{x:.2f}' for x in common)} THz. Сетка дискретная; это исследовательский диапазон, а не независимая валидация. neff>=60 здесь не обеспечено.",
'Для примера общий cutoff=1.25 THz (не рекомендуется автоматически заменять исходные значения):']
for t,e,a in zip(temps,es,fixed125):lines.append(f"{t} K: {a['eta']:.6f} +/- {a['se']:.6f}; сдвиг {100*(a['eta']/e['auto']['eta']-1):+.3f}%; jackknife реплик {e['jackknife']['1']['fixed']['1.25']['se']:.6f}.")
lines+=['Фиксация cutoff убирает компонент неопределённости его выбора; меньшая формальная ошибка не доказывает повышение точности. STACIE рекомендует маргинализацию. Все четыре оценки существенно коррелированы со своими альтернативными обработками.',
'Порог neff=60 является рекомендацией 20P для 3 параметров, а не количеством независимых реплик/кадров. На исходных 20 ps он недостигнут (25–34). Принудительное расширение окна повышает риск смещения модели. Например, 1873 K при cutoff=2 THz: cutoff_Z=5.55; при 3 THz neff≈62, cutoff_Z≈22.39. Формально меньшая ошибка здесь не преимущество.',
'', 'Окна времени и каналы (не независимые новые измерения):']
for t,e in zip(temps,es):
    lines.append(f'{t} K:')
    for label in ['first10','second10','discard1ps','offdiag3','noDC','neff20','neff30']:
        a=e['variants'][label];lines.append(f"  {label}: eta={a['eta']:.6f}, SE={a['se']:.6f}, cost_Z={a['cost_zscore']:.3f}, cutoff_Z={a['criterion_zscore']:.3f}")
    lines.append('  Модели: '+ '; '.join(f"{k}: {a['eta']:.6f} +/- {a['se']:.6f}, neff={a['neff']:.1f}" for k,a in m[str(t)]['models'].items()))
lines+=['Парное сравнение половин при едином cutoff=1.25 THz (jackknife целых реплик):']
for t in temps:
    a=w[str(t)]['paired_jackknife']['1.25'];lines.append(f"{t} K: вторая минус первая половина {a['second_minus_first']:.6f} +/- {a['se']:.6f}, Z={a['z']:.3f}.")
lines+=['Для 1941 K автоматический парный jackknife: разность 0.757785 +/- 0.347060 mPa s, Z=2.183. На фиксированном cutoff 1.25/1.5 THz Z=2.25/2.25. Это умеренный сигнал чувствительности к времени при нескольких проверенных температурах; он не доказывает дрейф и не разрешает выбор только более удобной половины. Удаление первых 1 ps меняет полную оценку 1941 K всего на +0.94%.',
'Деление на 10 ps снижает частотное разрешение и может ухудшить cutoff-диагностику. Различия половин нельзя автоматически интерпретировать как физический дрейф. Пять каналов сохраняются вместе при jackknife. Для групп по пять всего 3 группы при N=15 и 6 при N=30, поэтому такие ошибки сами неточны.',
'', 'Дополнительные старые серии SACF:',
'1773 K: 25 реплик, 500 ps суммарно. Интеграл при 2/3/5 ps = 4.826/4.884/4.851 mPa s; SE при 5 ps ~0.531. Сохранённый TDM-fit 4.862 согласуется с хвостом, но это не результат STACIE.',
'1943 K: 50 реплик, 1000 ps суммарно. Интеграл при 1/2/3/5 ps = 3.431/3.329/3.167/2.703 mPa s; хвост не образует устойчивого плато. Сохранённый TDM-fit 2.990 чувствителен к выбору хвоста (tau2 упирается в 1000 fs, alpha=1.86). Нельзя подставлять его вместо точки 1941 K. Для полноценного STACIE нужны pressure_* исходного расчёта, новый MD для этого не нужен, если файлы сохранились.',
'SACF содержит три диагональные автокорреляции сдвиговых компонент и лаги только до 5 ps. Из этого нельзя восстановить исходные пять каналов давления, выполнить честный ресэмплинг временных рядов и применить стандартную статистику Gamma к FFT обрезанной ACF.',
'', 'Практическая рекомендация:',
'1. Для рисунка сохранить исходные eta +/- SE; заменить соединение точек ломаной на отдельную гладкую Arrhenius-линию с указанием модели. Уже выполнен отдельный рисунок temperature_smooth.png/.svg.',
'2. Сохранить стандартную обработку STACIE и пять каналов. Не увеличивать neff_min или cutoff только ради гладкости; усреднение 5–20 fs проверено как диагностическая вариация, менять block_size=1 не требуется.',
'3. При необходимости отдельного единого cutoff-протокола заранее зафиксировать критерии спектральной пригодности, проверить все температуры и сообщать модельную чувствительность и jackknife, не выбирать окно по гладкости температурной кривой.',
'4. Для 1773/1943 K найти сохранившиеся pressure-файлы, если эти точки нужны для включения в общий STACIE-ряд. Текущий аудит не запускает MD и не изменяет статью.',
'', 'Источники методики:',
'https://molmod.github.io/stacie/theory/cutoff.html',
'https://molmod.github.io/stacie/preparing_inputs/data_sufficiency.html',
'https://molmod.github.io/stacie/properties/shear_viscosity.html',
'', 'Воспроизведение: analyze.py, supplement.py, model_check.py, windows.py, finalize.py. Полные результаты: results.json, model_check.json, supplement.json, windows.json; манифесты исходных pressure с SHA256 в results.json. Файлы input_*.npz — производные локальные копии каналов напряжения. Копия исходников библиотеки: source/stacie; контрольные суммы: stacie_source_hashes.json.']
(OUT/'report_RU.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
