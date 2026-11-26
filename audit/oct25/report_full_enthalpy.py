"""Figures, trajectory diagnostics and readable report for completed A100 series."""
from pathlib import Path
import json,sys,html
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];RUN=HERE/'pbe_full_enthalpy_gpu';OUT=RUN/'analysis'
sys.path.insert(0,str(ROOT/'audit/oct20/deps'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from ase.io.trajectory import Trajectory
from scipy.stats import linregress
d=json.loads((OUT/'statistics.json').read_text());w=d['windows'];p=w['production'];raw=np.load(OUT/'time_series.npz')
steps=['5000','15000','20000'];colors=['#555555','#1377b8','#d36b19'];labels=['5k','15k','20k']
names=sorted(p['groups']['5000'],key=lambda n:p['groups']['5000'][n]['x_Al']);x=np.array([p['groups']['5000'][n]['x_Al'] for n in names])
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
def save(fig,name):
 fig.savefig(OUT/(name+'.png'),bbox_inches='tight');fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def series(win,step,key):return [win['groups'][step][n]['rho'] if key=='rho' else win['mixing'][step][n][key] for n in names]
def sig(a):return a['ci95'][0]*a['ci95'][1]>0
fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for step,c,l in zip(steps,colors,labels):
 for a,key in zip(ax,['rho','H']):
  vals=series(p,step,key);a.errorbar(x,[v['mean'] for v in vals],yerr=[v['sem'] for v in vals],fmt='o-',color=c,label=l,capsize=3,ms=4,lw=1)
for a in ax:a.set_xlabel(r'$x_{\mathrm{Al}}$');a.grid(alpha=.2);a.legend(title='PBE checkpoint')
ax[0].set_ylabel(r'Density (g cm$^{-3}$)');ax[1].set_ylabel(r'$\Delta H_{mix}$ (kJ mol$^{-1}$)')
fig.suptitle('2000 K nominal; 30 ps production; error bars = block SEM')
save(fig,'density_enthalpy_5k_15k_20k')
fig,ax=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for pair,c,off in [('15000-5000',colors[1],-.008),('20000-5000',colors[2],.008)]:
 for a,key in zip(ax,['rho','H','U']):
  vals=[p['differences'][pair][n][key] for n in names];yy=np.array([v['mean'] for v in vals]);err=np.array([[v['mean']-v['ci95'][0] for v in vals],[v['ci95'][1]-v['mean'] for v in vals]])
  a.errorbar(x+off,yy,yerr=err,fmt='o',capsize=3,color=c,label=pair.replace('15000','15k').replace('20000','20k').replace('5000','5k'),ms=4)
for a in ax:a.axhline(0,color='k',lw=.7);a.set_xlabel(r'$x_{\mathrm{Al}}$');a.grid(alpha=.2);a.legend()
ax[0].set_ylabel(r'$\Delta\rho$ (g cm$^{-3}$)');ax[1].set_ylabel(r'$\Delta(\Delta H_{mix})$ (kJ mol$^{-1}$)');ax[2].set_ylabel(r'$\Delta(\Delta U_{mix})$ (kJ mol$^{-1}$)')
fig.suptitle('Checkpoint changes; pointwise 95% intervals (no multiplicity correction)')
save(fig,'checkpoint_differences_95CI')
fig,ax=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for a,step in zip(ax,steps[1:]):
 for win,label in [('production','Production 0-30 ps'),('production_guarded','Production 1-30 ps'),('pooled_5','Eq >5 + prod >1 ps'),('pooled_10','Eq >10 + prod >1 ps'),('pooled_15','Eq >15 + prod >1 ps'),('pooled_20','Eq >20 + prod >1 ps')]:
  yy=[w[win]['mixing'][step][n]['H']['mean']-p['mixing'][step][n]['H']['mean'] for n in names]
  a.plot(x,yy,'o-',ms=3,lw=1,label=label)
 a.set_title('PBE '+step);a.set_xlabel(r'$x_{\mathrm{Al}}$');a.set_ylabel('Change from production estimate (kJ/mol)');a.grid(alpha=.2);a.legend(fontsize=8)
fig.suptitle('Window sensitivity: same archived 5k baseline; overlapping windows are correlated')
save(fig,'equilibration_window_sensitivity')
# Per-state traces: 1 ps means; unshifted H is shown relative to production mean.
with PdfPages(OUT/'all_time_traces.pdf') as pdf:
 for name in names:
  fig,ax=plt.subplots(3,2,figsize=(11,8),layout='constrained')
  for j,step in enumerate(steps[1:]):
   for stage,offset in [('equilibration',0),('production',30)]:
    key=step+'|'+name+'|'+stage+'|';t=raw[key+'time']
    for i,k in enumerate(['rho','H','T']):
     vals=raw[key+k];tt=[];yy=[]
     for end in range(1,31):
      mask=(t>end-1+1e-8)&(t<=end+1e-8);tt.append(end-.5+offset);yy.append(vals[mask].mean())
     if k=='H':yy=np.array(yy)-p['groups'][step][name]['H']['mean']
     ax[i,j].plot(tt,yy,'o-',ms=3,color=colors[j+1],lw=1)
     ax[i,j].axvline(30,c='k',ls='--',lw=.8);ax[i,j].axvspan(0,10,color='gray',alpha=.1);ax[i,j].grid(alpha=.2)
     ax[i,j].set_ylabel({'rho':'Density (g/cm3)','H':'H - production mean (kJ/mol)','T':'ASE temperature (K)'}[k])
   ax[0,j].set_title('PBE '+labels[j+1]);ax[2,j].set_xlabel('Global time (ps)')
  fig.suptitle(name+': 1 ps block means; velocity/thermostat reset at 30 ps');pdf.savefig(fig)
  if name in ['Al0.3Ti0.7','Al0.6Ti0.4']:fig.savefig(OUT/('trace_'+name+'.png'),bbox_inches='tight')
  plt.close(fig)
# Liquid-state sanity check: time-origin averaged diffusion, affine cell change removed.
diff={}
tasks=json.loads((RUN/'tasks.json').read_text());cfg=json.loads((RUN/'protocol.json').read_text())
for task in tasks:
 step=str(task['checkpoint']);name=cfg['states'][task['state']]['composition'];traj=Trajectory(str(RUN/step/'results'/str(task['id'])/'production.traj'))
 atoms=list(traj);traj.close();frac=np.array([a.get_scaled_positions(wrap=True) for a in atoms]);cells=np.array([a.cell.array for a in atoms])
 delta=np.diff(frac,axis=0);delta-=np.round(delta);cart=np.einsum('tni,tij->tnj',delta,.5*(cells[1:]+cells[:-1]));cart-=cart.mean(axis=1,keepdims=True)
 unwrapped=np.concatenate([np.zeros_like(cart[:1]),np.cumsum(cart,axis=0)]);lags=np.arange(1,101);msd=np.array([np.mean(np.sum((unwrapped[lag:]-unwrapped[:-lag])**2,axis=-1)) for lag in lags]);fit=linregress(lags*.1,msd)
 diff[step+'|'+name]={'MSD_10ps_A2':float(msd[-1]),'D_fit_0p1_10ps_A2_ps':float(fit.slope/6),'r_squared':float(fit.rvalue**2),'note':'Pooled species; finite-size diagnostic, no diffusion error estimate; not a rigorous phase classification.'}
(OUT/'liquid_motion_qc.json').write_text(json.dumps(diff,indent=2))
# Additional coarse-sampling diagnostic: compare GPU log with saved-trajectory cadence.
coarse={}
for step in steps[1:]:
 for name in names:
  key=step+'|'+name+'|production|';t=raw[key+'time'];v=raw[key+'T'];mask=(t>0)&(np.abs(t*10-np.round(t*10))<1e-6)
  coarse[step+'|'+name]={'T_dense_K':float(v[1:].mean()),'T_100fs_K':float(v[mask].mean())}
(OUT/'temperature_sampling_qc.json').write_text(json.dumps(coarse,indent=2))
# Scientific checks on derived quantities and uncertainty implementation.
for win in w.values():
 for step in steps:
  for n in ['Ti','Al']:assert win['mixing'][step][n]['H']['mean']==0
 for name in names:
  for k in ['H','U','rho']:
   a=win['differences'];assert abs(a['20000-5000'][name][k]['mean']-a['15000-5000'][name][k]['mean']-a['20000-15000'][name][k]['mean'])<1e-9
for g in p['groups'].values():
 for state in g.values():
  for k in ['rho','H','U']:
   check=state[k]['block_checks']['5'];b=np.array(check['block_means']);assert abs(b.std(ddof=1)/len(b)**.5-check['sem'])<1e-9
lines=['PBE: сравнение 5k, 15k и 20k — завершённая серия A100',
'Все 22 расчёта (11 составов × 2 checkpoint) завершились без ошибок. Slurm 20960642, tesla-a101, две A100, 08:07:08. Скалярные логи и полные траектории обеих стадий скопированы и проверены по SHA256.',
'Протокол: 200 атомов, номинально 2000 K и 1 bar, шаг 1 fs; 30 ps уравновешивания + 30 ps production. Исходные координаты — конечные жидкие конфигурации старой серии 5k. Две стадии разделены повторной генерацией скоростей и перезапуском NPT. Скалярные логи: 10 fs; траектории: 100 fs. Для 5k доступны 30 ps production с шагом сохранения 100 fs.',
'ОСНОВНЫЕ ВЫВОДЫ',
'Плотность меняется слабо: максимальное отличие от 5k составляет 0.00960 g/cm3 (15k) и 0.01271 g/cm3 (20k), оба при x_Al=0.2; это 0.25% и 0.33%. Многие отдельные различия сопоставимы со статистической неопределённостью. Изменение энтальпии существенно: средняя часть кривой становится менее экзотермичной.',
'Для 15k отличие ΔHmix от 5k положительно и номинальный точечный 95%-интервал не включает ноль при x_Al=0.2–0.8. Для 20k это x_Al=0.2–0.5 и 0.7. Включение позднего уравновешивания после 10 ps сохраняет эти наборы точек. Это разведочное сравнение без поправки на множественные проверки; точки коррелированы через общие чистые компоненты.',
'Различия 20k–15k меньше и не разрешены для большинства составов. Наиболее заметно x_Al=0.6 (Ti0.4Al0.6): -1.064 kJ/mol, 95% CI [-1.928,-0.200]; с расширенным окном сдвиг -0.721 kJ/mol. Также устойчивы номинальные отличия при x_Al=0.8 и 0.9. Не следует объявлять 15k и 20k термодинамически эквивалентными или выбирать лучший только по этой серии без внешнего эталона.',
'СРЕДНИЕ ± SEM: только 30 ps production. x обозначает атомную долю Al; kJ/mol атомов.',
'x_Al | rho5k | rho15k | rho20k (g/cm3) | Hmix5k | Hmix15k | Hmix20k (kJ/mol)']
for name,xx in zip(names,x):
 vals=[]
 for key in ['rho','H']:
  for step in steps:
   a=p['groups'][step][name]['rho'] if key=='rho' else p['mixing'][step][name]['H'];digits=5 if key=='rho' else 3;vals.append(f"{a['mean']:.{digits}f} ± {a['sem']:.{digits}f}")
 lines.append(f'{xx:.1f} | '+' | '.join(vals))
lines+=['ДОБАВЛЕНИЕ УРАВНОВЕШИВАНИЯ',
'Разумное общее расширенное окно для проверки чувствительности: equilibration 10–30 ps плюс production 1–30 ps; итого 49 ps. Первые 10 ps и первый 1 ps после перезапуска не включаем. Это предложенный консервативный срез, а не доказанная для каждой траектории точная длительность релаксации. Отсутствие значимого расхождения не доказывает стационарность.',
'Позднее уравновешивание в целом согласуется с production; для окна 10–30 ps нет номинально значимых расхождений средних rho/H/U с production 1–30 ps. При этом для 15k Ti0.7Al0.3 последний 10 ps отрезок уравновешивания отличается по H/U; у 15k Ti0.2Al0.8 есть слабый тренд rho на 10–30 ps. Поэтому не предлагаю молча заменять основную 30 ps оценку расширенной: обе оценки сохранены, основной результат и его устойчивость видны отдельно.',
'При добавлении equilibration >10 ps средняя ΔHmix сдвигается максимум на 0.349 kJ/mol для 15k и 0.171 kJ/mol для 20k. Изменение rho не превышает 0.00217 g/cm3. Вывод о менее отрицательной энтальпии относительно 5k сохраняется. Окна после 5/10/15/20 ps приведены в statistics.json и на отдельном графике; они перекрываются, поэтому не являются независимыми оценками.',
'Признаки чувствительности к временному окну: в старой серии 5k меняется rho Ti0.7Al0.3 между половинами production и H/U Ti0.5Al0.5; для 20k меняется rho Ti0.9Al0.1. У 15k Al различие H между половинами также номинально значимо. Это поводы увеличить статистику, а не доказательство незавершённой релаксации: проверок много и длина записи ограничена.',
'НЕОПРЕДЕЛЁННОСТИ И ОПРЕДЕЛЕНИЕ ЭНТАЛЬПИИ',
'H = <(Epot + Ekin + P_external V)/N>; ΔHmix(x)=H(x)−(1−x)H(Ti)−xH(Al). Одинаковое определение применено ко всем трём моделям из исходных записей; старую формулу с противоположным знаком мгновенного давления не используем. Значения здесь могут отличаться от прежних таблиц с другим определением H. Разности потенциальной энергии смешения ΔUmix также сохранены и подтверждают физический сдвиг, а не только шум кинетической энергии.',
'SEM: максимум оценок по блокам 3, 5 и 6 ps, минимум четыре блока. Для коротких 10 ps QC-отрезков используются 2 ps блоки. Это практическая оценка ошибки среднего с учётом временной корреляции, не гарантия покрытия. Student/Welch 95%-интервалы рассчитаны из дисперсий блоков; для ΔHmix учтены ошибки смеси и обоих чистых компонентов. Блоки не пересекают перезапуск на 30 ps. При объединении стадий применены веса по длительности. В statistics.json есть блоки 1/2/3/5/6/10 ps и оценки автокорреляции.',
'Одна траектория на состояние: межрепликовый разброс и конечный размер 200 атомов не входят в SEM. Общие исходные координаты/семена для новых checkpoint могут создавать корреляции; распространение ошибок между траекториями выполнено в приближении независимости. Ошибки абсолютной энергии и плотности, обусловленные моделью, тоже не входят в SEM.',
'ДИАГНОСТИКА ТЕМПЕРАТУРЫ, ДАВЛЕНИЯ И ЖИДКОГО СОСТОЯНИЯ',
'ASE MelchionnaNPT задаёт Ekin_target=3(N−1)kBT/2, а get_temperature делит на 3N. Для N=200 ожидаемое значение в логе при номинальных 2000 K равно 1990 K: новые записи дают 1989.91–1990.13 K. Это соглашение о поступательных степенях свободы, а не охлаждение на 10 K. Старые редкие кадры 5k дают 1971.6–1995.0 K; вклад конечной выборки кинетической энергии в ΔHmix достигает 0.270 kJ/mol. Поэтому дополнительно сравнивается ΔUmix (при одинаковой заданной температуре кинетический вклад в ансамблевую энтальпию смешения сокращается).',
f"Контроль частоты сохранения: если новые GPU-логи проредить до 100 fs, средние температуры отдельных траекторий составят {min(v['T_100fs_K'] for v in coarse.values()):.1f}–{max(v['T_100fs_K'] for v in coarse.values()):.1f} K вместо около 1990 K при записи через 10 fs. Таким образом, редкая выборка сама может существенно смещать оценку среднего T; значения старых кадров нельзя напрямую интерпретировать как температуру всей траектории.",
'Среднее давление новых траекторий составляет около 0.005–0.010 GPa при заданных 0.0001 GPa; старые редкие кадры дают от -0.102 до +0.167 GPa. Значения P и их блочные ошибки сохранены. Мгновенное давление не подставляется вместо внешнего в H. Для окончательного прецизионного сравнения полезен повтор 5k с тем же частым логированием, а также отдельная проверка параметров интегрирования/баростата; нынешние SEM не учитывают такие систематические эффекты.',
f"Во всех 22 новых production-траекториях есть поступательная диффузия: MSD за лаг 10 ps = {min(v['MSD_10ps_A2'] for v in diff.values()):.1f}–{max(v['MSD_10ps_A2'] for v in diff.values()):.1f} Å². Удалено аффинное изменение ячейки и движение центра масс. Это проверка подвижности жидкости, не строгая структурная классификация и не публикационная оценка коэффициента диффузии.",
'РЕКОМЕНДАЦИЯ',
'Сохранить 30 ps production как основное сравнение и 49 ps как проверку чувствительности; позднее уравновешивание действительно полезно и не меняет основного вывода. Для окончательного выбора checkpoint приоритетны независимые продолжения/реплики Ti0.4Al0.6 для 15k и 20k и обновление статистики 5k (прежде всего Ti, Al и Ti0.5Al0.5, затем Ti0.7Al0.3). Чистые компоненты влияют сразу на всю кривую смешения. Данные поддерживают существенность смены 5k, но сами по себе не решают компромисс между 15k (DFT-DOS) и 20k (ошибка энергии на тесте). Новые MD автоматически не запускались; статья не изменялась.',
'ФАЙЛЫ',
'statistics.json — все средние, SEM, интервалы разностей, окна, QC; time_series.npz — численные временные ряды; all_time_traces.pdf — временные ряды всех составов; liquid_motion_qc.json — диффузионная диагностика. Все исходные файлы находятся в ../15000/results и ../20000/results; manifest/collection JSON в корне серии содержат SHA256. Для воспроизведения: analyze_full_enthalpy.py, затем report_full_enthalpy.py.']
text='\n\n'.join(lines);(OUT/'REPORT_RU.txt').write_text(text,encoding='utf-8')
imgs=''.join(f'<figure><img src="{f}.png" style="width:100%"><figcaption><a href="{f}.pdf">PDF</a></figcaption></figure>' for f in ['density_enthalpy_5k_15k_20k','checkpoint_differences_95CI','equilibration_window_sensitivity'])
body=''.join('<p>'+html.escape(line)+'</p>' if ' | ' not in line else '<pre>'+html.escape(line)+'</pre>' for line in lines)
(OUT/'REPORT.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><title>PBE MD: 5k/15k/20k</title><style>body{font:16px/1.6 system-ui;max-width:1200px;margin:35px auto;padding:20px}pre{font:12px monospace;white-space:pre-wrap}figure{margin:25px 0}a{color:#1377b8}</style><h1>PBE: плотность и энтальпия смешения</h1>'+imgs+'<p><a href="all_time_traces.pdf">Временные ряды всех составов</a> · <a href="statistics.json">Полная статистика JSON</a> · <a href="REPORT_RU.txt">Текст отчёта</a></p>'+body+'</html>',encoding='utf-8')
print('Saved reports and figures:',OUT)
print('Diffusion MSD range',min(v['MSD_10ps_A2'] for v in diff.values()),max(v['MSD_10ps_A2'] for v in diff.values()))
print('Coarse temperature range',min(v['T_100fs_K'] for v in coarse.values()),max(v['T_100fs_K'] for v in coarse.values()))
