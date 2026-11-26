from pathlib import Path
import json,hashlib,time,sys,os
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT.parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(WORK/'audit/oct27/viscosity_5k_vs_15k/python_deps'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfg=read(ROOT/'protocol.json');latest=read(ROOT/'collected/analysis/latest.json');rows=[];sources={}
for T,n in [(1873,20),(2000,20),(2100,25)]:
 sid=f'Ti_{T}K';f=ROOT/f'collected/analysis/{sid}/batch_{n:04d}/summary.json';d=read(f)
 assert d==read(ROOT/f'collected/analysis/latest_{sid}.json') and latest['points'][sid]['n_replicas']==n
 assert d['protocol_sha256']==sha(ROOT/'protocol.json') and d['checkpoint_sha256']==cfg['checkpoint']['sha256'] and len(d['points'][0]['sources'])==n
 sources[f.relative_to(ROOT).as_posix()]=sha(f);rows.append(d['points'][0])
row=rows[-1];full=row['windows']['40ps'];previous=read(ROOT/'collected/analysis/Ti_2100K/batch_0020/summary.json')['points'][0]
history=[read(ROOT/f'collected/analysis/Ti_2100K/batch_{n:04d}/summary.json')['points'][0] for n in [15,20,25]]
base=read(ROOT/'baseline_5k.json');old=read(ROOT/'known_15k.json');b=next(r for r in base if r['T']==2100)
diagdir=ROOT/'diagnostics/Ti_2100K_N25_cutoff';diag=read(diagdir/'diagnostic_results.json');assert diag.get('finished_unix')
comparisons={w:dict(vs_N20_percent=100*(v['eta_mPa_s']/previous['windows'][w]['eta_mPa_s']-1),SE_change_percent=100*(v['eta_std_mPa_s']/previous['windows'][w]['eta_std_mPa_s']-1),vs_5k_percent=100*(v['eta_mPa_s']/b['eta_mPa_s']-1)) for w,v in row['windows'].items()}
thermo={w:sum(x['window_means'][w]['Temp'] for x in diag['thermo'])/25 for w in ['first20ps','second20ps']}
assessment=dict(reviewed_unix=time.time(),temperature_K=2100,n_replicas=25,windows=row['windows'],comparisons=comparisons,new_batch_40ps=row['new_batch_40ps'],previous_replicas_40ps=row['previous_replicas_40ps'],new_batch_z=row['new_batch_z'],new_batch_vs_previous_percent=100*(row['new_batch_40ps']['eta_mPa_s']/previous['windows']['40ps']['eta_mPa_s']-1),checks=d['checks'],pass_streak=d['pass_streak'],stop_recommended=d['stop_recommended'],review_required=d['review_required'],campaign_points=rows,sources_sha256=sources,historical15k_at2100=None,visual_review={'files':['convergence.png']+[f'{w}/{kind}.png' for w in ['40ps','first20ps','second20ps','new_batch_40ps'] for kind in ['spectrum','extras']],'notes':'All nine source figures inspected. Full40 cutoff weight sharply concentrated near1.5THz but weighted cutoffZ3.533 fails; first20 mode near1.6THz also fails atZ2.520. Second20 mode near1.6THz passes. New five main mode near1.8THz with small secondary mode near6.7THz; its cost/cutoff diagnostics pass and weight above5THz is3.60%. New5 agrees with previous20, but previous20 cutoff failure limits formal Z interpretation. No evidence of physical drift established. Source image minor-axis labels crowded but readable; originals unchanged.'},common_cutoff_1p5=diag['fixed_1p5'],paired_jackknife_1p5=diag['paired_jackknife'],weight_above5THz={w:x['weight_above5THz'] for w,x in diag['automatic'].items()},mean_temperature_K=thermo,runtime_verification='runtime-check.json PASS: one A100, checkpoint/protocol/raw hashes, 512 Ti, target2100K,newreplicas21-25,40001 samples,40ps,V/(kBT).',article_profile_sha256=sha(WORK/'audit/nov01/viscosity_article_combined/figure_profile.json'),caveats=['Formal SE excludes model/protocol bias and cutoff selection uncertainty.','Cumulative and 20/40 ps estimates are correlated.','Fixed cutoff is a sensitivity test, not a replacement for official results.','Historical5k remains separate; agreement with it is not a stop criterion.','No historical15k2100K result exists.'])
(OUT/'assessment.json').write_text(json.dumps(assessment,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':11,'axes.titlesize':12,'legend.fontsize':9})
fig,axs=plt.subplots(1,2,figsize=(11.2,4.6));fig.subplots_adjust(left=.065,right=.985,bottom=.22,top=.86,wspace=.25)
def series(ax,data,x,**style):
 ax.errorbar([x(r) for r in data],[r['eta_mPa_s'] for r in data],yerr=[r['eta_std_mPa_s'] for r in data],capsize=3,lw=1.3,ms=5,**style)
new=[dict(r['windows']['40ps'],T=r['temperature_K'],N=r['n_replicas']) for r in rows]
series(axs[0],new,lambda r:r['T'],fmt='D-',color='#18745B',label='Новая серия 15k, 40 пс')
series(axs[0],[r for r in base if r['T'] in [1873,2000,2100]],lambda r:r['T'],fmt='s:',color='#777777',mfc='white',label='Исторические 5k, 20 пс')
series(axs[0],old,lambda r:r['T'],fmt='o--',color='#356CB0',mfc='white',label='Исторические 15k, 20 пс')
for r in new:
 axs[0].annotate(f"N={r['N']}",(r['T'],r['eta_mPa_s']),xytext=(7,-20),textcoords='offset points',color='#18745B',fontsize=9)
 if r['cost_z']>=2 or r['criterion_z']>=2:axs[0].plot(r['T'],r['eta_mPa_s'],'D',mfc='white',mec='#18745B',ms=5,zorder=5)
axs[0].set(title='Чистый Ti: текущие оценки',xlabel='Температура, K',ylabel='Вязкость, мПа·с',xlim=(1850,2145),ylim=(2.25,3.65),xticks=[1873,2000,2100]);axs[0].legend(loc='upper right',frameon=False)
for w,label,col in [('40ps','40 пс','#18745B'),('first20ps','Первые 20 пс','#C37817'),('second20ps','Вторые 20 пс','#5265AB')]:
 vals=[dict(r['windows'][w],N=r['n_replicas']) for r in history];series(axs[1],vals,lambda r:r['N'],fmt='o-',color=col,label=label)
 for r in vals:
  if r['cost_z']>=2 or r['criterion_z']>=2:axs[1].plot(r['N'],r['eta_mPa_s'],'o',mfc='white',mec=col,ms=5,zorder=5)
axs[1].set(title='2100 K: последние три пакета',xlabel='Число реплик N',ylabel='Вязкость, мПа·с',xticks=[15,20,25],xlim=(14,26),ylim=(2.05,2.95));axs[1].legend(loc='upper right',frameon=False)
for ax in axs:ax.grid(axis='y',alpha=.18);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Мониторинг PBE 15000 · 5 ноября 2026 · предварительные результаты',fontsize=13,y=.985)
fig.text(.5,.08,'Только полностью проанализированные пакеты. Ошибки: ±1 SE. Окна 20/40 пс коррелированы.',ha='center',fontsize=9,color='#444444')
fig.text(.5,.035,'Пустые маркеры новых результатов: не пройдена проверка cutoff/cost; оценки требуют осторожности.',ha='center',fontsize=9,color='#884444')
fig.savefig(OUT/'campaign_monitor.png',dpi=180);fig.savefig(OUT/'campaign_monitor.svg');plt.close(fig)
colors={'40ps':'#18745B','first20ps':'#C37817','second20ps':'#5265AB','new_batch_40ps':'#B45050'};labels={'40ps':'Все 25, 40 пс','first20ps':'Первые 20 пс','second20ps':'Вторые 20 пс','new_batch_40ps':'Новые 5, 40 пс'}
fig,axs=plt.subplots(1,2,figsize=(10.8,4.2),layout='constrained')
for w,a in diag['automatic'].items():
 h=a['history'];col=colors[w];x=[v['fcut']/1e12 for v in h]
 axs[0].plot(x,[v['fcut_weight'] for v in h],color=col,label=labels[w])
 axs[1].plot(x,[v['acint']*1000 for v in h],color=col,label=labels[w])
 f=diag['fixed_1p5'][w];axs[1].errorbar(1.5,f['acint']*1000,yerr=f['acint_std']*1000,fmt='o',color=col,capsize=3)
for ax in axs:ax.set_xscale('log');ax.set_xlim(.65,9);ax.set_xlabel('Cutoff, ТГц');ax.grid(alpha=.15)
axs[0].set(title='Автоматические веса cutoff',ylabel='Вес');axs[0].legend(frameon=False,fontsize=8)
axs[1].set(title='Чувствительность оценки к cutoff',ylabel='Вязкость, мПа·с',ylim=(2.1,3.15));axs[1].axvline(1.5,color='gray',ls=':',lw=1)
fig.suptitle('2100 K, N=25 · отдельная диагностика; точки: общий cutoff 1,5 ТГц ±1 SE',fontsize=11)
fig.savefig(diagdir/'cutoff_diagnostic.png',dpi=160);plt.close(fig)

lines=['2100 K, PBE15000, N=25. Все ошибки — формальные SE; официальный протокол неизменен.','']
for r in rows:lines.append(f"{r['temperature_K']} K N={r['n_replicas']}: "+'; '.join(f"{w}={v['eta_mPa_s']:.6f} +/- {v['eta_std_mPa_s']:.6f}" for w,v in r['windows'].items()))
lines+=['',f"N20->25: среднее40 {comparisons['40ps']['vs_N20_percent']:+.3f}%; SE {comparisons['40ps']['SE_change_percent']:+.3f}%.",f"Новые5: {row['new_batch_40ps']['eta_mPa_s']:.6f} +/- {row['new_batch_40ps']['eta_std_mPa_s']:.6f}; прежние20: {previous['windows']['40ps']['eta_mPa_s']:.6f} +/- {previous['windows']['40ps']['eta_std_mPa_s']:.6f}; формальное Z={row['new_batch_z']:.3f}, но cutoff прежних20 не проходит.",f"5k: {b['eta_mPa_s']:.6f} +/- {b['eta_std_mPa_s']:.6f}; отличие {comparisons['40ps']['vs_5k_percent']:+.3f}%. Исторического15k2100K нет.",'Проходят: ошибка2.2603%<=3%; размахN15/20/25=1.9958%<=2% (почти порог); max20/40=2.8994%<=3%; neff40=62.228>=60; cost всех окон и нового пакета<2; новый пакет Z=0.288<=2.5.','Не проходят: N25<30; cutoffZ40=3.533,first20=2.520. Second20=-0.156 и new5=-0.209 проходят. Серия прохождений0/2.','', 'Общий cutoff1.5ТГц — отдельная диагностика:']
for w,v in diag['fixed_1p5'].items():lines.append(f"{w}: eta={v['acint']*1000:.6f} +/- {v['acint_std']*1000:.6f}, costZ={v['cost_zscore']:.3f}, cutoffZ={v['criterion_zscore']:.3f}")
pair=diag['paired_jackknife'];lines+=[f"Парная разность first20-second20={pair['difference_mPa_s']:.6f} +/- {pair['standard_error_mPa_s']:.6f} мПа·с, jackknife по целым репликам.",f"Вес автоматических cutoff выше5ТГц у новых5={diag['automatic']['new_batch_40ps']['weight_above5THz']*100:.3f}%.",f"Средние температуры половин всех25: {thermo}.",'Общий cutoff1.5 почти не меняет полную40пс оценку, но cutoffZ3.217 всё ещё не проходит. Он не заменяет официальный результат и критерии остановки.','Доказательств физического дрейфа нет. Окна и накопленные оценки коррелированы; формальные ошибки не включают систематику модели и выбора cutoff.','Профиль статьи сохранён; расчёт штатно продолжает 1873K на одной A100.']
for f in [OUT/'assessment_RU.txt',diagdir/'assessment_RU.txt']:f.write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'comparisons':comparisons,'thermo':thermo,'checks':d['checks'],'new_batch_z':row['new_batch_z'],'new_batch_vs_previous_percent':assessment['new_batch_vs_previous_percent']},indent=2))
