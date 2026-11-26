from pathlib import Path
import json,hashlib,time,sys,os
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT.parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(WORK/'audit/oct27/viscosity_5k_vs_15k/python_deps'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfg=read(ROOT/'protocol.json'); latest=read(ROOT/'collected/analysis/latest.json')
reports={};sources={};rows=[]
for T,n in [(1873,20),(2000,20),(2100,20)]:
 sid=f'Ti_{T}K';f=ROOT/f'collected/analysis/{sid}/batch_{n:04d}/summary.json';d=read(f)
 assert d==read(ROOT/f'collected/analysis/latest_{sid}.json')
 assert latest['points'][sid]['n_replicas']==n
 assert d['protocol_sha256']==sha(ROOT/'protocol.json') and d['checkpoint_sha256']==cfg['checkpoint']['sha256']
 assert n%5==0 and len(d['points'][0]['sources'])==n
 reports[sid]=d;sources[f.relative_to(ROOT).as_posix()]=sha(f);rows.append(d['points'][0])
d=reports['Ti_2000K'];row=d['points'][0];full=row['windows']['40ps']
prev=read(ROOT/'collected/analysis/Ti_2000K/batch_0015/summary.json')['points'][0]
base=read(ROOT/'baseline_5k.json');old=read(ROOT/'known_15k.json')
history=[read(ROOT/f'collected/analysis/Ti_2000K/batch_{n:04d}/summary.json')['points'][0] for n in [10,15,20]]
b2000=next(x for x in base if x['T']==2000);o2000=next(x for x in old if x['T']==2000)
comparisons={w:{'vs_N15_percent':100*(r['eta_mPa_s']/prev['windows'][w]['eta_mPa_s']-1),'SE_change_percent':100*(r['eta_std_mPa_s']/prev['windows'][w]['eta_std_mPa_s']-1),'vs_5k_percent':100*(r['eta_mPa_s']/b2000['eta_mPa_s']-1),'vs_historical15k_percent':100*(r['eta_mPa_s']/o2000['eta_mPa_s']-1)} for w,r in row['windows'].items()}
visual=['convergence.png']+[f'{w}/{kind}.png' for w in ['40ps','first20ps','second20ps','new_batch_40ps'] for kind in ['spectrum','extras']]
assessment=dict(reviewed_unix=time.time(),temperature_K=2000,n_replicas=20,summary_sha256=sources['collected/analysis/Ti_2000K/batch_0020/summary.json'],windows=row['windows'],comparisons=comparisons,new_batch_40ps=row['new_batch_40ps'],previous_replicas_40ps=row['previous_replicas_40ps'],new_batch_z=row['new_batch_z'],new_batch_vs_previous_percent=100*(row['new_batch_40ps']['eta_mPa_s']/row['previous_replicas_40ps']['eta_mPa_s']-1),checks=d['checks'],pass_streak=d['pass_streak'],stop_recommended=d['stop_recommended'],review_required=d['review_required'],visual_review={'files':visual,'notes':'All nine source figures visually inspected. Cumulative full40 cutoff weight peaks near 1.5 THz, first20 near 1.7 THz, second20 near 1.6 THz. New five main cutoff mode lies near 2.4 THz; a tiny higher-frequency secondary peak is visible but not dominant. All cost/cutoff Z values pass. The lower new5 estimate yields Z2.997 against previous15 and causes failure of batch consistency and last-three stability. No physical drift inferred; no source image modified.'},runtime_verification='runtime-check.json: PASS; five new replica raw hashes, model/protocol identity, 512 Ti, 2000 K target, 40001 samples, 40 ps, one A100 verified; current 2100 K input checked.',campaign_points=rows,sources_sha256=sources,article_profile_sha256=sha(WORK/'audit/nov01/viscosity_article_combined/figure_profile.json'),caveats=['Formal estimator standard errors exclude model/protocol systematics.','Cumulative estimates and 20/40 ps windows are correlated; the convergence criteria are a heuristic.','Historical 15k and 5k references remain separate; agreement with 5k is not a stopping criterion.','The fixed article profile is unchanged; this figure shows only completed campaign batches and separate reference data.'])
runtime=read(OUT/'runtime-check.json');assessment['thermo_group_means']=runtime['thermo_group_means'];assessment['thermo_volume_change_percent']=100*(runtime['thermo_group_means']['new5']['40ps']['Volume']/runtime['thermo_group_means']['previous15']['40ps']['Volume']-1)
(OUT/'assessment.json').write_text(json.dumps(assessment,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':11,'axes.titlesize':12,'legend.fontsize':9,'pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(11.2,4.4));fig.subplots_adjust(left=.065,right=.985,bottom=.23,top=.86,wspace=.25)
def series(ax,data,x,**style):
 ax.errorbar([x(r) for r in data],[r['eta_mPa_s'] for r in data],yerr=[r['eta_std_mPa_s'] for r in data],capsize=3,lw=1.4,ms=5,**style)
series(axs[0],[dict(r['windows']['40ps'],T=r['temperature_K']) for r in rows],lambda r:r['T'],fmt='D-',color='#18745B',label='Новая серия 15k, 40 пс')
series(axs[0],[r for r in base if r['T'] in [1873,2000,2100]],lambda r:r['T'],fmt='s:',color='#777777',mfc='white',label='Исторические 5k, 20 пс')
series(axs[0],old,lambda r:r['T'],fmt='o--',color='#356CB0',mfc='white',label='Исторические 15k, 20 пс')
for r in rows:
 axs[0].annotate(f"N={r['n_replicas']}",(r['temperature_K'],r['windows']['40ps']['eta_mPa_s']),xytext=(7,-20),textcoords='offset points',color='#18745B',fontsize=9)
for r in rows:
 w=r['windows']['40ps']
 if w['cost_z']>=2 or w['criterion_z']>=2:axs[0].plot(r['temperature_K'],w['eta_mPa_s'],'D',mfc='white',mec='#18745B',ms=5,zorder=5)
axs[0].set(title='Чистый Ti: текущие оценки',xlabel='Температура, K',ylabel='Вязкость, мПа·с',xlim=(1850,2145),ylim=(2.25,3.65),xticks=[1873,2000,2100])
axs[0].legend(loc='upper right',frameon=False)
for w,label,col in [('40ps','40 пс','#18745B'),('first20ps','Первые 20 пс','#C37817'),('second20ps','Вторые 20 пс','#5265AB')]:
 series(axs[1],[dict(r['windows'][w],N=r['n_replicas']) for r in history],lambda r:r['N'],fmt='o-',color=col,label=label)
nb=row['new_batch_40ps'];axs[1].errorbar(20,nb['eta_mPa_s'],yerr=nb['eta_std_mPa_s'],fmt='D',color='#B45050',ms=5,capsize=3,label='Только новые 5, 40 пс')
axs[1].text(.97,.40,f"Новые 5 / прежние 15: Z={row['new_batch_z']:.2f}",transform=axs[1].transAxes,ha='right',fontsize=9,color='#B45050')
axs[1].set(title='2000 K: последние три пакета',xlabel='Число реплик N',ylabel='Вязкость, мПа·с',xticks=[10,15,20],xlim=(9,21),ylim=(2.25,3.25))
axs[1].legend(loc='lower left',frameon=False)
for ax in axs:
 ax.grid(axis='y',alpha=.18);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Мониторинг PBE 15000 · 5 ноября 2026 · предварительные результаты',fontsize=13,y=.985)
fig.text(.5,.08,'Показаны только полностью проанализированные пакеты. Ошибки: ±1 SE. Окна 20/40 пс коррелированы.',ha='center',fontsize=9,color='#444444')
fig.text(.5,.035,'Пустой маркер при 2100 K: проверка cutoff не пройдена; оценка предварительная.',ha='center',fontsize=9,color='#884444')
fig.savefig(OUT/'campaign_monitor.png',dpi=180);fig.savefig(OUT/'campaign_monitor.svg');plt.close(fig)


text_lines=['2000 K, N=20: полный анализ нового пакета. Все значения в мПа·с; ошибки — формальные SE.','']
for r in rows:text_lines.append(f"{r['temperature_K']} K, N={r['n_replicas']}: "+'; '.join(f"{w}: {v['eta_mPa_s']:.6f} +/- {v['eta_std_mPa_s']:.6f}" for w,v in r['windows'].items()))
text_lines+=['',f"Относительно N15: среднее40 {comparisons['40ps']['vs_N15_percent']:+.3f}%; SE {comparisons['40ps']['SE_change_percent']:+.3f}%.",f"Новые5: {row['new_batch_40ps']['eta_mPa_s']:.6f} +/- {row['new_batch_40ps']['eta_std_mPa_s']:.6f}; прежние15: {row['previous_replicas_40ps']['eta_mPa_s']:.6f} +/- {row['previous_replicas_40ps']['eta_std_mPa_s']:.6f}; Z={row['new_batch_z']:.3f}.",f"Исторические15k: {o2000['eta_mPa_s']:.6f} +/- {o2000['eta_std_mPa_s']:.6f}; отличие {comparisons['40ps']['vs_historical15k_percent']:+.3f}%.",f"Исторические5k: {b2000['eta_mPa_s']:.6f} +/- {b2000['eta_std_mPa_s']:.6f}; отличие {comparisons['40ps']['vs_5k_percent']:+.3f}%.",'Не проходят: N20<30; размахN10/15/20=3.7776%>2%; новый пакет Z=2.9965>2.5. Последовательных прохождений0/2.','Проходят: ошибка40=2.8225%<=3%; max20/40=1.1803%<=3%; neff40=60.480>=60; cost/cutoff всех трёх окон и новых5<2.','Cutoff Z: 40ps1.644,first200.310,second201.431,new51.517. Все9исходных графиков просмотрены; основная мода новых5 около2.4ТГц, не доминирующая высокочастотная мода.','Дополнительная проверка журналов2000K: все20реплик имеют401термодинамическую запись от0до40000шагов. СредняяT прежних15=1998.246K,новых5=1997.345K. СреднийV=9558.311/9598.035A^3 (+0.4156%). Среднее давление NVT=+147.168/-2849.661bar. РазбросV между репликами сохранён, в нормировке использован собственныйV каждой реплики. Причинная связь расхождения вязкости сV/давлением не установлена.','Расхождение нового пакета требует дальнейшего контроля, но не доказывает физический дрейф. Накопленные оценки и окна коррелированы; формальные ошибки не включают систематику модели и протокола.','Исторические ряды сохранены отдельно. Совпадение с5k не критерий остановки. Фиксированный профиль статьи сохранён.','Job20962153 RUNNING на одной A100. Все20исходных2000K проверены по хешам,512Ti,целевойT,40001отсчёту,40пс и V/(kBT).']
(OUT/'assessment_RU.txt').write_text('\n'.join(text_lines)+'\n',encoding='utf-8')
print(json.dumps({'comparisons':comparisons,'new_batch_z':row['new_batch_z'],'new_batch_vs_previous_percent':assessment['new_batch_vs_previous_percent'],'checks':d['checks'],'monitor_figure':str(OUT/'campaign_monitor.png')},indent=2))
