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
for T,n in [(1873,25),(2000,25),(2100,25)]:
 sid=f'Ti_{T}K';f=ROOT/f'collected/analysis/{sid}/batch_{n:04d}/summary.json';d=read(f)
 assert d==read(ROOT/f'collected/analysis/latest_{sid}.json')
 assert latest['points'][sid]['n_replicas']==n
 assert d['protocol_sha256']==sha(ROOT/'protocol.json') and d['checkpoint_sha256']==cfg['checkpoint']['sha256']
 assert n%5==0 and len(d['points'][0]['sources'])==n
 reports[sid]=d;sources[f.relative_to(ROOT).as_posix()]=sha(f);rows.append(d['points'][0])
d=reports['Ti_2000K'];row=d['points'][0];full=row['windows']['40ps']
prev=read(ROOT/'collected/analysis/Ti_2000K/batch_0020/summary.json')['points'][0]
base=read(ROOT/'baseline_5k.json');old=read(ROOT/'known_15k.json')
history=[read(ROOT/f'collected/analysis/Ti_2000K/batch_{n:04d}/summary.json')['points'][0] for n in [15,20,25]]
b2000=next(x for x in base if x['T']==2000);o2000=next(x for x in old if x['T']==2000)
comparisons={w:{'vs_N20_percent':100*(r['eta_mPa_s']/prev['windows'][w]['eta_mPa_s']-1),'SE_change_percent':100*(r['eta_std_mPa_s']/prev['windows'][w]['eta_std_mPa_s']-1),'vs_5k_percent':100*(r['eta_mPa_s']/b2000['eta_mPa_s']-1),'vs_historical15k_percent':100*(r['eta_mPa_s']/o2000['eta_mPa_s']-1)} for w,r in row['windows'].items()}
visual=['convergence.png']+[f'{w}/{kind}.png' for w in ['40ps','first20ps','second20ps','new_batch_40ps'] for kind in ['spectrum','extras']]
assessment=dict(reviewed_unix=time.time(),temperature_K=2000,n_replicas=25,summary_sha256=sources['collected/analysis/Ti_2000K/batch_0025/summary.json'],windows=row['windows'],comparisons=comparisons,new_batch_40ps=row['new_batch_40ps'],previous_replicas_40ps=row['previous_replicas_40ps'],new_batch_z=row['new_batch_z'],new_batch_vs_previous_percent=100*(row['new_batch_40ps']['eta_mPa_s']/row['previous_replicas_40ps']['eta_mPa_s']-1),checks=d['checks'],pass_streak=d['pass_streak'],stop_recommended=d['stop_recommended'],review_required=d['review_required'],visual_review={'files':visual,'notes':'All nine source figures inspected. Full40 cutoff mode near1.5THz, first20 near1.7THz, second20 near1.5THz, newest5 near2.1THz. All four summary cost/cutoff Z below2, no dominant high-frequency mode. New5 agrees with previous20 atZ0.590; previous batch discrepancy Z2.997 remains historical evidence and is not erased. Last-three stability still fails. No physical drift established; source minor labels crowded but readable and originals unchanged.'},runtime_verification='runtime-check.json: PASS; five new replica raw hashes, model/protocol identity, 512 Ti, 2000 K target, 40001 samples, 40 ps, one A100 verified; current 2100 K input checked.',campaign_points=rows,sources_sha256=sources,article_profile_sha256=sha(WORK/'audit/nov01/viscosity_article_combined/figure_profile.json'),caveats=['Formal estimator standard errors exclude model/protocol systematics.','Cumulative estimates and 20/40 ps windows are correlated; the convergence criteria are a heuristic.','Historical 15k and 5k references remain separate; agreement with 5k is not a stopping criterion.','The fixed article profile is unchanged; this figure shows only completed campaign batches and separate reference data.'])
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
axs[1].set(title='2000 K: последние три пакета',xlabel='Число реплик N',ylabel='Вязкость, мПа·с',xticks=[15,20,25],xlim=(14,26),ylim=(2.5,3.25))
axs[1].legend(loc='lower left',frameon=False)
for ax in axs:
 ax.grid(axis='y',alpha=.18);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Мониторинг PBE 15000 · 5 ноября 2026 · предварительные результаты',fontsize=13,y=.985)
fig.text(.5,.08,'Показаны только полностью проанализированные пакеты. Ошибки: ±1 SE. Окна 20/40 пс коррелированы.',ha='center',fontsize=9,color='#444444')
fig.text(.5,.035,'Пустой маркер при 2100 K: проверка cutoff не пройдена; оценка предварительная.',ha='center',fontsize=9,color='#884444')
fig.savefig(OUT/'campaign_monitor.png',dpi=180);fig.savefig(OUT/'campaign_monitor.svg');plt.close(fig)



text_lines=['2000 K, N=25: полный анализ нового пакета. Все значения мПа·с; ошибки формальные SE.','']
for r in rows:text_lines.append(f"{r['temperature_K']} K N={r['n_replicas']}: "+'; '.join(f"{w}: {v['eta_mPa_s']:.6f} +/- {v['eta_std_mPa_s']:.6f}" for w,v in r['windows'].items()))
text_lines+=['',f"N20->25: среднее40 {comparisons['40ps']['vs_N20_percent']:+.3f}%; SE {comparisons['40ps']['SE_change_percent']:+.3f}%.",f"Новые5: {row['new_batch_40ps']['eta_mPa_s']:.6f} +/- {row['new_batch_40ps']['eta_std_mPa_s']:.6f}; прежние20: {row['previous_replicas_40ps']['eta_mPa_s']:.6f} +/- {row['previous_replicas_40ps']['eta_std_mPa_s']:.6f}; Z={row['new_batch_z']:.3f}.",f"Исторические15k: {o2000['eta_mPa_s']:.6f} +/- {o2000['eta_std_mPa_s']:.6f}; отличие {comparisons['40ps']['vs_historical15k_percent']:+.3f}%.",f"Исторические5k: {b2000['eta_mPa_s']:.6f} +/- {b2000['eta_std_mPa_s']:.6f}; отличие {comparisons['40ps']['vs_5k_percent']:+.3f}%.",'Не проходят: N25<30; размахN15/20/25=4.2247%>2%. Серия полных прохождений0/2.','Проходят: ошибка2.3722%; max20/40=1.9119%; neff40=63.187; cost/cutoff всех окон и новых5<2; Zнового пакета0.590<=2.5.','Cutoff Z: full40=1.736,first20=0.253,second20=1.722,new5=0.111. Спектры и convergence визуально проверены.','Прошлый пакет16-20 имел Z2.997 против первых15; это сохраняется в истории. Новый пакет21-25 согласуется с накопленными20, но не устраняет автоматически проблему стабильности. Физический дрейф не установлен.','Окна и накопленные оценки коррелированы. Формальные ошибки не включают систематику модели и протокола, правило остановки эвристическое. Совпадение с5k не критерий остановки.','Хеши новых5исходных реплик,512Ti,2000K,40001отсчёт,40пс,V/(kBT),однаA100 проверены. Профиль статьи сохранён; расчёт продолжается2100K.']
(OUT/'assessment_RU.txt').write_text('\n'.join(text_lines)+'\n',encoding='utf-8');print(json.dumps({'comparisons':comparisons,'new_batch_z':row['new_batch_z'],'checks':d['checks']},indent=2))
