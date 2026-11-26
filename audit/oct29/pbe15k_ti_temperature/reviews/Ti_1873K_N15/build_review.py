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
for T,n in [(1873,15),(2000,10),(2100,15)]:
 sid=f'Ti_{T}K';f=ROOT/f'collected/analysis/{sid}/batch_{n:04d}/summary.json';d=read(f)
 assert d==read(ROOT/f'collected/analysis/latest_{sid}.json')
 assert latest['points'][sid]['n_replicas']==n
 assert d['protocol_sha256']==sha(ROOT/'protocol.json') and d['checkpoint_sha256']==cfg['checkpoint']['sha256']
 assert n%5==0 and len(d['points'][0]['sources'])==n
 reports[sid]=d;sources[f.relative_to(ROOT).as_posix()]=sha(f);rows.append(d['points'][0])
d=reports['Ti_1873K'];row=d['points'][0];full=row['windows']['40ps']
prev=read(ROOT/'collected/analysis/Ti_1873K/batch_0010/summary.json')['points'][0]
base=read(ROOT/'baseline_5k.json');old=read(ROOT/'known_15k.json')
history=[read(ROOT/f'collected/analysis/Ti_1873K/batch_{n:04d}/summary.json')['points'][0] for n in [5,10,15]]
b1873=next(x for x in base if x['T']==1873);o1873=next(x for x in old if x['T']==1873)
comparisons={w:{'vs_N10_percent':100*(r['eta_mPa_s']/prev['windows'][w]['eta_mPa_s']-1),'SE_change_percent':100*(r['eta_std_mPa_s']/prev['windows'][w]['eta_std_mPa_s']-1),'vs_5k_percent':100*(r['eta_mPa_s']/b1873['eta_mPa_s']-1),'vs_historical15k_percent':100*(r['eta_mPa_s']/o1873['eta_mPa_s']-1)} for w,r in row['windows'].items()}
visual=['convergence.png']+[f'{w}/{kind}.png' for w in ['40ps','first20ps','second20ps','new_batch_40ps'] for kind in ['spectrum','extras']]
assessment=dict(reviewed_unix=time.time(),temperature_K=1873,n_replicas=15,summary_sha256=sources['collected/analysis/Ti_1873K/batch_0015/summary.json'],windows=row['windows'],comparisons=comparisons,new_batch_40ps=row['new_batch_40ps'],previous_replicas_40ps=row['previous_replicas_40ps'],new_batch_z=row['new_batch_z'],new_batch_vs_previous_percent=100*(row['new_batch_40ps']['eta_mPa_s']/row['previous_replicas_40ps']['eta_mPa_s']-1),checks=d['checks'],pass_streak=d['pass_streak'],stop_recommended=d['stop_recommended'],review_required=d['review_required'],visual_review={'files':visual,'notes':'All nine source figures visually inspected. Full40 and first20 cutoff weights concentrated near 1.6 THz; second20 has a broader low-frequency shoulder and main mode near 1.5 THz. New five replicas have a broader single weight peak near 1.7 THz. All four cost/cutoff summary Z values below 2; no selected high-frequency mode. Cumulative mean nearly unchanged from N10; split-window spread increases but is not evidence of physical drift. Crowded minor cutoff-axis labels in source extras remain readable; no source image was modified.'},runtime_verification='runtime-check.json: PASS; five new replica raw hashes, model/protocol identity, 512 Ti, 1873 K target, 40001 samples, 40 ps, one A100 verified; current 2000 K input checked.',campaign_points=rows,sources_sha256=sources,article_profile_sha256=sha(WORK/'audit/nov01/viscosity_article_combined/figure_profile.json'),caveats=['Formal estimator standard errors exclude model/protocol systematics.','Cumulative estimates and 20/40 ps windows are correlated; the convergence criteria are a heuristic.','Historical 15k and 5k references remain separate; agreement with 5k is not a stopping criterion.','The fixed article profile is unchanged; this figure shows only completed campaign batches and separate reference data.'])
(OUT/'assessment.json').write_text(json.dumps(assessment,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':11,'axes.titlesize':12,'legend.fontsize':9,'pdf.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(11.2,4.4));fig.subplots_adjust(left=.065,right=.985,bottom=.19,top=.86,wspace=.25)
def series(ax,data,x,**style):
 ax.errorbar([x(r) for r in data],[r['eta_mPa_s'] for r in data],yerr=[r['eta_std_mPa_s'] for r in data],capsize=3,lw=1.4,ms=5,**style)
series(axs[0],[dict(r['windows']['40ps'],T=r['temperature_K']) for r in rows],lambda r:r['T'],fmt='D-',color='#18745B',label='Новая серия 15k, 40 пс')
series(axs[0],[r for r in base if r['T'] in [1873,2000,2100]],lambda r:r['T'],fmt='s:',color='#777777',mfc='white',label='Исторические 5k, 20 пс')
series(axs[0],old,lambda r:r['T'],fmt='o--',color='#356CB0',mfc='white',label='Исторические 15k, 20 пс')
for r in rows:
 axs[0].annotate(f"N={r['n_replicas']}",(r['temperature_K'],r['windows']['40ps']['eta_mPa_s']),xytext=(7,-20),textcoords='offset points',color='#18745B',fontsize=9)
axs[0].set(title='Чистый Ti: текущие оценки',xlabel='Температура, K',ylabel='Вязкость, мПа·с',xlim=(1850,2145),ylim=(2.25,3.65),xticks=[1873,2000,2100])
axs[0].legend(loc='upper right',frameon=False)
for w,label,col in [('40ps','40 пс','#18745B'),('first20ps','Первые 20 пс','#C37817'),('second20ps','Вторые 20 пс','#5265AB')]:
 series(axs[1],[dict(r['windows'][w],N=r['n_replicas']) for r in history],lambda r:r['N'],fmt='o-',color=col,label=label)
axs[1].set(title='1873 K: накопление реплик',xlabel='Число реплик N',ylabel='Вязкость, мПа·с',xticks=[5,10,15],xlim=(4,16),ylim=(2.3,3.65))
axs[1].legend(loc='lower right',frameon=False)
for ax in axs:
 ax.grid(axis='y',alpha=.18);ax.spines[['top','right']].set_visible(False)
fig.suptitle('Мониторинг PBE 15000 · 4 ноября 2026 · предварительные результаты',fontsize=13,y=.985)
fig.text(.5,.045,'Показаны только полностью проанализированные пакеты. Ошибки: ±1 SE. Окна 20/40 пс коррелированы.',ha='center',fontsize=9,color='#444444')
fig.savefig(OUT/'campaign_monitor.png',dpi=180);fig.savefig(OUT/'campaign_monitor.svg');plt.close(fig)
text_lines=['Новый полный пакет: 1873 K, N=15. Все значения в мПа·с; ошибки — формальные SE.', '']
for r in rows:
 text_lines.append(f"{r['temperature_K']} K; N={r['n_replicas']}; "+'; '.join(f"{w}: {v['eta_mPa_s']:.6f} +/- {v['eta_std_mPa_s']:.6f}" for w,v in r['windows'].items()))
text_lines+=['',f"Изменение 40 пс относительно N=10: {comparisons['40ps']['vs_N10_percent']:.3f}%; изменение SE: {comparisons['40ps']['SE_change_percent']:.3f}%.",f"Новые пять: {row['new_batch_40ps']['eta_mPa_s']:.6f} +/- {row['new_batch_40ps']['eta_std_mPa_s']:.6f}; предыдущие десять: {row['previous_replicas_40ps']['eta_mPa_s']:.6f} +/- {row['previous_replicas_40ps']['eta_std_mPa_s']:.6f}; Z={row['new_batch_z']:.3f}.",f"Исторические 15k: {o1873['eta_mPa_s']:.6f} +/- {o1873['eta_std_mPa_s']:.6f}; отличие {comparisons['40ps']['vs_historical15k_percent']:.3f}%.",f"Исторические 5k: {b1873['eta_mPa_s']:.6f} +/- {b1873['eta_std_mPa_s']:.6f}; отличие {comparisons['40ps']['vs_5k_percent']:.3f}%.",'Не пройдены: N=15<30; SE/eta=3.096%>3%; размах N=5/10/15=2.794%>2%; max20/40=3.048%>3%. Серия последовательных прохождений: 0 из 2.','Пройдены: neff40=64.315>=60; cost и cutoff Z всех трёх окон и новых пяти <2; согласие нового пакета Z=0.353<=2.5.','Спектры и convergence визуально проверены. Различие половин само по себе не доказывает физический дрейф; окна и накопленные оценки коррелированы.','Официальные результаты не объединяются с историческими. Зафиксированный профиль статьи сохранён.']
(OUT/'assessment_RU.txt').write_text('\n'.join(text_lines)+'\n',encoding='utf-8')
print(json.dumps({'comparisons':comparisons,'new_batch_z':row['new_batch_z'],'monitor_figure':str(OUT/'campaign_monitor.png')},indent=2))
