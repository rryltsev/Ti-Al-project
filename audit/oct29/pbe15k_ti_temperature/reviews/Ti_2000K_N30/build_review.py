from pathlib import Path
import json,time,hashlib,sys,os
W=Path(__file__).resolve().parents[5]
M=W/'audit/oct29/pbe15k_ti_temperature';S=W/'audit/nov05/pbe15k_ti_1941'
sys.path.insert(0,str(W/'audit/oct27/viscosity_5k_vs_15k/python_deps'))
os.environ.setdefault('MPLCONFIGDIR',str(M/'cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=read(M/'baseline_5k.json');old=read(M/'known_15k.json')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':11,'legend.fontsize':9,'svg.fonttype':'none'})
colors=['#18745B','#C37817','#5265AB'];windows=['40ps','first20ps','second20ps'];labels=['40 ps','First 20 ps','Second 20 ps']
CAVEATS=['Formal estimator SE; model, protocol and cutoff bias are not included.','Windows and accumulated estimates are correlated; these stopping criteria are a heuristic.','5k and historical 15k references remain separate; agreement with 5k is not a stop criterion.','Fixed article profile is unchanged; only complete analyzed batches are plotted.']
notes={2000:'All nine diagnostics visually inspected: full40 cutoff weight near1.5THz; first20 near1.7THz; second20 near1.5THz; new5 near1.8THz, unimodal main weight and broad new5 uncertainty. All weighted cost/cutoff Z<2. No physical drift inferred. Minor-axis labels overlap in source extras but modes and diagnostics readable.',1941:'All seven diagnostics visually inspected. Full40 main cutoff near2.8THz and a small lower-frequency shoulder; cutoff Z2.432 fails. First20 main cutoff near2.8THz passes. Second20 main cutoff near3.5THz with secondary high-frequency weight around7THz; cost Z2.777 and cutoff Z3.262 fail. This preliminary spectral fit does not establish physical drift. No previous accumulated estimate/new5-vs-old contrast at N5.'}
rows=[]
for T,N in [(1873,30),(2000,30),(2100,30)]:
 f=M/f'collected/analysis/Ti_{T}K/batch_{N:04d}/summary.json';d=read(f)
 assert d==read(M/f'collected/analysis/latest_Ti_{T}K.json')
 assert read(M/'collected/analysis/latest.json')['points'][f'Ti_{T}K']['n_replicas']==N
 rows.append(d['points'][0])
for root,T,N in [(M,2000,30),(S,1941,5)]:
 out=root/'reviews'/f'Ti_{T}K_N{N}';out.mkdir(parents=True,exist_ok=True)
 batch=root/f'collected/analysis/Ti_{T}K/batch_{N:04d}';f=batch/'summary.json';d=read(f);r=d['points'][0]
 assert d==read(root/f'collected/analysis/latest_Ti_{T}K.json')
 assert read(root/'collected/analysis/latest.json')['points'][f'Ti_{T}K']['n_replicas']==N
 assert d['protocol_sha256']==sha(root/'protocol.json') and d['checkpoint_sha256']==read(root/'protocol.json')['checkpoint']['sha256']
 assert len(r['sources'])==N and all(read(out/'runtime-check.json')['checks'].values())
 b=next(v for v in base if v['T']==T);o=next((v for v in old if v['T']==T),None)
 prev=read(root/f'collected/analysis/Ti_{T}K/batch_{N-5:04d}/summary.json')['points'][0] if N>5 else None
 cmp={k:{'vs_5k_percent':100*(v['eta_mPa_s']/b['eta_mPa_s']-1),'vs_historical15k_percent':100*(v['eta_mPa_s']/o['eta_mPa_s']-1) if o else None,'vs_previous_percent':100*(v['eta_mPa_s']/prev['windows'][k]['eta_mPa_s']-1) if prev else None,'SE_change_percent':100*(v['eta_std_mPa_s']/prev['windows'][k]['eta_std_mPa_s']-1) if prev else None} for k,v in r['windows'].items()}
 pics=['convergence.png']+[f'{k}/{kind}.png' for k in windows+(['new_batch_40ps'] if N>5 else []) for kind in ['spectrum','extras']]
 assessment=dict(reviewed_unix=time.time(),temperature_K=T,n_replicas=N,summary_sha256=sha(f),windows=r['windows'],comparisons=cmp,new_batch_40ps=r.get('new_batch_40ps'),previous_replicas_40ps=r.get('previous_replicas_40ps'),new_batch_z=r.get('new_batch_z'),checks=d['checks'],pass_streak=d['pass_streak'],stop_recommended=d['stop_recommended'],review_required=d['review_required'],visual_review=dict(notes=notes[T],files_sha256={n:sha(batch/n) for n in pics}),runtime_verification='runtime-check.json: PASS for all five new replicas and all staged/model/GPU/target/normalization checks.',article_profile_sha256=sha(W/'audit/nov01/viscosity_article_combined/figure_profile.json'),caveats=CAVEATS)
 (out/'assessment.json').write_text(json.dumps(assessment,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 text=[f'{T} K, N={N}. Новый полностью проанализированный пакет. Ошибки ниже: формальные SE.','']
 for k,v in r['windows'].items():text.append(f"{k}: {v['eta_mPa_s']:.6f} +/- {v['eta_std_mPa_s']:.6f} мПа·с; neff={v['neff']:.3f}; cost Z={v['cost_z']:.3f}; cutoff Z={v['criterion_z']:.3f}.")
 if prev:
  nb=r['new_batch_40ps'];pr=r['previous_replicas_40ps'];text.extend([f"Предыдущие N={N-5}: {pr['eta_mPa_s']:.6f} +/- {pr['eta_std_mPa_s']:.6f}. Изменение среднего40: {cmp['40ps']['vs_previous_percent']:+.3f}%; SE: {cmp['40ps']['SE_change_percent']:+.3f}%.",f"Новые5: {nb['eta_mPa_s']:.6f} +/- {nb['eta_std_mPa_s']:.6f}; Z против прежних={r['new_batch_z']:.3f}."])
 else:text.append('Это первые 5 новых реплик: предыдущей накопленной оценки и сравнения новых5 с прежними пока нет; эти критерии ещё не оцениваются.')
 text.append(f"5k отдельно: {b['eta_mPa_s']:.6f} +/- {b['eta_std_mPa_s']:.6f}; разница40 {cmp['40ps']['vs_5k_percent']:+.3f}%.")
 if o:text.append(f"Исторические15k отдельно: {o['eta_mPa_s']:.6f} +/- {o['eta_std_mPa_s']:.6f}; разница40 {cmp['40ps']['vs_historical15k_percent']:+.3f}%.")
 else:text.append('Исторического результата15k при1941K до этого запуска нет; старые5k не переименованы.')
 text.extend([json.dumps(d['checks'],ensure_ascii=False),f"Последовательных полных прохождений {d['pass_streak']}/2; stop={d['stop_recommended']}; review hold={d['review_required']}.",notes[T],*CAVEATS])
 (out/'assessment_RU.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
 if T==2000:
  fig,axs=plt.subplots(1,2,figsize=(11.2,4.5));fig.subplots_adjust(left=.075,right=.98,bottom=.23,top=.86,wspace=.27)
  a=axs[0]
  a.errorbar([x['temperature_K'] for x in rows],[x['windows']['40ps']['eta_mPa_s'] for x in rows],yerr=[x['windows']['40ps']['eta_std_mPa_s'] for x in rows],fmt='D-',color=colors[0],capsize=3,label='New PBE 15k, 40 ps')
  for data,fmt,col,label in [(base,'s:','#777777','Historical 5k, 20 ps'),(old,'o--','#356CB0','Historical 15k, 20 ps')]:
   rr=[v for v in data if v['T'] in [1873,2000,2100]];a.errorbar([v['T'] for v in rr],[v['eta_mPa_s'] for v in rr],yerr=[v['eta_std_mPa_s'] for v in rr],fmt=fmt,color=col,mfc='white',capsize=3,label=label)
  for x in rows:
   v=x['windows']['40ps'];a.annotate('N=30',(x['temperature_K'],v['eta_mPa_s']),xytext=(6,-19),textcoords='offset points',color=colors[0],fontsize=9)
   if v['cost_z']>=2 or v['criterion_z']>=2:a.plot(x['temperature_K'],v['eta_mPa_s'],'D',color=colors[0],mfc='white',zorder=5)
  a.set(title='Pure Ti: completed campaign batches',xlabel='Temperature (K)',ylabel='Viscosity (mPa s)',xticks=[1873,2000,2100],xlim=(1850,2140),ylim=(2.25,3.7));a.legend(frameon=False,loc='upper right')
  hist=[read(M/f'collected/analysis/Ti_2000K/batch_{n:04d}/summary.json')['points'][0] for n in [20,25,30]];a=axs[1]
  for k,label,col in zip(windows,labels,colors):a.errorbar([x['n_replicas'] for x in hist],[x['windows'][k]['eta_mPa_s'] for x in hist],yerr=[x['windows'][k]['eta_std_mPa_s'] for x in hist],fmt='o-',color=col,capsize=3,label=label)
  nb=r['new_batch_40ps'];a.errorbar(30,nb['eta_mPa_s'],yerr=nb['eta_std_mPa_s'],fmt='D',color='#B45050',capsize=3,label='New five only, 40 ps')
  a.set(title='2000 K: first complete pass (1/2)',xlabel='Number of replicas N',ylabel='Viscosity (mPa s)',xticks=[20,25,30],xlim=(19,31),ylim=(2.58,3.12));a.legend(frameon=False,loc='upper right')
  foot='Hollow campaign symbols: cutoff check failed. Other temperatures still need further batches.'
 else:
  fig,a=plt.subplots(figsize=(7.8,4.6));fig.subplots_adjust(left=.12,right=.98,bottom=.25,top=.85);axs=[a]
  a.axhspan(b['eta_mPa_s']-b['eta_std_mPa_s'],b['eta_mPa_s']+b['eta_std_mPa_s'],color='#999999',alpha=.15)
  a.axhline(b['eta_mPa_s'],color='#777777',linestyle='--',label='Historical 5k, 15 x 20 ps (separate)')
  for x,k,label,col in zip([0,1,2],windows,labels,colors):
   v=r['windows'][k];bad=v['cost_z']>=2 or v['criterion_z']>=2
   a.errorbar(x,v['eta_mPa_s'],yerr=v['eta_std_mPa_s'],fmt='o',color=col,mfc='white' if bad else col,capsize=4,ms=7)
   a.annotate(f"{v['eta_mPa_s']:.3f} +/- {v['eta_std_mPa_s']:.3f}",(x,v['eta_mPa_s']-v['eta_std_mPa_s']-.065),ha='center',color=col)
  a.set(title='New 1941 K series: first five replicas',ylabel='Viscosity (mPa s)',xticks=[0,1,2],xticklabels=labels,xlim=(-.55,2.55),ylim=(2.35,3.4));a.legend(frameon=False,loc='upper left')
  foot='Hollow symbols: cost and/or cutoff check failed. Preliminary result; no stop criteria passed in full.'
 for a in axs:a.grid(axis='y',alpha=.18);a.spines[['top','right']].set_visible(False)
 fig.suptitle('PBE checkpoint 15000 | 5 November 2026',fontsize=13,y=.98)
 fig.text(.5,.10,'Complete analyzed batches only. Error bars: +/-1 SE. The 20/40 ps windows are correlated.',ha='center',fontsize=9,color='#444444')
 fig.text(.5,.045,foot,ha='center',fontsize=8.5,color='#884444')
 for ext in ['png','svg']:fig.savefig(out/f'campaign_monitor.{ext}',dpi=180)
 plt.close(fig)
 print(T,N,json.dumps({'comparisons':cmp['40ps'],'checks':d['checks'],'pass_streak':d['pass_streak']}))
