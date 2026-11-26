from pathlib import Path
import json,time,hashlib,sys,os
W=Path(__file__).resolve().parents[5];M=W/'audit/oct29/pbe15k_ti_temperature';S=W/'audit/nov05/pbe15k_ti_1941'
sys.path.insert(0,str(W/'audit/oct27/viscosity_5k_vs_15k/python_deps'));os.environ.setdefault('MPLCONFIGDIR',str(M/'cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=read(M/'baseline_5k.json');old=read(M/'known_15k.json');wins=['40ps','first20ps','second20ps'];colors=['#18745B','#C37817','#5265AB'];labels=['40 ps','First 20 ps','Second 20 ps']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':11,'legend.fontsize':9,'svg.fonttype':'none'})
CAVEATS=['Formal SE excludes model, protocol and cutoff systematics.','Accumulated estimates and windows20/40ps are correlated; stopping is a heuristic.','Historical15k and5k remain separate; agreement with5k is not a stop criterion.','Fixed article profile and the earlier preview snapshot remain unchanged.']
NOTES={2100:'All nine diagnostic figures inspected. Main cutoff weights: full40 near1.5THz,first20 near1.7THz,second20 near1.5THz,new5 near1.8THz with a modest upper shoulder. No dominant high-frequency mode is visible. Full40 cutoffZ3.805 andfirst20Z2.882 fail;second20 andnew5 pass. Allcost pass. Persistent cutoff failure is not physical drift and is not repaired by substituting a post-hoc common-cutoff fit.',1941:'All nine diagnostic figures inspected. Full40 main cutoff near2THz, first20 near2.5THz,second20 near1.7THz,new5 near2THz. Second20 has a broad upper tail;new5 has a smaller high-frequency mode near7THz in addition to its main mode. All weighted cost/cutoff diagnostics pass now; source plot is not sufficient evidence of physical drift. N5 previous fit had cutoff failures; comparison is descriptive.'}
rows=[]
for T,N in [(1873,30),(2000,30),(2100,35)]:
 f=M/f'collected/analysis/Ti_{T}K/batch_{N:04d}/summary.json';d=read(f);assert d==read(M/f'collected/analysis/latest_Ti_{T}K.json');assert read(M/'collected/analysis/latest.json')['points'][f'Ti_{T}K']['n_replicas']==N;rows.append(d['points'][0])
for root,T,N,hns in [(M,2100,35,[25,30,35]),(S,1941,10,[5,10])]:
 out=root/f'reviews/Ti_{T}K_N{N}';batch=root/f'collected/analysis/Ti_{T}K/batch_{N:04d}';d=read(batch/'summary.json');r=d['points'][0];rt=read(out/'runtime-check.json')
 assert d==read(root/f'collected/analysis/latest_Ti_{T}K.json') and read(root/'collected/analysis/latest.json')['points'][f'Ti_{T}K']['n_replicas']==N
 assert d['protocol_sha256']==sha(root/'protocol.json') and d['checkpoint_sha256']==read(root/'protocol.json')['checkpoint']['sha256'] and len(r['sources'])==N
 assert all(rt['checks'].values()) and len(rt['new_batch_replicas'])==5 and all(x['checks']=='PASS' for x in rt['new_batch_replicas'])
 hist=[read(root/f'collected/analysis/Ti_{T}K/batch_{n:04d}/summary.json')['points'][0] for n in hns];prev=hist[-2];b=next(x for x in base if x['T']==T)
 comparisons={k:dict(vs_previous_percent=100*(v['eta_mPa_s']/prev['windows'][k]['eta_mPa_s']-1),SE_change_percent=100*(v['eta_std_mPa_s']/prev['windows'][k]['eta_std_mPa_s']-1),vs_5k_percent=100*(v['eta_mPa_s']/b['eta_mPa_s']-1),vs_historical15k_percent=None) for k,v in r['windows'].items()}
 pics=['convergence.png']+[f'{k}/{kind}.png' for k in wins+['new_batch_40ps'] for kind in ['spectrum','extras']]
 assessment=dict(reviewed_unix=time.time(),temperature_K=T,n_replicas=N,summary_sha256=sha(batch/'summary.json'),windows=r['windows'],comparisons=comparisons,new_batch_40ps=r['new_batch_40ps'],previous_replicas_40ps=r['previous_replicas_40ps'],new_batch_z=r['new_batch_z'],checks=d['checks'],pass_streak=d['pass_streak'],stop_recommended=d['stop_recommended'],review_required=d['review_required'],visual_review=dict(files_sha256={n:sha(batch/n) for n in pics},notes=NOTES[T]),runtime_verification='PASS:five new replica raw hashes/512Ti/targetT/40001 samples/40ps/model/protocol/GPU/own volume prefactor',campaign_points=rows if T==2100 else [r],article_profile_sha256=sha(W/'audit/nov01/viscosity_article_combined/figure_profile.json'),caveats=CAVEATS)
 (out/'assessment.json').write_text(json.dumps(assessment,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 full=r['windows']['40ps'];nb=r['new_batch_40ps'];pr=r['previous_replicas_40ps'];text=[f'{T} K N={N}: новый полный пакет. Все ошибки ниже — формальные SE.','']
 for k,v in r['windows'].items():text.append(f"{k}: {v['eta_mPa_s']:.6f} +/- {v['eta_std_mPa_s']:.6f} мПа·с; neff={v['neff']:.3f}; costZ={v['cost_z']:.3f}; cutoffZ={v['criterion_z']:.3f}.")
 text.extend([f"Прежние{N-5}: {pr['eta_mPa_s']:.6f}+/-{pr['eta_std_mPa_s']:.6f}; изменение40 {comparisons['40ps']['vs_previous_percent']:+.4f}%; изменениеSE {comparisons['40ps']['SE_change_percent']:+.3f}%.",f"Новые5: {nb['eta_mPa_s']:.6f}+/-{nb['eta_std_mPa_s']:.6f}; против прежнихZ={r['new_batch_z']:.3f}; costZ={nb['cost_z']:.3f}; cutoffZ={nb['criterion_z']:.3f}.",f"Отдельные5k: {b['eta_mPa_s']:.6f}+/-{b['eta_std_mPa_s']:.6f}; отличие {comparisons['40ps']['vs_5k_percent']:+.3f}%. Исторического15k при этойT нет.",json.dumps(d['checks'],ensure_ascii=False),f"Полных прохождений {d['pass_streak']}/2; stop={d['stop_recommended']}; review hold={d['review_required']}.",NOTES[T],*CAVEATS])
 if T==2100:text.append('Не проходит только cutoff полного40ps иfirst20ps (и общий флаг всех окон). N35,SE1.8668%,размах1.1565%,20/40=1.9676%,neff63.2667,newbatchZ0.4925 проходят. ПриN>=50 устойчивой статистике и плохой диагностике — reviewhold; сейчасN35.')
 else:text.append('Не проходит N10<30,SE3.4023%>3%,max20/40=5.5853%>3%. Для последних3 точек пока есть лишь2; размах ещё не оценивается. neff78.3323,всеcost/cutoff,newbatchZ0.6790 проходят.')
 (out/'assessment_RU.txt').write_text('\n'.join(text)+'\n',encoding='utf-8')
 if T==2100:
  fig,axs=plt.subplots(1,2,figsize=(11.2,4.5));fig.subplots_adjust(left=.075,right=.98,bottom=.23,top=.86,wspace=.27);a=axs[0]
  a.errorbar([x['temperature_K'] for x in rows],[x['windows']['40ps']['eta_mPa_s'] for x in rows],yerr=[x['windows']['40ps']['eta_std_mPa_s'] for x in rows],fmt='D-',color=colors[0],capsize=3,label='New PBE 15k, 40 ps')
  for dat,fmt,col,label in [(base,'s:','#777777','Historical 5k, 20 ps'),(old,'o--','#356CB0','Historical 15k, 20 ps')]:
   rr=[v for v in dat if v['T'] in [1873,2000,2100]];a.errorbar([v['T'] for v in rr],[v['eta_mPa_s'] for v in rr],yerr=[v['eta_std_mPa_s'] for v in rr],fmt=fmt,color=col,mfc='white',capsize=3,label=label)
  for x in rows:
   v=x['windows']['40ps'];a.annotate(f"N={x['n_replicas']}",(x['temperature_K'],v['eta_mPa_s']),xytext=(6,-19),textcoords='offset points',color=colors[0],fontsize=9)
   if v['cost_z']>=2 or v['criterion_z']>=2:a.plot(x['temperature_K'],v['eta_mPa_s'],'D',mfc='white',color=colors[0],zorder=5)
  a.set(title='Pure Ti: completed campaign batches',xlabel='Temperature (K)',ylabel='Viscosity (mPa s)',xticks=[1873,2000,2100],xlim=(1850,2140),ylim=(2.25,3.7));a.legend(frameon=False,loc='upper right');a=axs[1]
  a.set(title='2100 K: cutoff checks still fail',xlabel='Number of replicas N',ylabel='Viscosity (mPa s)',xticks=hns,xlim=(24,36),ylim=(2.32,2.95))
  legendloc='upper right';foot='Hollow symbols: cost and/or cutoff check failed. No temperature has completed the stop rule.'
 else:
  fig,a=plt.subplots(figsize=(8,4.7));fig.subplots_adjust(left=.12,right=.98,bottom=.24,top=.85);axs=[a]
  a.axhspan(b['eta_mPa_s']-b['eta_std_mPa_s'],b['eta_mPa_s']+b['eta_std_mPa_s'],color='#999999',alpha=.12)
  a.axhline(b['eta_mPa_s'],color='#777777',linestyle='--',label='Separate historical 5k, 15 x 20 ps')
  a.set(title='New 1941 K series: N=10',xlabel='Number of new replicas N',ylabel='Viscosity (mPa s)',xticks=hns,xlim=(4.3,10.7),ylim=(2.35,3.55));legendloc='upper right';foot='All current spectral checks pass; N, precision and 20/40 ps agreement still fail.'
 for k,label,col in zip(wins,labels,colors):
  a.errorbar(hns,[x['windows'][k]['eta_mPa_s'] for x in hist],yerr=[x['windows'][k]['eta_std_mPa_s'] for x in hist],fmt='o-',color=col,capsize=3,label=label)
  for n,x in zip(hns,hist):
   v=x['windows'][k]
   if v['cost_z']>=2 or v['criterion_z']>=2:a.plot(n,v['eta_mPa_s'],'o',mfc='white',color=col,zorder=5)
 a.errorbar(N+.12,nb['eta_mPa_s'],yerr=nb['eta_std_mPa_s'],fmt='D',color='#B45050',capsize=3,label='New five only, 40 ps');a.legend(frameon=False,loc=legendloc)
 for a in axs:a.grid(axis='y',alpha=.18);a.spines[['top','right']].set_visible(False)
 fig.suptitle('PBE checkpoint 15000 | 5 November 2026',fontsize=13,y=.98)
 fig.text(.5,.10,'Complete analyzed batches only. Error bars: +/-1 SE. The 20/40 ps windows are correlated.',ha='center',fontsize=9,color='#444444')
 fig.text(.5,.045,foot,ha='center',fontsize=8.5,color='#884444')
 for ext in ['png','svg']:fig.savefig(out/f'campaign_monitor.{ext}',dpi=180)
 plt.close(fig)
 print(T,N,json.dumps(comparisons['40ps']), 'new5Z',r['new_batch_z'])
