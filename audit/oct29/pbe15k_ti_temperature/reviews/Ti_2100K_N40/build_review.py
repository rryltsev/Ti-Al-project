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
NOTES={2100: 'All9source diagnostic figures inspected. Full40 cutoff weightnear1.5THz failsZ3.542;first20near1.7THz failsZ2.916;second20near1.5THz passes. New5 main cutoffnear3.4THz with smaller modesnear1.3and7THz, failscost2.535andcutoff4.255. New5Z0.014doesnotvalidate this failedfit. Fullneff59.740fails. No physicaldrift conclusion;keep officialresults unchanged and do not replace with posthocfits.', 1941: 'All9source diagnostic figures inspected. Full40 maincutoffnear1.8THz withlower shouldernear1.1THz failsZ3.649;first20near2.2THzpasses;second20near1.8THz failsZ2.453. New5 distinctlybimodal cutoffweightnear1and2THz failsZ2.589;costpasses. New5Z1.355 andhigher mean are descriptive because thefitdiagnosticsfail. This doesnotestablishphysicaldrift;allofficialpointsretained.'}
rows=[]
for T,N in [(1873,35),(2000,35),(2100,40)]:
 f=M/f'collected/analysis/Ti_{T}K/batch_{N:04d}/summary.json';d=read(f);assert d==read(M/f'collected/analysis/latest_Ti_{T}K.json');assert read(M/'collected/analysis/latest.json')['points'][f'Ti_{T}K']['n_replicas']==N;rows.append(d['points'][0])
for root,T,N,hns in [(M,2100,40,[30,35,40]),(S,1941,25,[15,20,25])]:
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
 if T==2100:text.append('Не проходятneff59.7403<60,cutoff40Z3.5425,first20Z2.9157,new5cost2.5348/cutoff4.2548. SE1.7431%,размах0.5779%,20/40=1.2210%,новыйZ0.0137формальнопроходят. Новыйпакетfitплохой;Zнеподтверждаетнадёжность. Полныхпрохождений0/2. ПриN>=50устойчивойстатистикеи плохойдиагностике—reviewhold;сейчасN40.')
 else:text.append('Не проходятN25<30,размах2.4367%>2%,20/40=6.0353%>3%,cutoff40Z3.6488/second20Z2.4533/new5Z2.5888. SE2.5145%,neff70.8619,всеcost,новыйZ1.3549формальнопроходят;неинтерпретироватьZкакподтверждениеприbadfit. Полныхпрохождений0/2.')
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
  a.set(title='2100 K: cutoff checks still fail',xlabel='Number of replicas N',ylabel='Viscosity (mPa s)',xticks=hns,xlim=(29,41),ylim=(2.32,2.95))
  legendloc='upper right';foot='Hollow symbols: cost and/or cutoff check failed. 2000 K is complete; 1873 K and 2100 K continue.'
 else:
  fig,a=plt.subplots(figsize=(8,4.7));fig.subplots_adjust(left=.12,right=.98,bottom=.24,top=.85);axs=[a]
  a.axhspan(b['eta_mPa_s']-b['eta_std_mPa_s'],b['eta_mPa_s']+b['eta_std_mPa_s'],color='#999999',alpha=.12)
  a.axhline(b['eta_mPa_s'],color='#777777',linestyle='--',label='Separate historical 5k, 15 x 20 ps')
  a.set(title='New 1941 K series: N=25',xlabel='Number of new replicas N',ylabel='Viscosity (mPa s)',xticks=hns,xlim=(14,26),ylim=(2.35,3.7));legendloc='upper left';foot='N, stability, 20/40 ps and cutoff checks fail; the new-five fit is also preliminary.'
 for k,label,col in zip(wins,labels,colors):
  a.errorbar(hns,[x['windows'][k]['eta_mPa_s'] for x in hist],yerr=[x['windows'][k]['eta_std_mPa_s'] for x in hist],fmt='o-',color=col,capsize=3,label=label)
  for n,x in zip(hns,hist):
   v=x['windows'][k]
   if v['cost_z']>=2 or v['criterion_z']>=2:a.plot(n,v['eta_mPa_s'],'o',mfc='white',color=col,zorder=5)
 a.errorbar(N+.12,nb['eta_mPa_s'],yerr=nb['eta_std_mPa_s'],fmt='D',color='#B45050',mfc='white' if nb['cost_z']>=2 or nb['criterion_z']>=2 else '#B45050',capsize=3,label='New five only, 40 ps');a.legend(frameon=False,loc=legendloc)
 for a in axs:a.grid(axis='y',alpha=.18);a.spines[['top','right']].set_visible(False)
 fig.suptitle('PBE checkpoint 15000 | 6 November 2026',fontsize=13,y=.98)
 fig.text(.5,.10,'Complete analyzed batches only. Error bars: +/-1 SE. The 20/40 ps windows are correlated.',ha='center',fontsize=9,color='#444444')
 fig.text(.5,.045,foot,ha='center',fontsize=8.5,color='#884444')
 for ext in ['png','svg']:fig.savefig(out/f'campaign_monitor.{ext}',dpi=180)
 plt.close(fig)
 print(T,N,json.dumps(comparisons['40ps']), 'new5Z',r['new_batch_z'])
