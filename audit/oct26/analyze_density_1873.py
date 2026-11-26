"""Audit completed 1873 K MD and make checkpoint and manuscript figure versions."""
from pathlib import Path
import csv, hashlib, json, os, sys
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
RUN=HERE/'pbe15k_density_1873'; RAW=RUN/'collected'; OUT=RUN/'analysis'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'audit/oct20/deps'))
os.environ['MPLBACKEND']='Agg'
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as student, linregress
from ase.io.trajectory import Trajectory
import ase.md.melchionna

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
save=lambda name,d:(OUT/name).write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
cfg=load(RUN/'protocol.json'); tasks=load(RUN/'tasks.json'); baseline=load(RUN/'baseline_5k.json')
pre=load(RAW/'15000/preflight.json'); completed=load(RAW/'15000/summary.json')
assert completed['status']=='completed' and completed['completed']==list(range(11)) and not completed['failed']
assert pre['status']=='passed' and 'A100' in pre['device']['name']
assert pre['model_sha256']==cfg['models']['15000']['sha256'] and pre['protocol_sha256']==sha(RUN/'protocol.json')
old_summary=load(RAW/'density_summary.json')
series={}; results={'5000':{},'15000':{}}; qc={}; motion={}

def stats(a,dt):
    a=np.asarray(a,dtype=float); assert np.isfinite(a).all()
    blocks={}
    for ps in [1,2,3,5,6,10]:
        n=int(round(ps/dt))
        if len(a)%n or len(a)//n<2: continue
        b=a.reshape(-1,n).mean(axis=1)
        blocks[str(ps)]={'sem':float(b.std(ddof=1)/np.sqrt(len(b))),
                         'blocks':len(b),'means':b.tolist()}
    primary={k:v for k,v in blocks.items() if k in ['3','5','6'] and v['blocks']>=4}
    if not primary: primary={k:v for k,v in blocks.items() if k=='2' and v['blocks']>=4}
    key=max(primary,key=lambda k:primary[k]['sem']); se=primary[key]['sem']; df=primary[key]['blocks']-1
    mean=float(a.mean()); half=float(student.ppf(.975,df)*se)
    return {'mean':mean,'sem':se,'df':df,'ci95':[mean-half,mean+half],
            'selected_block_ps':int(key),'block_checks':blocks,'samples':len(a)}

def difference(a,b):
    mean=a['mean']-b['mean']; var=a['sem']**2+b['sem']**2
    df=var**2/(a['sem']**4/a['df']+b['sem']**4/b['df']) if var else 1e9
    se=np.sqrt(var); ci=student.ppf(.975,df)*se
    return {'mean':float(mean),'sem':float(se),'ci95':[float(mean-ci),float(mean+ci)],'df':float(df)}

def arrays(rows):
    keys={'rho':'rho_g_cm3','V':'V_A3_atom','T':'T_K'}
    return {k:np.array([r[v] for r in rows]) for k,v in keys.items()}

for task in tasks:
    state=cfg['states'][task['state']]; name=state['composition']; x=state['x_Al']
    folder=RAW/'15000/results'/str(task['id']); meta=load(folder/'summary.json')
    assert meta['status']=='completed' and meta['job_id']=='20961139'
    for k,v in [('protocol_sha256',sha(RUN/'protocol.json')),('model_sha256',cfg['models']['15000']['sha256'])]: assert meta[k]==v
    base=baseline[name]; assert base['sha256']==sha(base['source'])
    b=base['rows']; assert [r['step'] for r in b]==list(range(0,30001,100))
    series['5000|'+name]=arrays(b[1:])
    for stage in ['equilibration','production']:
        rows=[json.loads(line) for line in (folder/(stage+'-thermo.jsonl')).read_text().splitlines()]
        assert [r['step'] for r in rows]==list(range(0,30001,10))
        arr=arrays(rows[1:]); arr['P']=np.array([r['P_GPa'] for r in rows[1:]])
        series['15000|'+name+'|'+stage]=arr
        with Trajectory(str(folder/(stage+'.traj'))) as tr:
            assert len(tr)==301
            assert all(len(a)==200 and np.count_nonzero(a.numbers==13)==round(x*200) for a in tr)
            density=np.array([a.get_masses().sum()/a.get_volume()*1.66053906660 for a in tr])
            np.testing.assert_allclose(density,[r['rho_g_cm3'] for r in rows[::10]],rtol=0,atol=1e-11)
            if stage=='production':
                frac=np.array([a.get_scaled_positions(wrap=True) for a in tr]); cells=np.array([a.cell.array for a in tr])
        if stage=='production':
            delta=np.diff(frac,axis=0); delta-=np.round(delta)
            cart=np.einsum('tni,tij->tnj',delta,.5*(cells[1:]+cells[:-1])); cart-=cart.mean(axis=1,keepdims=True)
            unwrap=np.concatenate([np.zeros_like(cart[:1]),np.cumsum(cart,axis=0)])
            lags=np.arange(1,101); msd=np.array([np.mean(np.sum((unwrap[l:]-unwrap[:-l])**2,axis=-1)) for l in lags])
            fit=linregress(lags*.1,msd)
            motion[name]={'MSD_10ps_A2':float(msd[-1]),'D_diagnostic_A2_ps':float(fit.slope/6),'r_squared':float(fit.rvalue**2),
                          'lags_ps':(lags*.1).tolist(),'MSD_A2':msd.tolist()}
    for checkpoint,key,dt in [('5000','5000|'+name,.1),('15000','15000|'+name+'|production',.01)]:
        a=series[key]
        results[checkpoint][name]={'x_Al':x,**{k:stats(a[k],dt) for k in ['rho','V','T']}}
        if checkpoint=='15000': results[checkpoint][name]['P']=stats(a['P'],dt)
    new=results['15000'][name]
    remote_row=next(r for r in old_summary['rows'] if r['composition']==name)
    assert abs(new['rho']['mean']-remote_row['rho_g_cm3'])<1e-12
    assert abs(new['rho']['sem']-remote_row['sem_g_cm3'])<1e-12
    prod=series['15000|'+name+'|production']; eq=series['15000|'+name+'|equilibration']
    qc[name]={'half_drift':difference(stats(prod['rho'][1500:],.01),stats(prod['rho'][:1500],.01)),
              'last10_minus_full_g_cm3':float(prod['rho'][-1000:].mean()-prod['rho'].mean()),
              'late_equil_minus_production':difference(stats(eq['rho'][-1000:],.01),new['rho']),
              'rho_100fs_minus_10fs_g_cm3':float(prod['rho'][9::10].mean()-prod['rho'].mean()),
              'T_10fs_ASE_K':new['T']['mean'],'T_100fs_ASE_K':float(prod['T'][9::10].mean()),
              'T_3N_minus_3_K':new['T']['mean']*200/199}
    qc[name]['rho_100fs_all_phase_offsets_g_cm3']=[float(prod['rho'][j::10].mean()-prod['rho'].mean()) for j in range(10)]
    print('Verified',name,'rho15k',round(new['rho']['mean'],6),'MSD10ps',round(msd[-1],2),flush=True)

# Each excess volume uses its own checkpoint's pure-liquid endpoints; exact zero at endpoints.
for checkpoint,group in results.items():
    for name,d in group.items():
        x=d['x_Al']
        d['Vm_cm3mol']=d['V']['mean']*.602214076
        if x in [0.,1.]: value=sem=0.
        else:
            value=(d['V']['mean']-(1-x)*group['Ti']['V']['mean']-x*group['Al']['V']['mean'])*.602214076
            sem=np.sqrt(d['V']['sem']**2+(1-x)**2*group['Ti']['V']['sem']**2+x*x*group['Al']['V']['sem']**2)*.602214076
        d['Vex_cm3mol']={'mean':float(value),'sem':float(sem)}
names=[s['composition'] for s in cfg['states']]; xs=np.array([s['x_Al'] for s in cfg['states']])
diffs={name:difference(results['15000'][name]['rho'],results['5000'][name]['rho']) for name in names}
for name in names: diffs[name]['relative_percent']=100*diffs[name]['mean']/results['5000'][name]['rho']['mean']
thermo=load(ROOT/'audit/oct20/thermo_reprocessed.json'); refs=load(ROOT/'audit/oct20/reference_values.json')
metrics={}
for model in ['pbe5k','pbe15k','sol','scan']:
    values={d['x_Al']:d['rho']['mean'] for d in results['5000' if model=='pbe5k' else '15000'].values()} if model.startswith('pbe') else {r['x_Al']:r['rho'] for r in thermo if r['T']==1873 and r['dft']==model}
    errs=[]; rels=[]
    for x,rho in sorted(values.items()):
        ref=float(np.mean(refs['density_table7'][f'{x:g}']))
        errs.append(abs(rho-ref)); rels.append(abs(rho-ref)/ref*100)
    metrics[model]={'MAE_g_cm3':float(np.mean(errs)),'MARE_percent':float(np.mean(rels)),'n_compositions':len(errs)}

summary={'job_id':'20961139','state':'COMPLETED','elapsed':'09:40:18','gpu':pre['device']['name'],
         'compositions':11,'equil_ps':30,'production_ps':30,'results':results,'density_differences':diffs,
         'qc':qc,'motion':motion,'reference_metrics':metrics,
         'uncertainty':'Maximum SEM from 3, 5, 6 ps blocks. Pointwise approximate Welch 95% intervals for independent-run differences; no multiple-testing adjustment. Shared pure endpoints correlate excess volumes across compositions.',
         'averaging':'Exclude initial frame; 0 < t <= 30 ps. 5k: 300 stored frames every 0.1 ps; 15k: 3000 scalar rows every 0.01 ps. Both use mean instantaneous M/V, not M/mean(V).',
         'sampling_limitation':'The old and new recording intervals differ. In particular, decimating the 15k Ti series from 10 fs to the saved 100 fs phase shifts its mean by 0.014286 g/cm3, exceeding the observed 5k-15k change. Checkpoint causality cannot be isolated by these curves; sampling sensitivity is not included in pointwise block intervals.',
         'motion_note':'Multi-origin MSD after removing affine cell change and COM motion, pooled species. Supports liquid-like mobility; not rigorous structural phase identification or a converged diffusion coefficient.',
         'temperature_note':'Melchionna thermostat targets 3(N-1) kinetic degrees of freedom; ASE unconstrained temperature uses 3N. Compare 200/199 times the logged temperature with nominal 1873 K.'}
save('statistics.json',summary)
table=[]
for name,x in zip(names,xs):
    a=results['5000'][name]; b=results['15000'][name]; d=diffs[name]
    table.append(dict(x_Al=float(x),composition=name,rho_5k=a['rho']['mean'],sem_5k=a['rho']['sem'],
                      rho_15k=b['rho']['mean'],sem_15k=b['rho']['sem'],delta_15k_5k=d['mean'],
                      delta_percent=d['relative_percent'],delta_ci95_low=d['ci95'][0],delta_ci95_high=d['ci95'][1],
                      Vex_5k_cm3mol=a['Vex_cm3mol']['mean'],Vex_15k_cm3mol=b['Vex_cm3mol']['mean'],
                      Vex_15k_sem_cm3mol=b['Vex_cm3mol']['sem']))
with (OUT/'density_5k_15k_1873K.csv').open('w',newline='',encoding='utf-8') as fp:
    writer=csv.DictWriter(fp,fieldnames=list(table[0])); writer.writeheader();writer.writerows(table)

plt.rcParams.update({'font.size':9,'axes.labelsize':10,'axes.titlesize':11,'legend.fontsize':9,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':240})
def figsave(fig,folder,name):
    for ext in ['pdf','png']: fig.savefig(folder/(name+'.'+ext),bbox_inches='tight')
    plt.close(fig)

# Both complete curves on one axis; inset resolves their small separation.
fig,ax=plt.subplots(figsize=(7.2,4.9)); fig.subplots_adjust(left=.11,right=.98,bottom=.14,top=.90)
for checkpoint,color,label,marker,ls in [('5000','#A96818','PBE 5k','o','--'),('15000','#2878B5','PBE 15k','s','-')]:
    vals=[results[checkpoint][n]['rho'] for n in names]
    ax.errorbar(xs,[v['mean'] for v in vals],yerr=[v['sem'] for v in vals],
                color=color,fmt=marker,linestyle=ls,ms=5,lw=1.5,capsize=2,
                mfc='white' if checkpoint=='5000' else color,label=label,zorder=3)
ax.set(xlabel=r'Al atomic fraction, $x_{\mathrm{Al}}$',ylabel=r'Density, $\rho$ (g cm$^{-3}$)',
       xlim=(-.025,1.025),ylim=(2.0,4.45),title='Liquid Ti-Al at 1873 K')
ax.set_xticks(np.arange(0,1.01,.1)); ax.grid(alpha=.15); ax.legend(loc='lower left',frameon=False)
ax.text(.025,.33,'30 ps production\nMain error bars: block SEM\n5k records: 100 fs; 15k records: 10 fs',transform=ax.transAxes,fontsize=8,color='.35')
fig.text(.5,.018,'Intervals exclude sensitivity to the recording interval.',ha='center',fontsize=8,color='.35')
ia=ax.inset_axes([.56,.64,.40,.28])
dy=np.array([diffs[n]['mean'] for n in names]); ci=np.array([diffs[n]['ci95'] for n in names])
ia.errorbar(xs,dy*1000,yerr=np.array([dy-ci[:,0],ci[:,1]-dy])*1000,fmt='o',color='#2878B5',ms=3,capsize=2,lw=.8)
ia.axhline(0,color='.45',lw=.7);ia.grid(alpha=.15)
ia.set(xlim=(-.04,1.04),xticks=[0,.2,.4,.6,.8,1],xlabel=r'$x_{\mathrm{Al}}$',
       ylabel=r'$\Delta\rho$ (mg cm$^{-3}$)',title='15k - 5k; pointwise 95% intervals')
ia.tick_params(labelsize=7);ia.xaxis.label.set_size(8);ia.yaxis.label.set_size(8);ia.title.set_size(8)
figsave(fig,OUT,'density_5k_vs_15k_1873K')

# New version of the existing two-panel article figure; retain the original file.
FIG=ROOT/'manuscript/figures'
fig,axs=plt.subplots(1,2,figsize=(7.0,2.9),layout='constrained')
for model,color,label in [('pbe','#2878B5','PBE (15k)'),('sol','#39835B','PBEsol'),('scan','#D55E00',r'r$^2$SCAN')]:
    if model=='pbe':
        x=xs; rho=[results['15000'][n]['rho']['mean'] for n in names]; vex=[results['15000'][n]['Vex_cm3mol']['mean'] for n in names]
    else:
        g=sorted([r for r in thermo if r['T']==1873 and r['dft']==model],key=lambda r:r['x_Al'])
        x=[r['x_Al'] for r in g];rho=[r['rho'] for r in g];vex=[r['Vex_cm3mol'] for r in g]
    axs[0].plot(x,rho,'o-',color=color,ms=3,lw=1,label=label)
    axs[1].plot(x,vex,'o-',color=color,ms=3,lw=1,label=label)
xx=[]; yy=[]; vv=[]
for x,vals in refs['density_table7'].items():
    x=float(x)
    for rho in vals:
        xx.append(x);yy.append(rho)
        vv.append(((1-x)*47.867+x*26.9815385)/rho-(1-x)*47.867/4.184-x*26.9815385/2.08)
axs[0].scatter(xx,yy,c='black',marker='s',s=14,zorder=5,label='Published densities')
axs[1].scatter(xx,vv,c='black',marker='s',s=14,zorder=5,label='From published densities')
axs[1].axhline(0,c='.6',lw=.7)
for ax in axs:
    ax.set(xlabel=r'$x_{\mathrm{Al}}$',xlim=(-.02,1.02));ax.legend(frameon=False,fontsize=7,loc='upper right')
axs[0].set(title='(a) Density, 1873 K',ylabel=r'$\rho$ (g cm$^{-3}$)')
axs[1].set(title='(b) Excess volume, 1873 K',ylabel=r'$V_m^{\mathrm{ex}}$ (cm$^3$ mol$^{-1}$)')
axs[1].set_ylim(-1.14,.60)
figsave(fig,FIG,'liquid_density_pbe15k')
caption=r'''\caption{Liquid density (a) and excess molar volume (b) at 1873~K. The PBE series uses checkpoint 15k and 30~ps production trajectories after 30~ps of equilibration; the PBEsol and r$^2$SCAN series retain the archived results. Black symbols in (a) are the printed densities in Table~7 of Ref.~\onlinecite{Brillo2025}; separate entries at the same composition are shown individually. Reference excess volumes in (b) are recomputed from those densities and their pure-component endpoints. Each MLIP excess volume uses its own pure-liquid references. Some experimental reference values represent extrapolated or undercooled-liquid states.}'''
(OUT/'article_caption.tex').write_text(caption+'\n',encoding='utf-8')

# Compact diagnostics for all compositions, saved as a PNG rather than another deliverable PDF.
fig,axs=plt.subplots(3,4,figsize=(12,7.4),layout='constrained')
for ax,name,x in zip(axs.flat,names,xs):
    for stage,offset,color in [('equilibration',0,'#999999'),('production',30,'#2878B5')]:
        a=series['15000|'+name+'|'+stage]['rho']; means=a.reshape(30,100).mean(axis=1)
        ax.plot(np.arange(30)+.5+offset,means-results['15000'][name]['rho']['mean'],'o-',ms=2,lw=.8,color=color)
    ax.axhline(0,c='.3',lw=.5);ax.axvline(30,c='.6',ls='--',lw=.6)
    ax.set(title=f'x(Al) = {x:.1f}',xlabel='Time (ps)',ylabel=r'$\rho-\bar\rho_{prod}$ (g cm$^{-3}$)');ax.grid(alpha=.12)
axs.flat[-1].axis('off');axs.flat[-1].text(.02,.7,'PBE 15k, 1873 K\nGray: equilibration\nBlue: production\nEach point: 1 ps mean',va='top')
fig.savefig(OUT/'density_time_traces.png',bbox_inches='tight');plt.close(fig)

maxname=max(names,key=lambda n:abs(diffs[n]['mean']))
min15=min(names,key=lambda n:results['15000'][n]['Vex_cm3mol']['mean'])
significant=[results['15000'][n]['x_Al'] for n in names if np.prod(diffs[n]['ci95'])>0]
lines=['PBE 15k, 1873 K: проверка завершённой серии и сравнение с 5k',
       'Задание 20961139 COMPLETED, ExitCode 0:0, 09:40:18, одна NVIDIA A100-PCIE-40GB.',
       'Все 11 составов: 200 атомов, 1 бар, 30 пс уравновешивания + 30 пс production. Проверены SHA256 99 файлов, 3001 скалярная запись и 301 кадр каждой стадии.',
       'Проверка CPU/GPU пройдена для всех исходных конфигураций. Пересчитанная из траекторий плотность совпадает с логами. Идентичность чекпойнта 15000 и протокола подтверждена.',
       f"Максимальная разность 15k-5k: {diffs[maxname]['mean']:.6f} г/см3 ({diffs[maxname]['relative_percent']:.3f}%) при x_Al={results['15000'][maxname]['x_Al']:.1f}.",
       f"Точечный 95%-интервал разности не включает ноль при x_Al={significant}; поправка на множественные сравнения не применялась.",
       'Основное среднее: 0 < t <= 30 пс. 5k: 300 кадров через 100 фс; 15k: 3000 записей через 10 фс. Учитывается среднее M/V, а не M/<V>. Ошибки: максимум SEM по блокам 3, 5, 6 пс.',
       f"Максимальный модуль дрейфа между половинами production 15k: {max(abs(v['half_drift']['mean']) for v in qc.values()):.6f} г/см3.",
       f"Максимальный сдвиг оценки 15k при усреднении только последних 10 пс: {max(abs(v['last10_minus_full_g_cm3']) for v in qc.values()):.6f} г/см3.",
       f"Прореживание 15k до 100 фс меняет плотность максимум на {max(abs(v['rho_100fs_minus_10fs_g_cm3']) for v in qc.values()):.6f} г/см3; сравнение близких чекпойнтов чувствительно к частоте сохранения.",
       f"MSD при лаге 10 пс: {min(v['MSD_10ps_A2'] for v in motion.values()):.2f}-{max(v['MSD_10ps_A2'] for v in motion.values()):.2f} A^2. Все траектории сохраняют диффузионную подвижность; это не строгая структурная классификация фаз.",
       f"Средняя температура по 3N в ASE: {min(v['T_10fs_ASE_K'] for v in qc.values()):.2f}-{max(v['T_10fs_ASE_K'] for v in qc.values()):.2f} K. По 3(N-1) степеням свободы термостата: {min(v['T_3N_minus_3_K'] for v in qc.values()):.2f}-{max(v['T_3N_minus_3_K'] for v in qc.values()):.2f} K.",
       f"Среднее давление 15k: {min(d['P']['mean'] for d in results['15000'].values()):.5f}-{max(d['P']['mean'] for d in results['15000'].values()):.5f} ГПа при заданных 0.0001 ГПа. Остаточное смещение давления и влияние численного протокола требуют отдельной проверки для прецизионных выводов; SEM плотности таких эффектов не учитывает.",
       f"Минимум избыточного объёма PBE15k: {results['15000'][min15]['Vex_cm3mol']['mean']:.5f} +/- {results['15000'][min15]['Vex_cm3mol']['sem']:.5f} см3/моль при x_Al={results['15000'][min15]['x_Al']:.1f}.",
       'Сравнение с опубликованными плотностями сохраняет прежнюю табличную конвенцию и её неопределённости; новые источники не добавлялись.',
       f"PBE15k: MAE={metrics['pbe15k']['MAE_g_cm3']:.6f} г/см3, MARE={metrics['pbe15k']['MARE_percent']:.3f}%; PBE5k: {metrics['pbe5k']['MAE_g_cm3']:.6f} г/см3, {metrics['pbe5k']['MARE_percent']:.3f}%.",
       'Разность между сериями мала. Номинальные интервалы не включают чувствительность к частоте записи; нельзя интерпретировать всю разность кривых как чистый эффект смены чекпойнта. Для чистого сравнения нужен 5k с тем же частым логированием. Дополнительные MD не запускались.',
       'Новая версия рисунка статьи: manuscript/figures/liquid_density_pbe15k.pdf; обе панели PBE обновлены согласованно, остальные серии и опубликованные точки сохранены. Исходный liquid_density.pdf сохранён. Подпись для новой версии: article_caption.tex.',
       'x_Al | rho5k +/- SEM | rho15k +/- SEM | delta15k-5k (г/см3)']
for row in table:
    lines.append(f"{row['x_Al']:.1f} | {row['rho_5k']:.6f} +/- {row['sem_5k']:.6f} | {row['rho_15k']:.6f} +/- {row['sem_15k']:.6f} | {row['delta_15k_5k']:+.6f}")
(OUT/'REPORT_RU.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
sources=[RUN/'protocol.json',RUN/'baseline_5k.json',RUN/'collection-manifest.json',ROOT/'audit/oct20/thermo_reprocessed.json',ROOT/'audit/oct20/reference_values.json',Path(__file__)]
save('figure_sources.json',{'sources':{str(p):sha(p) for p in sources},
    'article_output':str(FIG/'liquid_density_pbe15k.pdf'),'comparison_output':str(OUT/'density_5k_vs_15k_1873K.pdf'),
    'old_article_pdf_retained':str(FIG/'liquid_density.pdf'),
    'old_article_pdf_sha256':sha(FIG/'liquid_density.pdf'),
    'published_reference_convention':'Existing printed 1873 K Table 7 values; same duplicated entries and endpoint reference as original manuscript figure.'})
print('\n'.join(lines[:16]),flush=True)
