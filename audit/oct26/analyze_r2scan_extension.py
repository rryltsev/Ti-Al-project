"""Independent trajectory verification and 30/60 ps enthalpy uncertainty analysis."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
from scipy import stats
from ase import units
from ase.io.trajectory import Trajectory

HERE = Path(__file__).resolve().parent
RUN = HERE/'r2scan_enthalpy_extend30k'
SRC = RUN/'collected'
OUT = RUN/'analysis'
OUT.mkdir(exist_ok=True)
os.environ['MPLBACKEND'] = 'Agg'
os.environ['MPLCONFIGDIR'] = str(OUT/'.mplconfig')
import matplotlib.pyplot as plt

C = 96.48533212331
cfg = json.loads((SRC/'protocol.json').read_text())
baseline = json.loads((SRC/'baseline.json').read_text())
remote_summary = json.loads((SRC/'enthalpy_summary.json').read_text())['compositions']
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert json.loads((RUN/'collection-verification.json').read_text())['status'] == 'passed'
assert json.loads((SRC/'preflight.json').read_text())['status'] == 'passed'

def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

def blocks(values, frames=50):
    assert len(values) % frames == 0
    return np.asarray(values).reshape(-1, frames).mean(axis=1)

def sem(values):
    return float(np.std(values, ddof=1)/np.sqrt(len(values)))

def bh(p):
    p = np.asarray(p)
    order = np.argsort(p)
    q = np.empty_like(p)
    q[order] = np.minimum(1, np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1])
    return q

def atom_row(a):
    n, v = len(a), a.get_volume()
    ep, ek = a.get_potential_energy(), a.get_kinetic_energy()
    return np.array([(ep+ek+units.bar*v)/n*C, a.get_temperature(),
                     float(a.get_masses().sum()/v*1.66053906660), ep/n, ek/n, v/n])

series, verification, sources = {}, [], []
for state in cfg['states']:
    name = state['composition']
    original = Path(state['initial_source'])
    assert sha(original) == state['source_sha256']
    summary = json.loads((SRC/'results'/name/'summary.json').read_text())
    assert summary['status'] == 'completed' and summary['steps'] == 30000
    assert summary['model_sha256'] == cfg['model']['sha256'] and summary['protocol_sha256'] == sha(SRC/'protocol.json')
    assert summary['job_id'] == '20961890'
    path = SRC/'results'/name/'production.traj'
    with Trajectory(original) as oldtr, Trajectory(path) as newtr:
        assert len(oldtr) == len(newtr) == 301
        oldlast, newfirst = oldtr[-1], newtr[0]
        for field in ['positions','momenta','numbers']:
            assert np.array_equal(oldlast.arrays[field], newfirst.arrays[field])
        assert np.array_equal(oldlast.cell.array, newfirst.cell.array)
        for trajectory in [oldtr, newtr]:
            for a in trajectory:
                assert len(a) == 200 and np.mean(a.numbers == 13) == state['x_Al']
        old, new = np.array([atom_row(a) for a in oldtr]), np.array([atom_row(a) for a in newtr])
    assert np.isfinite(old).all() and np.isfinite(new).all()
    assert np.max(abs(old[:,0]-np.array([r['H_external_eV_atom']*C for r in baseline[name]['rows']]))) < 1e-9
    thermo = [json.loads(line) for line in (path.parent/'production-thermo.jsonl').read_text().splitlines()]
    assert [r['step'] for r in thermo] == list(range(0,30001,10))
    assert [r['production_step'] for r in thermo] == list(range(30000,60001,10))
    fields = ['H_external_eV_atom','T_K','rho_g_cm3','Epot_eV_atom','Ekin_eV_atom','V_A3_atom']
    logged = np.array([[r[f] for f in fields] for r in thermo[::10]])
    logged[:,0] *= C
    assert np.max(abs(logged-new)) < 1e-9
    series[name] = dict(x=state['x_Al'], old=old, new=new[1:], combined=np.concatenate([old,new[1:]]),
                        diagnostic=np.concatenate([old[1:],new[1:]]))
    sources.append(dict(original=str(original), original_sha256=sha(original),
                        new=str(path), new_sha256=sha(path)))
    verification.append(dict(composition=name, old_frames=301,new_frames=301,combined_frames=601,
                              diagnostic_frames=600,thermo_rows=3001,exact_restart_frame=True,
                              logged_thermo_matches_trajectory=True,wall_seconds=summary['wall_seconds']))
save('verification.json',dict(status='passed', job='20961890', compositions=verification, sources=sources))

names = list(series)
xs = np.array([series[n]['x'] for n in names])
internal = [n for n in names if 0 < series[n]['x'] < 1]
means, errors, covariance, perH = {}, {}, {}, {}
for window in ['old','new','combined']:
    means[window], errors[window], perH[window] = {}, {}, {}
    for name, s in series.items():
        diagnostic = s['old'][1:] if window == 'old' else s['new'] if window == 'new' else s['diagnostic']
        perH[window][name] = dict(mean=float(s[window][:,0].mean()), sem=sem(blocks(diagnostic[:,0])))
    al, ti = perH[window]['Al'], perH[window]['Ti']
    for name, s in series.items():
        x, h = s['x'], perH[window][name]
        means[window][name] = h['mean']-x*al['mean']-(1-x)*ti['mean'] if 0 < x < 1 else 0.
        errors[window][name] = float(np.sqrt(h['sem']**2+x*x*al['sem']**2+(1-x)**2*ti['sem']**2)) if 0 < x < 1 else 0.
    cov = np.zeros((11,11))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            x,y = series[a]['x'],series[b]['x']
            if 0 < x < 1 and 0 < y < 1:
                cov[i,j] = x*y*al['sem']**2+(1-x)*(1-y)*ti['sem']**2
                if i == j: cov[i,j] += perH[window][a]['sem']**2
    covariance[window] = cov.tolist()
    assert np.allclose(np.diag(cov), [errors[window][n]**2 for n in names],rtol=1e-12,atol=1e-15)

rows, segment_tests, obs_tests = [], [], []
for name in internal:
    s = series[name]
    x = s['x']
    row = dict(composition=name, x_Al=x, deltaH_30_kJ_mol=means['old'][name],
               SEM_30_kJ_mol=errors['old'][name],deltaH_60_kJ_mol=means['combined'][name],
               SEM_60_kJ_mol=errors['combined'][name],deltaH_new30_kJ_mol=means['new'][name],
               SEM_new30_kJ_mol=errors['new'][name],shift_60_minus_30=means['combined'][name]-means['old'][name],
               SEM_ratio_60_over_30=errors['combined'][name]/errors['old'][name])
    for key, remote_key in [('deltaH_30_kJ_mol','deltaH_old_kJ_mol'),('deltaH_60_kJ_mol','deltaH_kJ_mol'),
                            ('SEM_60_kJ_mol','deltaH_diagnostic_SEM_kJ_mol')]:
        assert abs(row[key]-remote_summary[name][remote_key]) < 1e-9
    rows.append(row)
    # Compare disjoint 30 ps segments, not nested 30/60 ps estimates.
    components = [(name,1.),('Al',x),('Ti',1-x)]
    terms, diff = [], 0.
    for component, weight in components:
        sign = 1 if component == name else -1
        sc = series[component]
        oldb, newb = blocks(sc['old'][1:,0]), blocks(sc['new'][:,0])
        diff += sign*weight*(newb.mean()-oldb.mean())
        terms.extend([(weight*sem(oldb))**2,(weight*sem(newb))**2])
    variance = sum(terms)
    df = variance**2/sum(v*v/5 for v in terms)
    segment_tests.append(dict(composition=name,x_Al=x,new_minus_old_kJ_mol=float(diff),
                             difference_SEM=float(np.sqrt(variance)),t=float(diff/np.sqrt(variance)),df=float(df),
                             p=float(2*stats.t.sf(abs(diff/np.sqrt(variance)),df))))
for row,q in zip(segment_tests,bh([r['p'] for r in segment_tests])): row['q_BH']=float(q)
for name,s in series.items():
    for index,observable in enumerate(['H_kJ_mol','T_K','rho_g_cm3']):
        oldb,newb=blocks(s['old'][1:,index]),blocks(s['new'][:,index])
        test=stats.ttest_ind(oldb,newb,equal_var=False)
        obs_tests.append(dict(composition=name,observable=observable,old_mean=float(oldb.mean()),new_mean=float(newb.mean()),
                              difference=float(newb.mean()-oldb.mean()),old_SEM=sem(oldb),new_SEM=sem(newb),p=float(test.pvalue)))
for row,q in zip(obs_tests,bh([r['p'] for r in obs_tests])): row['q_BH']=float(q)

sensitivity=[]
for frames in [10,20,30,50,100]:
    for window in ['old','combined']:
        variances={}
        for name,s in series.items():
            data=s['old'][1:,0] if window=='old' else s['diagnostic'][:,0]
            variances[name]=sem(blocks(data,frames))**2
        for name in internal:
            x=series[name]['x']
            error=np.sqrt(variances[name]+x*x*variances['Al']+(1-x)**2*variances['Ti'])
            sensitivity.append(dict(composition=name,window=window,block_ps=frames*.1,
                                    n_blocks=(300 if window=='old' else 600)//frames,SEM=float(error)))

restart=[]
for cut_ps in [0,1,2,5]:
    h={n:float(np.concatenate([s['old'][:,0],s['new'][cut_ps*10:,0]]).mean()) for n,s in series.items()}
    for name in internal:
        x=series[name]['x']; dh=h[name]-x*h['Al']-(1-x)*h['Ti']
        assert abs(dh-remote_summary[name]['restart_sensitivity_deltaH_kJ_mol'][str(cut_ps)])<1e-9
        restart.append(dict(composition=name,cut_new_start_ps=cut_ps,deltaH=dh,
                            shift_from_untrimmed=dh-means['combined'][name]))

i,j=names.index('Al0.6Ti0.4'),names.index('Al0.5Ti0.5')
cov=np.array(covariance['combined'])
min_difference=dict(deltaH_x06_minus_x05=means['combined'][names[i]]-means['combined'][names[j]],
                    SEM=float(np.sqrt(cov[i,i]+cov[j,j]-2*cov[i,j])),shared_endpoint_covariance_included=True)
report=dict(job_id='20961890',status='COMPLETED',elapsed='04:50:15',states=11,
            steps_added_each=30000,total_ps_each=60,T_K=2000,P_bar=1,model=cfg['model'],
            method=dict(enthalpy='(Epot+Ekin+P_ext V)/N; P_ext=1 bar',
                        mean_frames_30=301,mean_frames_60=601,boundary_duplicate_excluded=True,
                        diagnostic_frames_30=300,diagnostic_frames_60=600,block_ps=5,blocks_30=6,blocks_60=12,
                        error='SEM of block means; propagated alloy and pure-component variances. Diagnostic, not a 95% confidence interval.',
                        assumptions='Independent composition runs and approximately independent 5 ps blocks; systematic potential and finite-size errors excluded.',
                        segment_test='Welch-Satterthwaite approximation for disjoint old/new segments; BH adjustment across nine compositions. Correlations between segments, if present, are not estimated.',
                        restart='Exact atomic state preserved; thermostat/barostat internal variables reset. All new steps retained in primary estimate.'),
            comparison=rows,segment_tests=segment_tests,observable_segment_tests=obs_tests,
            block_sensitivity=sensitivity,restart_sensitivity=restart,min_x06_vs_x05=min_difference,
            sem_decreased=sum(r['SEM_ratio_60_over_30']<1 for r in rows),sem_increased=sum(r['SEM_ratio_60_over_30']>1 for r in rows),
            covariance=dict(composition_order=names,matrices=covariance))
save('comparison.json',report)

plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
blue,orange='#2469A0','#D66B20'
fig,axes=plt.subplots(1,2,figsize=(12,4.7))
for window,label,color,marker in [('old','30 ps',blue,'o'),('combined','60 ps',orange,'s')]:
    axes[0].errorbar(xs,[means[window][n] for n in names],yerr=[errors[window][n] for n in names],
                     color=color,marker=marker,label=label,linewidth=1.5,capsize=3,markersize=4)
    axes[1].plot(xs[1:-1],[errors[window][n] for n in internal],color=color,marker=marker,label=label)
axes[0].set(xlabel=r'Al atomic fraction $x_{Al}$',ylabel=r'$\Delta H_{mix}$ (kJ mol$^{-1}$ of atoms)',title='Mixing enthalpy')
axes[1].set(xlabel=r'Al atomic fraction $x_{Al}$',ylabel=r'Diagnostic SEM (kJ mol$^{-1}$ of atoms)',title='Uncertainty from 5 ps blocks')
for ax in axes:
    ax.set_xticks(np.linspace(0,1,11));ax.grid(alpha=.2);ax.legend(frameon=False)
fig.suptitle(r'r$^2$SCAN: 2000 K, 1 bar, 200 atoms',fontsize=14)
fig.tight_layout()
for ext in ['png','svg']:fig.savefig(OUT/('enthalpy_30_vs_60ps.'+ext),dpi=190,bbox_inches='tight')
plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(11,8))
for ax,name in zip(axes.flat,['Al0.2Ti0.8','Al0.3Ti0.7','Al0.5Ti0.5','Al0.6Ti0.4']):
    x=series[name]['x']
    time=np.arange(1,601)*.1
    cumulative={n:np.cumsum(s['diagnostic'][:,0])/np.arange(1,601) for n,s in series.items()}
    mix=cumulative[name]-x*cumulative['Al']-(1-x)*cumulative['Ti']
    ax.plot(time[49:],mix[49:],color=blue,label='Cumulative mean (t > 0)')
    bmix=blocks(series[name]['diagnostic'][:,0])-x*blocks(series['Al']['diagnostic'][:,0])-(1-x)*blocks(series['Ti']['diagnostic'][:,0])
    ax.scatter(np.arange(12)*5+2.5,bmix,color=orange,s=24,label='5 ps block values')
    ax.axvline(30,color='#666666',ls='--',lw=1,label='Restart boundary')
    ax.axhline(means['combined'][name],color=blue,ls=':',lw=1)
    ax.set(title=f'$x_{{Al}}$ = {x:.1f}',xlabel='Production time (ps)',ylabel=r'$\Delta H_{mix}$ (kJ mol$^{-1}$)')
    ax.grid(alpha=.2)
axes.flat[0].legend(frameon=False,fontsize=8)
fig.suptitle('Convergence diagnostics: compositions with increased SEM',fontsize=13)
fig.tight_layout()
for ext in ['png','svg']:fig.savefig(OUT/('convergence_diagnostics.'+ext),dpi=180,bbox_inches='tight')
plt.close(fig)

lines=['r2SCAN: сравнение энтальпии смешения за 30 и 60 пс',
       'Задание 20961890: COMPLETED, код 0:0, длительность 4:50:15, одна A100.',
       'Все 11 составов: дополнительно 30000 шагов по 1 фс; 2000 K, 1 бар, 200 атомов.',
       'Проверены контрольные суммы, точное совпадение стартовых кадров, 301 кадр и 3001 термодинамическая запись на состав.',
       '', 'x(Al)    ΔH30 ± SEM30          ΔH60 ± SEM60         ΔH60−ΔH30     SEM60/SEM30',
       'Все энергии и ошибки в кДж/моль атомов.']
for r in rows:
    lines.append(f"{r['x_Al']:.1f}     {r['deltaH_30_kJ_mol']:8.3f} ± {r['SEM_30_kJ_mol']:.3f}    {r['deltaH_60_kJ_mol']:8.3f} ± {r['SEM_60_kJ_mol']:.3f}     {r['shift_60_minus_30']:+.3f}          {r['SEM_ratio_60_over_30']:.3f}")
lines += ['', 'Для чистых Al и Ti энтальпия смешения и её ошибка тождественно равны нулю.',
          'H = <Epot + Ekin + p_ext V>/N; ΔH(x)=H(x)−x H(Al)−(1−x)H(Ti).',
          'Средние: исходные 301 кадр, объединённые 601; повторный граничный кадр исключён.',
          'SEM: шесть / двенадцать блоков по 5 пс; исходный t=0 исключён только из блоковой диагностики.',
          'В дисперсию ΔH включены сплав и чистые компоненты; общие эталоны коррелируют ошибки между составами.',
          '± обозначает диагностическую стандартную ошибку среднего, а не 95%-ный интервал.',
          'Систематические ошибки потенциала и конечного размера здесь не оценивались.',
          '', f"Ошибки уменьшились для {report['sem_decreased']} составов и выросли для {report['sem_increased']}.",
          f"Разность ΔH(0.6)−ΔH(0.5) на 60 пс: {min_difference['deltaH_x06_minus_x05']:+.3f} ± {min_difference['SEM']:.3f} кДж/моль атомов.",
          'Это не позволяет уверенно различить положения минимума x=0.5 и x=0.6.',
          '', 'Новые 30 пс минус старые 30 пс (для диагностики обе выборки без начального кадра):']
for r in segment_tests:
    lines.append(f"x={r['x_Al']:.1f}: {r['new_minus_old_kJ_mol']:+.3f} ± {r['difference_SEM']:.3f}; p={r['p']:.4f}, q_BH={r['q_BH']:.4f}")
lines += ['', 'Сравнение 30/60 пс использует вложенные выборки: значимость сдвига нельзя оценивать как для независимых средних.',
          'Диагностика различий выполнена по двум неперекрывающимся 30-пс участкам с приближением Welch–Satterthwaite и BH по 9 составам.',
          'До поправки за множественные сравнения p<0.05 у x=0.3, 0.5 и 0.6; после BH нет q<0.05 (минимальное q=0.137).',
          'Она предполагает приблизительную независимость блоков и не исключает медленных корреляций.',
          'Координаты, ячейка и импульсы при рестарте совпали точно. Переменные термостата/баростата сброшены.',
          'Основная оценка включает все новые шаги. Отдельно сохранена чувствительность к исключению первых 1/2/5 пс продолжения.',
          'Максимальная поправка к ΔH при исключении первых 1/2/5 пс продолжения: 0.041/0.043/0.085 кДж/моль атомов.',
          'Для x=0.6 SEM на 60 пс меняется с 0.200 при блоках 5 пс до 0.267 при блоках 10 пс; независимость 5-пс блоков не установлена.',
          'Отсутствие значимых после BH различий не доказывает стационарность. Для H, температуры и плотности также нет q<0.05 среди 33 сравнений участков.',
          'Удвоение выборки уточнило среднюю кривую, но не обеспечило равномерного уменьшения SEM. Снижение как 1/sqrt(2) нельзя приписывать этим данным.',
          'Чувствительность SEM к блокам 1/2/3/5/10 пс, тесты H/T/плотности и ковариации находятся в comparison.json.',
          'Рукопись и исходные траектории не изменены.']
(OUT/'REPORT_RU.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(dict(comparison=rows,segment_tests=segment_tests,min_difference=min_difference,
     significant_observables=[r for r in obs_tests if r['q_BH']<.05],
     maximum_restart_effect=max(abs(r['shift_from_untrimmed']) for r in restart),
     outputs=str(OUT)),ensure_ascii=False,indent=2))
