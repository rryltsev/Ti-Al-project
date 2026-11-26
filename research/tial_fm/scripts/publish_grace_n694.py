"""Independent local verification and publication, only after all six runs finish."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare_grace_n694 import ROOT,RUN,PREVIOUS,METHODS,load,dump,sha
from prepare_grace_pbe_seeds import PARENT,MASK,TI3
from grace_mask_aggregate import metrics
from publish_ti3al_pbe import metrics as dos_metrics

def main():
    data=RUN/'collected-final';h=load(data/'COLLECTION_SHA256.json')
    for n,v in h.items():assert sha(data/n)==v,n
    remote=load(RUN/'final-remote-verification.json');assert remote['status']=='PASS'
    result=load(data/'comparison/summary.json');assert result['status']=='PASS' and len(result['cases'])==12 and not result['r2SCAN_tested']
    ref=np.load(RUN/'package/reference_figure.npz');dft=np.loadtxt(RUN/'package/ti3al/source/Ti3Al_D019/total_dos.dat')
    memberships={n:load(RUN/'package'/('membership.json' if n==694 else 'membership-N200.json')) for n in [200,694]}
    spectra={};count=0
    for name,r in result['cases'].items():
        c=r['case'];n=c['N'];seed=c['seed'];method=c['method'];assert c['XC']=='PBE' and method in METHODS
        base=data if n==694 else (PREVIOUS/'collected-final' if seed!=10 else (PARENT if method=='full' else MASK)/'collected-final')
        w=base/'cases'/name;t=load(w/'training-summary.json');assert t['status']=='PASS' and t['hashes']==r['checkpoint_hashes'] and t['steps']==n*50
        for f in ['evaluation-summary.json','numerical-QC.json','zero-step.json','checkpoint-path-check.json','harmonics/summary.json']:assert load(w/f)['status']=='PASS'
        checks=[load(p) for p in (w/'checks').glob('*.json')];assert {v['steps'] for v in checks}=={1,n*50,*range(1000,n*50+1,1000)}
        assert all(v['status']=='PASS' and not v['frozen_changed'] and v['steps']==v['optimizer_iterations'] for v in checks);count+=len(checks)
        if n==694:
            v=remote['cases'][name];assert v['checkpoint']==t['hashes'] and v['training_summary_sha256']==sha(w/'training-summary.json')
            assert v['model_yaml_sha256']==sha(w/'seed'/str(seed)/'model.yaml')
        rows=load(w/'per-structure.json');assert {p:sorted(x['id'] for x in rows if x['split']==p) for p in ['train','validation']}==memberships[n]
        m=metrics(rows,load(w/'metrics.json'),n)
        for k,v in m['validation'].items():assert np.isclose(v,r['values'][k],rtol=1e-11,atol=1e-12)
        b=np.load(w/'harmonics/PBE/bands.npz');assert np.array_equal(b['qpoints'],ref['qpoints'])
        assert np.isclose(np.sqrt(np.mean((b['frequencies']-ref['PBE_DFT_bands'])**2)),r['values']['TiAl_band_RMSE_THz'],atol=1e-12)
        mesh=np.load(w/'harmonics/PBE/mesh50.npz');assert mesh['weights'].sum()==125000
        assert int(np.sum(mesh['weights'][:,None]*(mesh['frequencies']<-.001)))==r['phonons']['imaginary_modes_weighted']
        if n==694 or seed!=10:ti=base/'ti3al';at=ti;tn=name
        else:ti=TI3/'collected';at=TI3/'collected-asr';tn='GRACE-'+method+'-N200-seed10'
        d=np.load(ti/'analysis'/tn/'dos_extended.npz');dm=dos_metrics(d['frequency'],d['dos'],dft[:,0],dft[:,1]/8)
        assert np.isclose(dm['DOS_Wasserstein_THz'],r['values']['Ti3Al_DOS_W1_THz'],atol=1e-12) and np.isclose(dm['normalized_DOS_L1'],r['values']['Ti3Al_DOS_L1'],atol=1e-12)
        tm=np.load(ti/'analysis'/tn/'mesh14.npz');assert tm['weights'].sum()==14**3
        assert int(np.sum(tm['weights'][:,None]*(tm['frequencies']<-.001)))==r['Ti3Al']['mesh_imaginary']
        ad=np.load(at/'analysis-asr'/tn/'dos.npz');am=dos_metrics(ad['frequency'],ad['dos'],dft[:,0],dft[:,1]/8)
        assert np.isclose(am['DOS_Wasserstein_THz'],r['values']['Ti3Al_DOS_ASR_W1_THz'],atol=1e-12)
        assert int((np.load(at/'analysis-asr'/tn/'bands.npz')['frequencies']<-.001).sum())==r['Ti3Al']['ASR_band_imaginary']
        spectra[name]=dict(TiAl=b['frequencies'],frequency=d['frequency'],DOS=d['dos'])
    for n,methods in result['groups'].items():
        for method,g in methods.items():
            for k,v in g.items():
                vals=[result['cases'][f'PBE-{method}-N{n}-seed{s}']['values'][k] for s in [10,20,30]]
                assert np.allclose(vals,v['values']) and np.isclose(np.mean(vals),v['mean']) and np.isclose(np.std(vals,ddof=1),v['sample_SD'])
    for method,values in result['paired_N694_minus_N200'].items():
        for key,delta in values.items():
            expected=[result['cases'][f'PBE-{method}-N694-seed{s}']['values'][key]-result['cases'][f'PBE-{method}-N200-seed{s}']['values'][key] for s in [10,20,30]]
            assert np.allclose(expected,delta,rtol=1e-12,atol=1e-12)
    assert count==282
    result['local_verification']=dict(status='PASS',files=len(h),models=12,parameter_checks=count,independent_metric_recalculation=True)
    out=RUN/'comparison-local';out.mkdir(exist_ok=True);dump(out/'comparison.json',result)
    figs=ROOT/'reports/grace-pbe-n694';figs.mkdir(exist_ok=True)
    specs=[('E_aligned_RMSE_meV_atom','Energy (meV/atom)'),('F_RMSE_eV_A','Force (eV/Å)'),('stress_RMSE_GPa','Stress (GPa)'),('TiAl_band_RMSE_THz','TiAl RMSE (THz)'),('Ti3Al_DOS_W1_THz','Ti3Al DOS W1 (THz)'),('Ti3Al_DOS_L1','Ti3Al DOS L1')]
    combos=[(n,m) for n in ['200','694'] for m in METHODS];labels=[n+' '+('Full' if m=='full' else '+Radial') for n,m in combos]
    fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    for ax,(key,title) in zip(axes.flat,specs):
        gs=[result['groups'][n][m][key] for n,m in combos]
        ax.bar(range(4),[g['mean'] for g in gs],yerr=[g['sample_SD'] for g in gs],capsize=4,color=['#167a9e','#8356a5','#66afca','#b799cb'])
        for i,g in enumerate(gs):ax.scatter([i-.08,i,i+.08],g['values'],c='black',s=14,zorder=5)
        ax.set_xticks(range(4),labels,rotation=25);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    fig.suptitle('GRACE PBE · N200 versus N694 · mean ± sample SD, seeds10/20/30')
    for ext in ['png','pdf']:fig.savefig(figs/f'comparison.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    x=ref['path_coordinate'];breaks=np.r_[0,np.flatnonzero(np.linalg.norm(np.diff(ref['qpoints'],axis=0),axis=1)>.05)+1,len(x)]
    for row,method in enumerate(METHODS):
        ax,ad=axes[row]
        for start,end in zip(breaks[:-1],breaks[1:]):
            for branch in range(6):ax.plot(x[start:end],ref['PBE_DFT_bands'][start:end,branch],c='black',ls='--',lw=1,label='PBE DFT' if start==branch==0 else None)
        ad.plot(dft[:,0],dft[:,1]/8,c='black',ls='--',lw=1.2,label='PBE DFT')
        for n,color in [(200,'#167a9e'),(694,'#d47729')]:
            ss=[spectra[f'PBE-{method}-N{n}-seed{s}'] for s in [10,20,30]]
            arr=np.array([s['TiAl'] for s in ss]);mu=arr.mean(axis=0);sd=arr.std(axis=0,ddof=1)
            for start,end in zip(breaks[:-1],breaks[1:]):
                for branch in range(6):
                    ax.plot(x[start:end],mu[start:end,branch],c=color,lw=1,label=f'N={n}' if start==branch==0 else None)
                    ax.fill_between(x[start:end],(mu-sd)[start:end,branch],(mu+sd)[start:end,branch],color=color,alpha=.17)
            assert all(np.array_equal(s['frequency'],ss[0]['frequency']) for s in ss)
            arr=np.array([s['DOS'] for s in ss]);mu=arr.mean(axis=0);sd=arr.std(axis=0,ddof=1);freq=ss[0]['frequency']
            ad.plot(freq,mu,c=color,lw=1,label=f'N={n}');ad.fill_between(freq,mu-sd,mu+sd,color=color,alpha=.17)
        ax.set(title=method+' · TiAl',xlabel='Archived band-path coordinate',ylabel='Frequency (THz)',ylim=(-.2,9));ax.legend(fontsize=9)
        ad.set(title=method+' · Ti3Al fixed Cmcm cell',xlabel='Frequency (THz)',ylabel='DOS (modes / atom / THz)',xlim=(-.5,10.5));ad.legend(fontsize=9)
    fig.suptitle('GRACE PBE · N200 versus N694 · mean ± sample SD across seeds10/20/30')
    for ext in ['png','pdf']:fig.savefig(figs/f'spectra.{ext}',dpi=180)
    plt.close(fig)
    lines=['# GRACE PBE: N694 против N200','', 'Все шесть обучений20991849_0–5 и сравнение20991850 завершены COMPLETED / ExitCode0:0. Production вместе с оценкой и экспортом занял3.0625 A100 GPU-hours; CPU-постобработка84s. Optimizer gate, numerical/export QC и проверка frozen tensors пройдены. Старые ошибки20991836/20991837 относятся к восстановленной первой попытке smoke; production не падал.','', 'Все12 моделей проверены. Среднее ± выборочное SD по seeds10/20/30; меньше — лучше. Validation149 общий, выравнивание энергии только по train соответствующего N. TiAl RMSE относится к отсортированным ветвям, Ti₃Al W₁ — к распределению частот DOS.','', '| N | Маска | E, meV/атом | F, eV/Å | Stress, GPa | TiAl RMSE, THz | Ti₃Al W₁, THz | Ti₃Al L1 |','|---:|---|---:|---:|---:|---:|---:|---:|']
    for n,m in combos:
        g=result['groups'][n][m];lines.append('| '+n+' | '+m+' | '+' | '.join(f'{g[k]["mean"]:.6f} ± {g[k]["sample_SD"]:.6f}' for k,_ in specs)+' |')
    lines+=['','![Сравнение](grace-pbe-n694/comparison.png)','','## Выводы по трём seeds','',
      'При N694 обе маски улучшают F, stress и общий TiAl band RMSE во всех3 seeds, одновременно ухудшая Ti₃Al W₁ и L1 во всех3 seeds. Это устойчивое различие свойств при данном разбиении; увеличение train не даёт универсального улучшения.','',
      '- Full N694 против N200: F −4.99%, stress −21.68%, TiAl RMSE −8.82%; Ti₃Al W₁ +41.83%, L1 +12.85%. Средняя aligned E меняется лишь на−0.88%, знак изменения зависит от seed.',
      '- Radial N694 против N200: F −3.93%, stress −11.20%, TiAl RMSE −4.47%; Ti₃Al W₁ +34.08%, L1 +9.36%. Средняя aligned E практически не меняется (−0.11%).',
      '- Внутри N694 full лучше radial по F, stress и общему TiAl RMSE во всех3 seeds; средний выигрыш1.95%,6.71% и20.03%. Radial лучше по Ti₃Al W₁ и L1 во всех3 seeds; W₁ ниже на24.67%. Различие средней энергии небольшое относительно seed-разброса.',
      '- Из этих четырёх сочетаний full N694 — основной кандидат по validation F/stress и общей дисперсии TiAl; radial N200 остаётся лучшим по обеим метрикам DOS Ti₃Al. Единой лучшей модели по всем свойствам нет.','',
      'У full большой stress-разброс N200 исчезает в этой серии N694:0.06569/0.06628/0.06498GPa для seeds10/20/30. Это наблюдение по трём seeds, а не гарантия отсутствия неустойчивых запусков.','',
      'Общий TiAl RMSE скрывает локальное ухудшение: ошибка Γ-синглета у full растёт с0.03742±0.00705 до0.33545±0.09277THz, во всех3 seeds. У radial она уменьшается с0.56403±0.03036 до0.43090±0.06516THz. Поэтому улучшение интегрального RMSE не означает улучшения каждой моды.','',
      '![Спектры](grace-pbe-n694/spectra.png)','','## Дополнительные метрики','',
      '| N | Маска | Raw E, meV/атом | Ошибка Γ-синглета, THz | Ti₃Al W₁ после ASR, THz |','|---:|---|---:|---:|---:|']
    for n,m in combos:
        g=result['groups'][n][m];lines.append('| '+n+' | '+m+' | '+' | '.join(f'{g[k]["mean"]:.6f} ± {g[k]["sample_SD"]:.6f}' for k in ['E_raw_RMSE_meV_atom','TiAl_Gamma_singlet_error_THz','Ti3Al_DOS_ASR_W1_THz'])+' |')
    lines+=['','Raw E у full N694 имеет большой seed-разброс; выравнивание удаляет композиционный сдвиг, но не улучшает силы или stress. ASR практически не меняет Ti₃Al W₁ и его ранжирование.','','## Парные изменения N694 − N200','', 'Отрицательное изменение означает улучшение. Порядок seeds:10/20/30.','']
    for m,vals in result['paired_N694_minus_N200'].items():
        lines.append('### '+m+'\n')
        for k,_ in specs:lines.append('- '+k+': '+', '.join(f'{x:+.6f}' for x in vals[k]))
        lines.append('')
    assert all(c['phonons']['imaginary_modes_weighted']==c['Ti3Al']['mesh_imaginary']==c['Ti3Al']['ASR_band_imaginary']==0 for c in result['cases'].values())
    lines+=['','## Проверки и ограничения','', '50 эпох в обоих случаях:10000 updates для N200 и34700 для N694; вычислительный бюджет различается. Нельзя приписывать изменение только размеру данных отдельно от числа обновлений. SD характеризует только seed-разброс одного разбиения. Test149 и архивные49 не использованы, r²SCAN не включён.','', 'Ti₃Al: фиксированная архивная Cmcm-ячейка, DFT FORCE_SETS недоступен, поэтому сравнивается DOS. У всех12 моделей нет мод ниже−0.001THz на TiAl ASR mesh50³, Ti₃Al raw mesh14³ и проверенном Ti₃Al пути после ASR. Это не доказывает стабильность всей зоны Бриллюэна или длительной MD. Исходная сходимость DFT-релаксации Ti₃Al неизвестна.','', 'Условность реконструкции жидкостных родителей при утраченных траекториях и неопределённость пересечения с FM pretraining сохраняются. Происхождение PBE DFT energy field остаётся ранее отмеченным ограничением.','',f'Проверены SHA-256 {len(h)} собранных файлов,6 текущих финальных checkpoint/экспортов/Adam archives и {count} записей контроля параметров. Validation, TiAl RMSE, Ti₃Al W₁/L1/ASR, числа мнимых мод, средние, SD и парные изменения пересчитаны локально.','', 'Следующий приоритет: сохранить full N694 и radial N200 как основные контрастные модели; проверить покрытие Ti₃Al-состояний разрешённым train и отделить влияние числа updates контрольным опытом с одинаковым бюджетом. Новые обучения в этом шаге не запускались.']
    report=ROOT/'reports/30_grace_pbe_n694_results.md';report.write_text('\n'.join(lines)+'\n',encoding='utf8')
    state=load(ROOT/'state.json');state.update(status='GRACE_PBE_N694_comparison_complete',active_training_jobs=0,completed_training_jobs=46,fully_qualified_training_jobs=46)
    state['grace_pbe_n694'].update(status='complete',optimizer='PASS',comparison=str(out/'comparison.json'),report=str(report));state['current_gates'].update(GRACE_PBE_N694_comparison='PASS',GRACE_PBE_N694_optimizer='PASS')
    if 'reports/30_grace_pbe_n694_results.md' not in state['reports']:state['reports'].append('reports/30_grace_pbe_n694_results.md')
    state['next_actions']=['Retain GRACE full N694 for validation F/stress and TiAl bands, and radial N200 for Ti3Al DOS.','Investigate Ti3Al coverage and equal-update controls before attributing the N694 tradeoff to dataset size alone.','Keep r2SCAN paused and held-out test149/archival49 outside selection.']
    dump(ROOT/'state.json',state)
    matrix=load(ROOT/'protocol/experiment_matrix.yaml')
    for row in matrix['training_designs']:
        if row.get('architecture')=='GRACE-2L' and row.get('functional')=='PBE' and row.get('train_size')==694 and row.get('adaptation') in METHODS and row.get('training_seed') in [10,20,30]:
            name=f'PBE-{row["adaptation"]}-N694-seed{row["training_seed"]}'
            row.update(state='completed_qualified',blocked_by=[],results=str(out/'comparison.json'),final_checkpoint_hashes=result['cases'][name]['checkpoint_hashes'])
    dump(ROOT/'protocol/experiment_matrix.yaml',matrix)
    (ROOT/'reports/summary.md').write_text('# Ti–Al FM: текущее состояние\n\nВсе6 GRACE PBE N694 завершены и проверены. По трём seeds N694 улучшает F/stress и общий TiAl RMSE, но ухудшает DOS Ti₃Al у обеих масок. Full N694 — лучший по F/stress/TiAl среди сравниваемых GRACE; radial N200 — по Ti₃Al W₁/L1. У full N694 ухудшается Γ-синглет при улучшении общего RMSE. r²SCAN не включён.\n\n[Сравнение N694/N200](30_grace_pbe_n694_results.md) · [Все4 маски N200](28_grace_pbe_three_seed_results.md).\n',encoding='utf8')
    print(report);print(result['local_verification'])
if __name__=='__main__':main()
