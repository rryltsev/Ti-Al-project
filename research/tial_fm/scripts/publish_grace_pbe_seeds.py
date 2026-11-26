"""Publish only after all twelve PBE models are independently verified locally."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare_grace_pbe_seeds import ROOT,RUN,PARENT,MASK,TI3,load,dump,sha,METHODS
from compare_grace_n200 import recompute
from publish_ti3al_pbe import metrics as dos_metrics
def main():
    data=RUN/'collected-final';h=load(data/'COLLECTION_SHA256.json')
    for n,v in h.items():assert sha(data/n)==v,n
    remote=load(RUN/'final-remote-verification.json');assert remote['status']=='PASS'
    result=load(data/'comparison/summary.json');assert result['status']=='PASS' and len(result['cases'])==12
    assert not result['r2SCAN_tested'] and all(r['case']['XC']=='PBE' for r in result['cases'].values())
    subsets=load(ROOT/'datasets/nested-train-v1-20261106/subsets.json');ref=np.load(RUN/'package/reference_figure.npz');dft=np.loadtxt(RUN/'package/ti3al/source/Ti3Al_D019/total_dos.dat')
    spectra={};checks_count=0
    for name,r in result['cases'].items():
        c=r['case'];seed=c['seed'];method=c['method']
        base=data if seed!=10 else (PARENT if method in ['full','restricted'] else MASK)/'collected-final'
        w=base/'cases'/name;t=load(w/'training-summary.json');assert t['status']=='PASS' and t['hashes']==r['checkpoint_hashes']
        assert t['steps']==10000 and t['epochs']==50 and not t['test_access']
        for filename in ['numerical-QC.json','evaluation-summary.json','zero-step.json','checkpoint-path-check.json','harmonics/summary.json']:assert load(w/filename)['status']=='PASS'
        checks=[load(p) for p in (w/'checks').glob('*.json')]
        assert {v['steps'] for v in checks}=={1,*range(1000,10001,1000)}
        assert all(v['status']=='PASS' and not v['frozen_changed'] and v['optimizer_iterations']==v['steps'] for v in checks)
        checks_count+=len(checks)
        if seed!=10:
            v=remote['cases'][name];assert v['checkpoint']==t['hashes']
            assert v['training_summary_sha256']==sha(w/'training-summary.json')
            assert v['model_yaml_sha256']==sha(w/'seed'/str(seed)/'model.yaml')
        m=recompute(load(w/'per-structure.json'),load(w/'metrics.json'),subsets)
        for k,v in m['validation'].items():assert np.isclose(v,r['values'][k],rtol=1e-11,atol=1e-12)
        bands=np.load(w/'harmonics/PBE/bands.npz');assert np.array_equal(bands['qpoints'],ref['qpoints'])
        assert np.isclose(np.sqrt(np.mean((bands['frequencies']-ref['PBE_DFT_bands'])**2)),r['values']['TiAl_band_RMSE_THz'],atol=1e-12)
        mesh=np.load(w/'harmonics/PBE/mesh50.npz');assert mesh['weights'].sum()==125000
        assert int(np.sum(mesh['weights'][:,None]*(mesh['frequencies']<-.001)))==r['phonons']['imaginary_modes_weighted']
        folder=data/'ti3al' if seed!=10 else TI3/'collected';ti_name=name if seed!=10 else 'GRACE-'+method+'-N200-seed10'
        d=np.load(folder/'analysis'/ti_name/'dos_extended.npz');mm=dos_metrics(d['frequency'],d['dos'],dft[:,0],dft[:,1]/8)
        assert np.isclose(mm['DOS_Wasserstein_THz'],r['values']['Ti3Al_DOS_W1_THz'],atol=1e-12)
        assert np.isclose(mm['normalized_DOS_L1'],r['values']['Ti3Al_DOS_L1'],atol=1e-12)
        tm=np.load(folder/'analysis'/ti_name/'mesh14.npz');assert tm['weights'].sum()==14**3
        assert int(np.sum(tm['weights'][:,None]*(tm['frequencies']<-.001)))==r['Ti3Al']['mesh_imaginary']
        af=data/'ti3al' if seed!=10 else TI3/'collected-asr'
        ad=np.load(af/'analysis-asr'/ti_name/'dos.npz');am=dos_metrics(ad['frequency'],ad['dos'],dft[:,0],dft[:,1]/8)
        assert np.isclose(am['DOS_Wasserstein_THz'],r['values']['Ti3Al_DOS_ASR_W1_THz'],atol=1e-12)
        ab=np.load(af/'analysis-asr'/ti_name/'bands.npz')['frequencies'];assert int((ab<-.001).sum())==r['Ti3Al']['ASR_band_imaginary']
        spectra[name]=dict(TiAl=bands['frequencies'],frequency=d['frequency'],DOS=d['dos'])
    for method,g in result['groups'].items():
        for k,v in g.items():
            vals=[result['cases'][f'PBE-{method}-N200-seed{s}']['values'][k] for s in [10,20,30]]
            assert np.isclose(np.mean(vals),v['mean']) and np.isclose(np.std(vals,ddof=1),v['sample_SD'])
    assert checks_count==132
    result['local_verification']=dict(status='PASS',files=len(h),models=12,parameter_checks=checks_count,independent_validation_and_phonon_recalculation=True,ASR_DOS_recomputed=True,test_access=False)
    out=RUN/'comparison-local';out.mkdir(exist_ok=True);dump(out/'comparison.json',result)
    figs=ROOT/'reports/grace-pbe-three-seeds';figs.mkdir(exist_ok=True)
    specs=[('E_aligned_RMSE_meV_atom','Energy (meV/atom)'),('F_RMSE_eV_A','Force (eV/Å)'),('stress_RMSE_GPa','Stress (GPa)'),('TiAl_band_RMSE_THz','TiAl bands (THz)'),('Ti3Al_DOS_W1_THz','Ti3Al DOS W1 (THz)'),('Ti3Al_DOS_L1','Ti3Al DOS L1')]
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for ax,(key,title) in zip(axes.flat,specs):
        means=[result['groups'][m][key]['mean'] for m in METHODS];sd=[result['groups'][m][key]['sample_SD'] for m in METHODS]
        ax.bar(range(4),means,yerr=sd,capsize=4,color=['#167a9e','#d47729','#39966a','#8356a5'])
        for i,m in enumerate(METHODS):ax.scatter([i-.08,i,i+.08],result['groups'][m][key]['values'],color='black',s=12,zorder=4)
        ax.set_xticks(range(4),['Full','Restricted','+MLP','+Radial'],rotation=20);ax.set_title(title);ax.grid(axis='y',alpha=.2)
    fig.suptitle('GRACE PBE · N200 · seeds10/20/30 · mean ± sample SD')
    for ext in ['png','pdf']:fig.savefig(figs/f'comparison.{ext}',dpi=180)
    plt.close(fig)
    fig,(ax,ad)=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    x=ref['path_coordinate'];breaks=np.r_[0,np.flatnonzero(np.linalg.norm(np.diff(ref['qpoints'],axis=0),axis=1)>.05)+1,len(x)]
    for start,end in zip(breaks[:-1],breaks[1:]):
        for b in range(6):ax.plot(x[start:end],ref['PBE_DFT_bands'][start:end,b],color='black',ls='--',lw=1,label='PBE DFT' if start==0 and b==0 else None)
    ad.plot(dft[:,0],dft[:,1]/8,color='black',ls='--',lw=1.4,label='PBE DFT')
    for method,color in zip(METHODS,['#167a9e','#d47729','#39966a','#8356a5']):
        ss=[spectra[f'PBE-{method}-N200-seed{s}'] for s in [10,20,30]]
        arr=np.array([s['TiAl'] for s in ss]);mean=arr.mean(axis=0);sd=arr.std(axis=0,ddof=1)
        for start,end in zip(breaks[:-1],breaks[1:]):
            for b in range(6):
                ax.plot(x[start:end],mean[start:end,b],color=color,lw=1,label=method if start==0 and b==0 else None)
                ax.fill_between(x[start:end],(mean-sd)[start:end,b],(mean+sd)[start:end,b],color=color,alpha=.15)
        assert all(np.array_equal(s['frequency'],ss[0]['frequency']) for s in ss)
        arr=np.array([s['DOS'] for s in ss]);mean=arr.mean(axis=0);sd=arr.std(axis=0,ddof=1);f=ss[0]['frequency']
        ad.plot(f,mean,color=color,lw=1,label=method);ad.fill_between(f,mean-sd,mean+sd,color=color,alpha=.15)
    ax.set(title='TiAl · fixed PBE DFT cell',xlabel='Archived band-path coordinate',ylabel='Frequency (THz)',ylim=(-.2,9));ax.legend(fontsize=8,ncol=2)
    ad.set(title='Ti3Al · archived Cmcm cell',xlabel='Frequency (THz)',ylabel='DOS (modes / atom / THz)',xlim=(-.5,10.5));ad.legend(fontsize=8)
    fig.suptitle('GRACE PBE N200 · mean ± sample SD across seeds10/20/30')
    for ext in ['png','pdf']:fig.savefig(figs/f'spectra.{ext}',dpi=180)
    plt.close(fig)
    def fmt(g,k,digits=5):
        if k=='F_RMSE_eV_A':digits=6
        return f'{g[k]["mean"]:.{digits}f} ± {g[k]["sample_SD"]:.{digits}f}'
    lines=['# GRACE N200 PBE: четыре маски, три seeds','',
           'Все восемь повторов 20991808_0–7 и сравнение 20991810 завершились COMPLETED / ExitCode0:0. Все12 моделей (включая seed10): 10000 шагов/50 эпох, fp64; optimizer/mask, numerical/export QC, фононы и provenance PASS. r²SCAN не включён.','',
           'Среднее ± выборочное SD по seeds10/20/30. Validation149 одинаков для всех моделей. E aligned — остаточная композиционная поправка, вычисленная только на train200. Фононы TiAl — RMSE отсортированных частот в604 q-точках; Ti₃Al — W₁ распределения частот DOS. Эти две фононные метрики не взаимозаменяемы.','',
           '| Маска | E aligned, meV/atom | F, eV/Å | Stress, GPa | TiAl bands, THz | Ti₃Al W₁, THz |','|---|---:|---:|---:|---:|---:|']
    for method in METHODS:
        g=result['groups'][method];lines.append('| '+method+' | '+' | '.join(fmt(g,k) for k in ['E_aligned_RMSE_meV_atom','F_RMSE_eV_A','stress_RMSE_GPa','TiAl_band_RMSE_THz','Ti3Al_DOS_W1_THz'])+' |')
    lines+=['','![Метрики](grace-pbe-three-seeds/comparison.png)','','## Что устойчиво по seeds','',
            '- Full имеет наименьшую F RMSE и ошибку TiAl bands в каждом из трёх seeds. Относительно restricted средний выигрыш около2.85% по F и29.35% по TiAl bands.','- Restricted + radial имеет наименьший Ti₃Al W₁ во всех трёх seeds: в среднем на20.32% ниже full и на16.03% ниже restricted. Относительно full F RMSE выше лишь на0.87%, но ошибка TiAl bands выше на19.34%.','- Restricted имеет наименьшую среднюю aligned E; он лучше full в3/3 seeds, но различия между четырьмя масками малы и зависят от seed. Radial seed30 даёт минимальную E среди всех отдельных запусков; это не основание выбирать seed по удачному результату.','- +MLP улучшает Ti₃Al W₁ и L1 относительно restricted в3/3 seeds, но ухудшает F и TiAl bands в3/3. Его выигрыш локален для этого свойства, а не универсален.','',
            '## Энергия, Γ-мода и форма DOS','', '| Маска | E raw, meV/atom | Ошибка Γ-синглета TiAl, THz | Ti₃Al L1 | Ti₃Al W₁ после ASR, THz |','|---|---:|---:|---:|---:|']
    for method in METHODS:
        g=result['groups'][method];lines.append('| '+method+' | '+' | '.join(fmt(g,k) for k in ['E_raw_RMSE_meV_atom','TiAl_Gamma_singlet_error_THz','Ti3Al_DOS_L1','Ti3Al_DOS_ASR_W1_THz'])+' |')
    lines+=['','По L1 формы DOS лучший результат у +MLP, а не у radial. W₁ и L1 отражают разные аспекты спектра; единой победившей маски по всем метрикам нет. ASR практически не меняет W₁ GRACE и его ранжирование.','',
            '![Спектры](grace-pbe-three-seeds/spectra.png)','','## Stress full: существенный разброс','',
            'Full stress RMSE по seeds10/20/30: 0.07000 / 0.07525 / 0.10623 GPa. Поэтому его среднее0.08383±0.01958 выше, чем у restricted0.07694±0.00096. Full лучше restricted в seeds10/20, но хуже в seed30; прежний вывод о преимуществе full по stress, основанный на seed10, не устойчив.','',
            'Seed30 не исключён. Его численные, export и optimizer-проверки PASS; локальный пересчёт подтверждает ошибку. Самая большая отдельная validation-структура даёт лишь3.79% суммы квадратов stress-ошибок: результат не определяется единственным кадром. Причина повышенной чувствительности stress к seed отдельно не установлена.','',
            '## Проверки и ограничения','',f'Проверены SHA-256 {len(h)} собранных файлов и текущие хеши8 новых финальных checkpoint, экспортов и сжатых Adam archives. Локально независимо пересчитаны validation-метрики, TiAl RMSE, Ti₃Al W₁/L1, ASR W₁, число мнимых мод и mean/SD. Проверены {checks_count} записей контроля параметров; frozen tensors не менялись. Первые пять batches совпадают между масками одного seed; новые seeds20/30 различаются.','',
            'У всех12 моделей нет мод ниже−0.001THz на TiAl ASR mesh50³, Ti₃Al raw mesh14³ и проверенном Ti₃Al пути после ASR. Это не доказательство стабильности во всей зоне Бриллюэна или длительной MD. Ti₃Al — фиксированная архивная Cmcm-ячейка с неизвестной сходимостью исходной релаксации; DFT FORCE_SETS недоступен, поэтому DFT-ошибка ветвей Ti₃Al не заявляется. Три seeds оценивают вариацию обучения при одном разбиении, а не полную неопределённость.','',
            'Test149 и архивные49 не использованы. Гарантия группового разделения жидкостных родителей остаётся условной при утраченных траекториях; пересечение с FM pretraining неизвестно. Для PBE остаётся ранее отмеченное ограничение происхождения исходного DFT energy field.','',
            'Следующий приоритет: PBE GRACE full и restricted + radial на N694 по трём seeds. Restricted оставить контрольной точкой по энергии/stress; +MLP сохраняет интерес для формы Ti₃Al DOS. N694 не должен считаться автоматически лучше N200: обе фазы и validation проверяются заново. Новые обучения в этом шаге не запускались.','',
            'Машиночитаемые результаты: `runs/'+RUN.name+'/comparison-local/comparison.json`; парные разности относительно full сохранены в paired_minus_full.']
    text='\n'.join(lines)+'\n'
    report=ROOT/'reports/28_grace_pbe_three_seed_results.md';report.write_text(text,encoding='utf8')
    state=load(ROOT/'state.json');state.update(status='GRACE_PBE_three_seed_comparison_complete',active_training_jobs=0,completed_training_jobs=40,fully_qualified_training_jobs=40)
    state['grace_pbe_repeats'].update(status='complete',comparison=str(out/'comparison.json'),report=str(report))
    state['current_gates']['GRACE_PBE_three_seed_comparison']='PASS'
    if 'reports/28_grace_pbe_three_seed_results.md' not in state['reports']:state['reports'].append('reports/28_grace_pbe_three_seed_results.md')
    dump(ROOT/'state.json',state)
    state['next_actions']=['Prioritize PBE GRACE full and restricted_radial N694 three-seed controls; retain N200 comparisons across both phases.','Investigate seed sensitivity of GRACE full stress; keep r2SCAN and test149 outside new selection work.']
    dump(ROOT/'state.json',state)
    matrix=load(ROOT/'protocol/experiment_matrix.yaml')
    for row in matrix['training_designs']:
        if row.get('architecture')=='GRACE-2L' and row.get('functional')=='PBE' and row.get('train_size')==200 and row.get('training_seed') in [20,30]:
            name=f'PBE-{row["adaptation"]}-N200-seed{row["training_seed"]}'
            row.update(state='completed_qualified',blocked_by=[],results=str(out/'comparison.json'),final_checkpoint_hashes=result['cases'][name]['checkpoint_hashes'])
    dump(ROOT/'protocol/experiment_matrix.yaml',matrix)
    (ROOT/'reports/summary.md').write_text('# Ti–Al FM: текущее состояние\n\nGRACE PBE N200: все12 моделей четырёх масок × seeds10/20/30 завершены и проверены. Full лучше F и TiAl-фононов; +radial лучше Ti₃Al W₁; restricted лучше средней энергии/stress; +MLP лучше Ti₃Al L1. У full stress выражен seed-разброс. r²SCAN не включён в новые тесты.\n\n[Три seeds GRACE PBE](28_grace_pbe_three_seed_results.md) · [Ti₃Al: сравнение моделей](26_ti3al_pbe_phonons.md) · [DPA4 N694](18_final_n694_grace_a100.md).\n',encoding='utf8')
    print(report);print(result['local_verification'])
if __name__=='__main__':main()
