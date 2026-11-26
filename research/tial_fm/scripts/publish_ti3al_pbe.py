"""Independent local DOS metrics, seed summary and Ti3Al report."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare_ti3al_pbe import ROOT,RUN,load,dump,sha

def metrics(freq,dos,df,dd):
    r=np.interp(freq,df,dd,left=0,right=0)
    area=np.trapezoid(dos,freq);rarea=np.trapezoid(r,freq)
    c=np.r_[0,np.cumsum((dos[1:]+dos[:-1])/2*np.diff(freq))]/area
    cr=np.r_[0,np.cumsum((r[1:]+r[:-1])/2*np.diff(freq))]/rarea
    return dict(DOS_integral_per_atom=float(area),DFT_tabulated_integral_per_atom=float(rarea),normalized_DOS_L1=float(np.trapezoid(abs(dos/area-r/rarea),freq)),DOS_Wasserstein_THz=float(np.trapezoid(abs(c-cr),freq)),DOS_peak_THz=float(freq[np.argmax(dos)]))

def main():
    data=RUN/'collected';out=RUN/'comparison-local';out.mkdir(exist_ok=True)
    h=load(data/'COLLECTION_SHA256.json')
    for n,v in h.items():assert sha(data/n)==v,n
    extra=RUN/'collected-asr';eh=load(extra/'COLLECTION_SHA256.json')
    for n,v in eh.items():assert sha(extra/n)==v,n
    asrs=load(extra/'analysis-asr/summary.json');assert asrs['status']=='PASS'
    summary=load(data/'analysis/summary.json');assert summary['status']=='PASS' and len(summary['cases'])==16
    df=np.loadtxt(RUN/'package/source/Ti3Al_D019/total_dos.dat')
    pd=np.loadtxt(RUN/'package/source/Ti3Al_D019/projected_dos.dat')
    results={};groups={}
    for name,c in summary['cases'].items():
        d=np.load(data/'analysis'/name/'dos_extended.npz');v=metrics(d['frequency'],d['dos'],df[:,0],df[:,1]/8)
        for k,x in v.items():assert np.isclose(x,c['metrics'][k],rtol=1e-11,atol=1e-12),(name,k)
        mesh=np.load(data/'analysis'/name/'mesh14.npz')
        assert mesh['weights'].sum()==14**3 and mesh['frequencies'].shape[1]==24
        assert int(np.sum(mesh['weights'][:,None]*(mesh['frequencies']<-.001)))==c['mesh14']['imaginary_modes_weighted']
        bands=np.load(data/'analysis'/name/'bands.npz')
        for scale,key in [(.5,'half'),(2.,'double')]:
            assert np.isclose(np.sqrt(np.mean((bands[key]-bands['main'])**2)),c['displacement_convergence'][str(scale)]['band_RMSE_to_001A_THz'])
        result=dict(**v,imaginary_modes_weighted=c['mesh14']['imaginary_modes_weighted'],min_frequency_THz=c['mesh14']['min_THz'],max_residual_force_eV_A=c['max_residual_force_eV_A'],max_residual_stress_GPa=float(np.max(abs(np.array(c['residual_stress_GPa'])))),half_step_band_RMSE_THz=c['displacement_convergence']['0.5']['band_RMSE_to_001A_THz'],double_step_band_RMSE_THz=c['displacement_convergence']['2.0']['band_RMSE_to_001A_THz'],ASR_band_RMSE_change_THz=c['ASR_sensitivity']['band_RMSE_change_THz'],mesh20_Wasserstein_THz=c['mesh20_metrics']['DOS_Wasserstein_THz'])
        a=asrs['cases'][name];assert a['force_constants_sha256']==sha(data/'analysis'/name/'force_constants_ASR.npy')
        ad=np.load(extra/'analysis-asr'/name/'dos.npz');av=metrics(ad['frequency'],ad['dos'],df[:,0],df[:,1]/8)
        for k,x in av.items():assert np.isclose(x,a['ASR_DOS_metrics'][k],rtol=1e-11,atol=1e-12)
        ab=np.load(extra/'analysis-asr'/name/'bands.npz')['frequencies']
        assert int((ab<-.001).sum())==a['ASR_band_imaginary_count']
        result.update(Gamma_raw_min_THz=float(min(a['raw_Gamma_THz'])),Gamma_ASR_min_THz=float(min(a['ASR_Gamma_THz'])),ASR_DOS_Wasserstein_THz=av['DOS_Wasserstein_THz'],ASR_band_imaginary_count=a['ASR_band_imaginary_count'])
        results[name]=dict(case=c['case'],metrics=result)
        key=f'{c["case"]["backend"]} {c["case"]["method"]} N{c["case"]["N"]}'
        groups.setdefault(key,[]).append(name)
    grouped={}
    for key,names in groups.items():
        grouped[key]=dict(models=names,seeds=[results[n]['case']['seed'] for n in names],metrics={k:dict(mean=float(np.mean([results[n]['metrics'][k] for n in names])),sample_std=float(np.std([results[n]['metrics'][k] for n in names],ddof=1)) if len(names)>1 else None) for k in next(iter(results.values()))['metrics']})
    audit=load(data/'prepared/overlap-screen.json')
    result=dict(status='PASS',cases=results,groups=grouped,checks=dict(files=len(h)+len(eh),models=16,independent_DOS_recalculation=True,imaginary_modes_recomputed=True,ASR_DOS_recomputed=True,r2SCAN_tested=False,heldout_predictions=False),overlap_screen=audit)
    dump(out/'comparison.json',result)
    figdir=ROOT/'reports/ti3al-pbe-phonons';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(3,3,figsize=(14,10),constrained_layout=True)
    colors={'full':'#167a9e','lora16':'#d47729','restricted':'#d47729','restricted_mlp':'#39966a','restricted_radial':'#8356a5'}
    for row,(backend,N) in enumerate([('DPA4',200),('DPA4',694),('GRACE',200)]):
        for col,kind in enumerate(['total','Al','Ti']):
            ax=axs[row,col]
            rf,rd=(df[:,0],df[:,1]/8) if kind=='total' else (pd[:,0],pd[:,1:3].sum(axis=1)/8 if kind=='Al' else pd[:,3:].sum(axis=1)/8)
            ax.plot(rf,rd,color='black',lw=1.5,ls='--',label='PBE DFT')
            for group,g in grouped.items():
                cs=results[g['models'][0]]['case']
                if cs['backend']!=backend or cs['N']!=N:continue
                arrays=[];grid=np.linspace(-1,11,2401)
                for name in g['models']:
                    p=data/'analysis'/name
                    z=np.load(p/('dos_extended.npz' if kind=='total' else 'projected_dos.npz'))
                    arrays.append(np.interp(grid,z['frequency'],z['dos'] if kind=='total' else z[kind],left=0,right=0))
                arr=np.array(arrays);mean=arr.mean(axis=0);method=cs['method'];color=colors[method]
                ax.plot(grid,mean,color=color,lw=1.1,label=method)
                if len(arr)>1:
                    sd=arr.std(axis=0,ddof=1);ax.fill_between(grid,mean-sd,mean+sd,color=color,alpha=.2)
            ax.set(title=f'{backend} N={N} · {kind}',xlabel='Frequency (THz)',ylabel='Modes / atom / THz',xlim=(-.5,10.5));ax.grid(alpha=.15)
            if col==0:ax.legend(fontsize=7)
    fig.suptitle('Ti₃Al archived Cmcm cell · PBE · 2×2×2 · δ=0.01 Å · tetrahedron mesh14³\nDPA4: mean ± sample SD, 3 seeds; GRACE: seed10')
    for ext in ['png','pdf']:fig.savefig(figdir/f'DOS.{ext}',dpi=170)
    plt.close(fig)
    lines=['# Ti₃Al: проверка PBE-потенциалов по DFT-DOS','',
           'Рассчитаны все 16 завершённых PBE-моделей: DPA4 N200/N694 LoRA/full × seeds10/20/30 и GRACE N200 full/restricted/+MLP/+radial seed10. r²SCAN исключён из новых тестов по указанию пользователя. Новые обучения и DFT не запускались.','',
           '## Что воспроизводится','',
           'Фиксированная геометрия из архива Ti3Al_D019: 8 атомов в ячейке, суперячейка2×2×2 (64 атома), восемь сохранённых смещений ±0.01 Å, архивные массы Al26.981539/Ti47.867. Геометрия имеет Cmcm(63), поэтому имя каталога D019 не является подтверждением идеальной D0₁₉. Предварительная DFT-релаксация не восстановлена.','',
           'DFT-референс — сохранённые total_dos.dat и projected_dos.dat, тетраэдрический метод, фактическая сетка14³. VASP: PBE, Al/Ti_sv PAW, ENCUT480 eV, kmesh4³, EDIFF1e-8. Loose mesh.conf не использован. FORCE_SETS отсутствует локально, исходный каталог u3173 недоступен u3129; пользователь подтвердил отсутствие копии. Поэтому ошибок фононных ветвей относительно DFT здесь нет.','',
           'Силовые константы MLIP получены traditional solver без дополнительной ASR-правки; влияние ASR рассчитано отдельно. Использован Phonopy4.8.2, архив создан в2.40.0. Без исходных DFT-сил нельзя независимо воспроизвести исходную обработку force constants или доказать побитовое соответствие версий. Нормировка DOS — моды/(атом·THz), ожидаемый интеграл3. Архивная полная DOS имеет интеграл2.98745, сумма projected DOS — 3.00148; они сохранены как отдельные референсы без подгонки.','',
           '## Сравнение полной DOS','',
           'W₁ — расстояние Вассерштейна по частоте (THz), меньше лучше. Перед вычислением W₁ и L1 каждая DOS нормирована по своему интегралу; исходные ненормированные кривые и интегралы сохранены. Метрики используют расширенную сетку −5…20 THz, чтобы не потерять хвосты или отрицательные частоты. DPA4: среднее ± выборочное SD по трём seeds; GRACE: одно значение.','',
           '| Модель | W₁, THz | L1 нормированной DOS | min ν, THz | Мнимые моды, взвешенное число |','|---|---:|---:|---:|---:|']
    def fmt(v,sd=True):return f'{v["mean"]:.4f} ± {v["sample_std"]:.4f}' if v['sample_std'] is not None and sd else f'{v["mean"]:.4f}'
    for key,g in grouped.items():
        v=g['metrics'];counts=[results[n]['metrics']['imaginary_modes_weighted'] for n in g['models']]
        lines.append(f'| {key} | {fmt(v["DOS_Wasserstein_THz"])} | {fmt(v["normalized_DOS_L1"])} | {fmt(v["min_frequency_THz"])} | {", ".join(map(str,counts))} |')
    lines+=['','GRACE restricted + radial имеет минимальный W₁ (0.2317 THz): на18.2% ниже full и на16.4% ниже restricted. По L1 лучший из GRACE — +MLP (0.4007), практически рядом с restricted (0.4017); +radial даёт0.4106. Следовательно, преимущество radial относится к частотному распределению по W₁, а не ко всем возможным мерам сходства формы DOS.', '',
            'У DPA4 N200 LoRA лучше full по W₁ во всех трёх seeds. На N694 порядок меняется: full лучше LoRA во всех трёх seeds. При увеличении train LoRA ухудшается с0.3565 до0.4633 THz (+30.0%); для full изменение0.4016→0.4128 THz значительно меньше. Увеличение train не гарантирует улучшения этого свойства. Среди данных pipelines все четыре GRACE имеют меньший W₁, чем DPA4, но это не доказательство превосходства архитектуры: различаются pretraining и рецепты, GRACE имеет только один seed.', '',
            '![DOS](ti3al-pbe-phonons/DOS.png)','','Вклады Al/Ti показаны в расчёте на один атом всей системы, поэтому их сумма равна полной projected DOS. На рисунке у всех моделей заметен сдвиг высокочастотной группы относительно DFT; более низкий W₁ не означает полного совпадения спектра. Отрицательные частоты не удалены из численных метрик; если они выходят за показанный диапазон рисунка, это видно из таблицы и сохранённых массивов.','',
            '## Численная чувствительность и остаточные силы','',
            '| Модель | Fmax, eV/Å | max stress, GPa | δ0.005 vs0.01, THz | δ0.02 vs0.01, THz | ASR change, THz | W₁ mesh20³, THz |','|---|---:|---:|---:|---:|---:|---:|']
    for key,g in grouped.items():
        v=g['metrics'];keys=['max_residual_force_eV_A','max_residual_stress_GPa','half_step_band_RMSE_THz','double_step_band_RMSE_THz','ASR_band_RMSE_change_THz','mesh20_Wasserstein_THz']
        lines.append('| '+key+' | '+' | '.join(fmt(v[k]) for k in keys)+' |')
    lines+=['','Ошибки при изменении δ и ASR — RMSE по одним и тем же частотно отсортированным ветвям на архивном пути. Это численная чувствительность MLIP, не ошибка к DFT. Mesh20³ меняет только интегрирование MLIP-DOS; DFT остаётся на14³. Сходимость по размеру суперячейки не проверена. Остаточные силы/stress характеризуют MLIP в фиксированной архивной ячейке, не новую релаксацию. У GRACE Fmax0.097–0.149 eV/Å: ячейку нельзя считать равновесной для этих моделей, несмотря на более близкую DOS.','',
            '## Акустические моды Γ и ASR','',
            'Сетка14³ без Gamma-centering не содержит Γ. Поэтому отсутствие мнимых мод на ней не исключает отрицательных частот в Γ. У DPA4 raw минимум Γ составляет −0.055…−0.159 THz. После ASR акустические частоты порядка10⁻⁷ THz, и мод ниже −0.001 THz на проверенном пути нет. У GRACE raw Γ около−0.00015 THz. Raw результаты сохранены; ASR не заменяет их молча. Стабильность полного Brillouin zone или длительной MD этим тестом не доказана.', '',
            '| Модель | min Γ raw, THz | min Γ ASR, THz | W₁ после ASR, THz |','|---|---:|---:|---:|']
    for key,g in grouped.items():
        v=g['metrics'];lines.append(f'| {key} | {fmt(v["Gamma_raw_min_THz"])} | {v["Gamma_ASR_min_THz"]["mean"]:.2e} | {fmt(v["ASR_DOS_Wasserstein_THz"])} |')
    lines+=['','Ранжирование средних W₁ после ASR сохраняется; меняются небольшие детали отдельных seeds. Проверки δ и q-mesh также не объясняют основные различия между группами.','',
            '## Provenance и область выводов','',f'Хеши {len(h)+len(eh)} собранных файлов проверены локально, DOS-метрики и число мнимых мод пересчитаны независимо, включая дополнительную ASR-DOS. Все16 моделей закреплены хешами; никаких изменений весов. Для сравнения с train/validation проверены {len(audit["candidates"])} структуры того же состава Ti:Al=3:1 по периодическим парным распределениям и размеру примитивной ячейки. Минимальное относительное различие парных гистограмм0.976; ближайшие структуры имеют216 атомов в найденной примитивной ячейке. Подробности в prepared/overlap-screen.json. Это screening, а не доказательство независимости от деформированных эквивалентов. Пересечение с pretraining FM неизвестно; тест Ti₃Al обозначен development property diagnostic. Test149/архивные49 не оценивались.', '',
            'Численные результаты: `runs/'+RUN.name+'/comparison-local/comparison.json`. Полный сбор содержит силы для25 геометрий каждой модели, force constants, bands, DOS/PDOS, mesh14, проверки δ и ASR, хеши и Slurm-логи.']
    report=ROOT/'reports/26_ti3al_pbe_phonons.md';report.write_text('\n'.join(lines)+'\n',encoding='utf8')
    state=load(ROOT/'state.json');state['ti3al_phonons'].update(status='complete',comparison=str(out/'comparison.json'),report=str(report),models=16,ASR_job='20991481',ASR_status='PASS',DFT_force_sets='unavailable; user confirms no copy')
    if 'reports/26_ti3al_pbe_phonons.md' not in state['reports']:state['reports'].append('reports/26_ti3al_pbe_phonons.md')
    state['next_actions']=['Interpret PBE Ti3Al DOS together with TiAl phonons; verify GRACE seed variability before model selection.','Keep r2SCAN excluded from new property tests until user resumes it.']
    dump(ROOT/'state.json',state)
    suite=load(ROOT/'protocol/property_test_suite.json')
    for test in suite['tests']:
        if test['id']=='Ti3Al-archived-Cmcm-phonons':test.update(status='complete',comparison=str(out/'comparison.json'))
    dump(ROOT/'protocol/property_test_suite.json',suite)
    (ROOT/'reports/summary.md').write_text('# Ti–Al FM: текущее состояние\n\nДобавлен Ti₃Al: все16 PBE-потенциалов проверены по полной и парциальной DFT-DOS в фиксированной архивной геометрии Cmcm. r²SCAN исключён из новых тестов.\n\n[Ti₃Al PBE: результаты](26_ti3al_pbe_phonons.md) · [Четыре маски GRACE N200](25_grace_mask_ablation_results.md) · [DPA4 N694 и A100 precision](18_final_n694_grace_a100.md).\n',encoding='utf8')
    print(json.dumps(grouped,indent=2));print(report)
if __name__=='__main__':main()
