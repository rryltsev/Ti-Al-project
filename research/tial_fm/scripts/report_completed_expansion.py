"""Figures and a human-readable three-seed/size comparison from verified results."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/dpa4-replicates-full-20261106-02'
OLD=ROOT/'runs/dpa4-n200-pilot-20261106-02';OUT=ROOT/'reports/multiseed-fulltrain-20261107'
NAME={'lora16':'LoRA16','full':'Full FT'};COLOR={'lora16':'#0072B2','full':'#D55E00'}

def fmt(s,digits=3):
    return f"{s['mean']:.{digits}f}"+(f" ± {s['sample_SD']:.{digits}f}" if s['sample_SD'] is not None else '')

def main():
    d=json.loads((RUN/'comparison-20261107/comparison.json').read_text());assert d['status']=='PASS'
    g=d['seed_summary'];cases=d['cases'];OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    metrics=[('E_aligned_meV_atom','Aligned energy RMSE\n(meV/atom)',1),('F_component_eV_A','Force RMSE\n(meV/Å)',1000),
             ('stress_component_GPa','Stress RMSE\n(MPa)',1000),('phonon_sorted_RMSE_THz','Phonon sorted RMSE\n(THz)',1),('Gamma_singlet_THz','Γ c-singlet\n(THz)',1)]
    fig,axes=plt.subplots(2,5,figsize=(15,6.6),constrained_layout=True)
    for row,xc in enumerate(['PBE','r2SCAN']):
        for col,(metric,title,scale) in enumerate(metrics):
            ax=axes[row,col];ax.axvspan(1.5,3.5,color='#f1f3f4',zorder=0)
            for ix,(n,method) in enumerate([(200,'lora16'),(200,'full'),(694,'lora16'),(694,'full')]):
                s=g[f'{xc}-{method}-N{n}'][metric];color=COLOR[method]
                if n==200:
                    ax.errorbar(ix,s['mean']*scale,yerr=s['sample_SD']*scale,fmt='_',ms=19,lw=1.5,capsize=5,color=color,zorder=4)
                    ax.scatter(ix+np.array([-.09,0,.09]),np.array(s['values'])*scale,s=30,color=color,edgecolor='white',linewidth=.5,zorder=5)
                else:ax.scatter(ix,s['mean']*scale,s=55,marker='D',color=color,edgecolor='white',linewidth=.6,zorder=5)
            if metric=='Gamma_singlet_THz':
                target=cases[f'{xc}-full-N200-seed10']['reference_Gamma']['singlet_c_THz']
                ax.axhline(target,color='#222222',ls=':',lw=1.2)
                ax.text(.98,.96,f'DFT {target:.3f}',ha='right',va='top',transform=ax.transAxes,fontsize=8)
            ax.set(xlim=(-.4,3.4),xticks=range(4),xticklabels=['LoRA\n200','Full\n200','LoRA\n694','Full\n694'],title=title)
            ax.grid(axis='y',alpha=.2);ax.margins(y=.18)
            if col==0:ax.set_ylabel('r²SCAN' if xc=='r2SCAN' else xc,fontsize=12,fontweight='bold')
    fig.suptitle('DPA4-Neo: same validation149 and fixed DFT phonon reference\nN200: dots = seeds 10/20/30; bars = mean ± sample SD. Shaded N694: seed10 only.',fontsize=13)
    for ext in ['png','pdf']:fig.savefig(OUT/f'metrics-seed-spread.{ext}',dpi=180)
    plt.close(fig)
    ref=np.load(OLD/'package/reference_figure.npz');q=ref['qpoints'];x=ref['path_coordinate'];ticks=[]
    for label,v in [(r'$\Gamma$',[0,0,0]),('K',[1/3,1/3,0]),('M',[.5,0,0]),('A',[0,0,.5])]:
        ticks += [(float(x[i]),label) for i in np.where(np.max(np.abs(q-v),axis=1)<1e-9)[0]]
    ticks=sorted(set(ticks));fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
    dos_results={}
    for row,xc in enumerate(['PBE','r2SCAN']):
        df=np.load(OLD/'collected/harmonics'/xc/'DFT/bands.npz')['frequencies']
        dfdos=np.load(OLD/'collected/harmonics'/xc/'DFT/dos.npz')
        for col,n in enumerate([200,694]):
            ax=axes[row,col];high=float(df.max());seeds=[10,20,30] if n==200 else [10]
            for mode in range(6):ax.plot(x,df[:,mode],color='#222222',lw=1.15,label='DFT' if mode==0 else None,zorder=4)
            for method in ['lora16','full']:
                bands=[];dos_l1=[]
                for seed in seeds:
                    c=cases[f'{xc}-{method}-N{n}-seed{seed}'];p=Path(c['case_folder'])/'evaluation/harmonics';a=np.load(p/'bands.npz')
                    assert np.array_equal(a['qpoints'],q);bands.append(a['frequencies'])
                    dd=np.load(p/'dos.npz');assert abs(np.trapezoid(dd['dos'],dd['frequency'])-3)<1e-5
                    grid=np.arange(min(dd['frequency'].min(),dfdos['frequency'].min()),max(dd['frequency'].max(),dfdos['frequency'].max())+.002,.002)
                    delta=np.interp(grid,dd['frequency'],dd['dos'],left=0,right=0)-np.interp(grid,dfdos['frequency'],dfdos['dos'],left=0,right=0)
                    dos_l1.append(float(np.trapezoid(np.abs(delta),grid)))
                dos_results[f'{xc}-{method}-N{n}']=dict(values=dos_l1,mean=float(np.mean(dos_l1)),sample_SD=float(np.std(dos_l1,ddof=1)) if n==200 else None)
                a=np.array(bands);mean=a.mean(axis=0);high=max(high,float(a.max()))
                for mode in range(6):
                    if n==200:ax.fill_between(x,a[:,:,mode].min(axis=0),a[:,:,mode].max(axis=0),color=COLOR[method],alpha=.18,linewidth=0)
                    ax.plot(x,mean[:,mode],color=COLOR[method],ls='--' if method=='full' else '-',lw=1,label=NAME[method] if mode==0 else None,zorder=5)
            ax.set(title=f"{'r²SCAN' if xc=='r2SCAN' else xc}: N={n}"+(' (3 seeds)' if n==200 else ' (seed10)'),
                   xticks=[t[0] for t in ticks],xticklabels=[t[1] for t in ticks],xlim=(x.min(),x.max()),ylim=(-.9,high+.25),ylabel='Frequency (THz)')
            for v,_ in ticks:ax.axvline(v,color='#dddddd',lw=.6,zorder=0)
            ax.grid(axis='y',alpha=.18);ax.axhline(0,color='#777777',lw=.6);ax.legend(loc='lower left',ncol=3,frameon=False,fontsize=9)
    fig.suptitle('α-Ti: matched DFT geometries, 4×4×3, ±0.01 Å, identical force processing\nLeft: mean sorted bands and min–max seed envelopes. All 16 models stable on the tested 50³ mesh.',fontsize=13)
    for ext in ['png','pdf']:fig.savefig(OUT/f'phonons-N200-N694.{ext}',dpi=180)
    plt.close(fig)
    (OUT/'DOS-L1.json').write_text(json.dumps(dos_results,indent=2)+'\n')

    text='''# DPA4: N200 по трём seeds и полный train694

Проверено 7 ноября 2026 года. **Все 16 моделей завершены**: 12 N200 (PBE/r²SCAN × LoRA16/full FT × seeds10/20/30) и 4 N694 (тот же набор методов/XC, seed10). Новые GPU arrays20988583 и20988587, CPU QC20988585/20988588 и сводки20988586/20988589 завершены с exit0. Очередь этих заданий пуста.

Собраны 2112 новых файлов с проверкой SHA256, включая 12 финальных merged weights. Вновь вычисленные локально метрики всех12 новых моделей совпали с кластерными. Проверены хеши final/native/export/QC и260 checkpoint-проверок, включая40 ранее завершённого seed10. Zero-step, frozen weights, маски, native/export и конечные разности E/F/virial — PASS. Все seeds включены; выбор лучшего seed или checkpoint не выполнялся.

## Validation и разброс обучения

В таблицах RMSE; для N200 показаны **среднее ± sample SD между тремя training seeds**, не standard error и не доверительный интервал по структурам. N694 имеет только seed10, поэтому его разброс не определён. Validation149 одна и та же у всех16 моделей, включает23 консервативные группы состава. Повторы seeds не превращают эти149 структур в447 независимых наблюдений.

E aligned — энергия после двух элементных сдвигов, подобранных исключительно на train соответствующего запуска. Это дополнительная gauge-коррекция для метрики; raw E приведена отдельно. Силы объединяются по атомам и компонентам, stress — по шести симметричным компонентам. Checkpoint фиксирован после50 nominal epochs:10000 шагов для N200 и34700 для N694.

| N | XC | Метод | E raw, meV/atom | E aligned, meV/atom | F, eV/Å | σ, GPa |
|---:|---|---|---:|---:|---:|---:|
'''
    for n in [200,694]:
        for xc in ['PBE','r2SCAN']:
            for method in ['lora16','full']:
                s=g[f'{xc}-{method}-N{n}'];text+=f"| {n} | {xc} | {NAME[method]} | {fmt(s['E_raw_meV_atom'])} | {fmt(s['E_aligned_meV_atom'])} | {fmt(s['F_component_eV_A'],5)} | {fmt(s['stress_component_GPa'],5)} |\n"
    text+='''
**N200:** LoRA имеет меньшие aligned-E и F RMSE для каждого из трёх парных seeds обоих XC. Среднее преимущество по F ≈2.47% для PBE и2.46% для r²SCAN; оно заметно больше наблюдаемого seed-разброса. При этом средняя raw E ошибка меньше у full FT: утверждение об энергии зависит от явного train-only выравнивания. По stress full FT лучше в2/3 seeds PBE и3/3 seeds r²SCAN, но величина разницы меняется сильнее, чем разница F.

Парные разности Δ=LoRA−full при одинаковом seed; отрицательная RMSE-разность благоприятна LoRA:

| XC | Метрика | Средняя Δ ± SD | Диапазон Δ по seeds | Seeds с меньшей ошибкой LoRA / full |
|---|---|---:|---:|---:|
'''
    for xc in ['PBE','r2SCAN']:
        pair=d['paired_LoRA_minus_full'][f'{xc}-N200']
        for metric,label,digits in [('E_aligned_meV_atom','E aligned, meV/atom',4),('F_component_eV_A','F, eV/Å',6),('stress_component_GPa','σ, GPa',6),('phonon_sorted_RMSE_THz','Spectral RMSE, THz',4),('Gamma_singlet_abs_error_THz','Γ singlet abs error, THz',4)]:
            s=pair[metric];text+=f"| {xc} | {label} | {fmt(s,digits)} | [{s['min']:.{digits}f}; {s['max']:.{digits}f}] | {s['LoRA_lower_count']} / {s['full_lower_count']} |\n"
    text+='''
**N694, seed10:** full FT имеет меньшие точечные aligned-E/stress RMSE; LoRA сохраняет небольшое преимущество по F (≈0.74–0.75%). Парный bootstrap2000 повторов по23 целым validation-группам даёт интервалы ΔF: PBE[−0.000844;−0.000066]eV/Å, r²SCAN[−0.000991;−0.000058]eV/Å. Для ΔE aligned интервалы PBE[−0.0229;+0.1571] и r²SCAN[−0.0561;+0.2228]meV/atom включают0; stress-интервалы также включают0. Поэтому преимущество full FT по этим двум validation-метрикам пока точечное. Эти bootstrap-интервалы условны для seed10 и не заменяют недостающие training seeds N694.

Validation содержит133 liquid/relaxation,7 γ-TiAl,7 β-Ti/bcc и2 α-Ti/hcp; общее качество в основном отражает жидкостный пул. Метрики семейств, элементов, MAE, p95/p99/max и индивидуальные residuals сохранены в исходных metrics/per_structure файлах. Три training seeds недостаточны для общего заявления о превосходстве метода за пределами этого протокола.

![Метрики и seed-разброс](multiseed-fulltrain-20261107/metrics-seed-spread.png)

## Фононы и идентификация мод

Одинаковые XC-specific фиксированные DFT-геометрии α-Ti, supercell4×4×3, смещения±0.01Å, масса47.867, central-pair/mean-drift/symfc/ASR. ML-релаксации нет. Spectral RMSE рассчитана на604 одинаковых q-точках пути Γ–K–M–Γ–A. Частотно упорядоченная ошибка отдельно сверена с eigenvector-overlap assignment и проекторами вырожденных подпространств; она не выдаётся за непрерывное отслеживание конкретной ветви.

DFT Γ doublet/singlet: **PBE3.964/5.408THz; r²SCAN4.587/5.189THz**.

| N | XC | Метод | Sorted RMSE, THz | Overlap-matched RMSE, THz | Γ doublet, THz | Γ c-singlet, THz | Средняя абсолютная ошибка синглета, THz |
|---:|---|---|---:|---:|---:|---:|---:|
'''
    for n in [200,694]:
        for xc in ['PBE','r2SCAN']:
            for method in ['lora16','full']:
                s=g[f'{xc}-{method}-N{n}'];values=[fmt(s[k]) for k in ['phonon_sorted_RMSE_THz','phonon_overlap_RMSE_THz','Gamma_doublet_THz','Gamma_singlet_THz','Gamma_singlet_abs_error_THz']]
                text+=f"| {n} | {xc} | {NAME[method]} | "+' | '.join(values)+' |\n'
    text+='''
На N200 **full FT лучше по общей spectral RMSE во всех трёх seeds обоих XC**, а LoRA лучше по абсолютной ошибке c-singlet во всех трёх. Для PBE средняя частота синглета LoRA5.403 близка кDFT5.408, но ошибка не равна0.005: средняя абсолютная ошибка отдельных seeds0.063THz. Для r²SCAN соответствующая абсолютная ошибка0.093THz. Усреднение частот может компенсировать ошибки разных знаков, поэтому оно не подменяет среднюю ошибку отдельных моделей.

При N694 full FT точнее по спектру и по синглету, LoRA — по doublet. Для r²SCAN синглет LoRA падает с5.115 (N200 seed10) до4.735THz, хотя spectral RMSE улучшается с0.223 до0.201THz. У full FT синглет меняется4.896→4.866THz, spectral RMSE0.181→0.156THz. Улучшение среднего спектра не означает улучшения каждой моды.

У **всех16 моделей** на mesh50³ нет частот ниже−0.001THz, уже на raw FC, а также после central/ASR. Отрицательные остатки порядка10⁻⁷THz считаются численным нулём. У исходной FM на этих ячейках были выраженные мнимые ветви; её spectral RMSE0.698(PBE) и1.010(r²SCAN). Ни один из новых результатов не доказывает долговременную или высокотемпературную MD-устойчивость.

| N | XC | Метод | Относит. D-matrix ошибка | Относит. FC ошибка | DOS L1, modes/atom | Ошибка Γ–A поперечной скорости, % |
|---:|---|---|---:|---:|---:|---:|
'''
    for n in [200,694]:
        for xc in ['PBE','r2SCAN']:
            for method in ['lora16','full']:
                key=f'{xc}-{method}-N{n}';s=g[key];text+=f"| {n} | {xc} | {NAME[method]} | {fmt(s['relative_Dmat_error'],4)} | {fmt(s['relative_FC_error'],4)} | {fmt(dos_results[key],4)} | {fmt(s['Gamma-A_v1_error_pct'],2)} |\n"
    text+='''
DOS посчитана с одинаковыми mesh50³, Gaussianσ=0.05THz и нормировкой3моды/атом; интегралы проверены. Акустические скорости сравнены на тех же малых q, с сохранённой проверкой q=0.0005/0.001/0.002/0.004. В частности, большая общая точность full FT не устраняет занижение поперечных скоростей r²SCAN вдоль c. Сходимость на малых q относится к фиксированным FC, а не к DFT cutoff/kmesh/supercell.

![Спектры и диапазон seeds](multiseed-fulltrain-20261107/phonons-N200-N694.png)

## Что изменило увеличение train

Сравнение ниже строго парное: **seed10 N694 против seed10 N200**, а не против среднего трёх seeds. Отрицательный процент означает уменьшение ошибки.

| XC | Метод | Δ E aligned | Δ F | Δ stress | Δ spectral RMSE |
|---|---|---:|---:|---:|---:|
'''
    for key,v in d['paired_size_seed10'].items():
        xc,method=key.split('-',1);text+=f"| {xc} | {NAME[method]} | "+' | '.join(f"{v[k]['percent_change']:+.2f}%" for k in ['E_aligned_meV_atom','F_component_eV_A','stress_component_GPa','phonon_sorted_RMSE_THz'])+' |\n'
    text+='''
Больше данных улучшает F/stress во всех четырёх контролях, но aligned E меняется неоднородно. Фононы PBE ухудшаются по sorted RMSE, r²SCAN улучшаются; синглет может ухудшиться при улучшении спектра. N694 использует3.47раза больше optimizer steps при тех же50эпохах, поэтому это практический контроль полного train с изменённым вычислительным бюджетом, а не изолированное влияние числа меток при фиксированных шагах. N694 использует694train+149development=843метки/XC; N200 —349.

## Стоимость, ограничения и дальнейшее решение

'''
    costs=d['costs'];text+=f"Двенадцать новых задач использовали **{costs['new_GPU_hours']:.3f} GPU·h**: N200 seeds20/30 —{costs['N200_new_GPU_hours']:.3f}, N694 —{costs['N694_GPU_hours']:.3f}. С учётом исходного seed10 все16моделей —{costs['new_GPU_hours']+costs['prior_N200_seed10_GPU_hours']:.3f} GPU·h, включая экспорт/оценку в GPU-allocation. Это выше предварительных≈14GPU·h для продолжения; в отчёте используются фактические значения Slurm. Ранее сохранённая неудачная preflight-попытка01 добавила13GPU-секунд и0optimizer steps.\n\n"
    text+='''Все задачи использовали A100-PCIE-40GB. Peak allocated memory: LoRA≈3.56–3.57GiB, full≈3.81–3.82GiB. Обучаемых параметров325158 против1124686; у LoRA285696адаптерных и39462исходных. Существенного выигрыша LoRA по времени эти рабочие запуски не показывают; разные wall times не являются изолированным timing benchmark. Данные, preprocessing, masks и seed-потоки проверены независимо от времени.

Техническое сравнение N200 по трём seeds завершено и воспроизводит компромисс: LoRA — validation aligned E/F и c-singlet, full FT — спектр в целом. **Универсального победителя нет.** N694 full FT интересен как кандидат по спектру, но имеет один seed, а небольшие validation E/stress преимущества неубедительны по условному group-bootstrap. Для устойчивого выбора на полном train логично повторить N694 наseeds20/30; эти дополнительные обучения в данном статусном запросе не запускались. Переход к дорогой MD или автоматический выбор «лучшего» checkpoint не выполнен.

Test149 и архивный benchmark49 не использовались для настройки/выбора. Группы по nTi/nAl остаются консервативным заменителем утраченных liquid-родителей; гарантия условна на сохранение состава в MD+relaxation. Фононы — physical development, не blind test. Различия обучающих/phonon DFT-протоколов и недоказанная cutoff/kmesh/supercell convergence сохраняются. Новых DFT0.

## Каждый seed отдельно

| N | XC | Метод | Seed | E aligned | F, eV/Å | Stress, GPa | Spectral RMSE, THz | Γ singlet, THz |
|---:|---|---|---:|---:|---:|---:|---:|---:|
'''
    for key,c in sorted(cases.items(),key=lambda kv:(kv[1]['N'],kv[1]['XC'],kv[1]['method'],kv[1]['seed'])):
        v=c['values'];text+=f"| {c['N']} | {c['XC']} | {NAME[c['method']]} | {c['seed']} | {v['E_aligned_meV_atom']:.4f} | {v['F_component_eV_A']:.6f} | {v['stress_component_GPa']:.6f} | {v['phonon_sorted_RMSE_THz']:.4f} | {v['Gamma_singlet_THz']:.4f} |\n"
    text+='''
## Артефакты

- [Полное сравнение16моделей JSON](../runs/dpa4-replicates-full-20261106-02/comparison-20261107/comparison.json).
- [Условный group-bootstrap для каждого seed](../runs/dpa4-replicates-full-20261106-02/comparison-20261107/conditional-group-bootstrap.json).
- [Манифест собранных файлов, Slurm accounting и hashes весов](../runs/dpa4-replicates-full-20261106-02/collected/COLLECTION_MANIFEST.json).
- [Исходная автоматическая N200-сводка](../runs/dpa4-replicates-full-20261106-02/collected/aggregates/N200/summary.json) и [N694-сводка](../runs/dpa4-replicates-full-20261106-02/collected/aggregates/N694/summary.json).
- [PDF метрик](multiseed-fulltrain-20261107/metrics-seed-spread.pdf), [PDF спектров](multiseed-fulltrain-20261107/phonons-N200-N694.pdf), [DOS L1](multiseed-fulltrain-20261107/DOS-L1.json).

Новые final merged weights: `runs/dpa4-replicates-full-20261106-02/collected/training-plan01/cases/<case>/merged/10000.pt` или `34700.pt`. Native optimizer checkpoints и frozen exports остались на UMT; пути и SHA256 сохранены в манифесте. Предыдущие данные/отчёты/failed attempts не перезаписаны. Git repository остаётся unborn без author identity; источники и hashes программ сохранены вместе с результатами.
'''
    report=ROOT/'reports/14_multiseed_fulltrain_comparison.md';report.write_text(text,encoding='utf8')
    files=[report,RUN/'comparison-20261107/comparison.json',RUN/'comparison-20261107/conditional-group-bootstrap.json']+list(OUT.glob('*.png'))+list(OUT.glob('*.pdf'))+[OUT/'DOS-L1.json']
    (OUT/'SHA256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
    print(report)

if __name__=='__main__':main()
