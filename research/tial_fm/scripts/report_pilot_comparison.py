"""Write the completed pilot comparison and publication-exportable phonon plot."""
from pathlib import Path
import hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]; RUN=ROOT/'runs/dpa4-n200-pilot-20261106-02'
OUT=ROOT/'reports/n200-pilot-comparison-20261106'
NAMES={'original':'Исходная FM','lora16':'LoRA16','full':'Full FT'}
SHORT={'POSCAR_alpha-Ti-hcp_orthogonal':'α-Ti/hcp','POSCAR_beta-Ti-bcc':'β-Ti/bcc',
       'POSCAR_gamma_AlTi_L1_0':'γ-TiAl','unresolved_liquid_relax_pool':'Liquid + relaxation'}

def main():
    d=json.loads((RUN/'comparison-v1/comparison.json').read_text()); assert d['status']=='PASS'
    cases=d['cases']; physics=d['baseline_harmonics']; OUT.mkdir(exist_ok=True)
    ref=np.load(RUN/'package/reference_figure.npz'); q=ref['qpoints']; x=ref['path_coordinate']
    ticks=[]
    for label,v in [(r'$\Gamma$',[0,0,0]),('K',[1/3,1/3,0]),('M',[.5,0,0]),('A',[0,0,.5])]:
        ticks += [(float(x[i]),label) for i in np.where(np.max(np.abs(q-v),axis=1)<1e-9)[0]]
    ticks=sorted(set(ticks))
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),gridspec_kw={'width_ratios':[3.6,1.2]},constrained_layout=True)
    for row,xc in enumerate(['PBE','r2SCAN']):
        ax,ad=axes[row]; allfreq=[]
        for name,color,style,label in [('original','#aaaaaa','--','Original FM'),('DFT','#222222','-','DFT'),
                                      ('lora16','#0072B2','--','LoRA16'),('full','#D55E00','-.','Full FT')]:
            folder=(RUN/'collected/harmonics'/xc/name if name in ['DFT','original'] else
                    RUN/'collected-pilots/training-plan01/cases'/f'{xc}-{name}'/'evaluation/harmonics')
            bands=np.load(folder/'bands.npz'); freq=bands['frequencies']; dos=np.load(folder/'dos.npz')
            assert np.array_equal(q,bands['qpoints']); allfreq.append(freq)
            for j in range(6):ax.plot(x,freq[:,j],color=color,ls=style,lw=1.1,label=label if j==0 else None)
            ad.plot(dos['dos'],dos['frequency'],color=color,ls=style,lw=1.2)
        ax.set(xticks=[t[0] for t in ticks],xticklabels=[t[1] for t in ticks],xlim=(x.min(),x.max()),
               ylabel='Frequency (THz)',title=('r²SCAN' if xc=='r2SCAN' else xc)+' reference geometry')
        ymax=max(float(a.max()) for a in allfreq)+.35
        for a in [ax,ad]:
            a.set_ylim(-1.6,ymax);a.axhline(0,color='#777777',lw=.7);a.grid(axis='y',alpha=.17)
        for v,_ in ticks:ax.axvline(v,color='#dddddd',lw=.6,zorder=0)
        ax.legend(frameon=False,ncol=2,fontsize=9,loc='lower left')
        ad.set(xlabel='DOS (modes/atom/THz)');ad.tick_params(labelleft=False)
        ax.text(.99,.035,'Spectral RMSE: LoRA %.3f / full %.3f THz' %
                (physics[xc]['lora16']['sorted_RMSE_THz'],physics[xc]['full']['sorted_RMSE_THz']),
                transform=ax.transAxes,ha='right',fontsize=9)
        ax.text(.99,.105,'50³ mesh: no imaginary modes for either pilot',
                transform=ax.transAxes,ha='right',fontsize=9)
    fig.suptitle('DPA4-Neo N=200: final 10,000-step checkpoints, seed 10\nα-Ti; fixed DFT cells; 4×4×3; ±0.01 Å; identical ASR; DOS σ=0.05 THz',fontsize=13)
    for ext in ['png','pdf']:fig.savefig(OUT/('phonons-LoRA-full.'+ext),dpi=200)
    plt.close(fig)

    text='''# DPA4 N=200: завершённый пилот LoRA16 и full FT

Все четыре запуска завершены: Slurm array **20988454_0–3**, по **10 000 шагов**, 50 nominal epochs, seed 10. Финальный checkpoint задан заранее; выбора лучшего по validation не было. Все четыре задачи метрик 20988460 и численного QC 20988469 завершены с exit 0. Дополнительная постобработка сохранённых FC (20988553) заняла 29 CPU-секунд. Новых DFT: **0**.

Собраны 687 файлов исходного результата с проверкой SHA256, включая четыре финальных merged checkpoint, train/validation-предсказания, силовые константы, спектры, DOS, журналы и конфигурации. Результаты дополнены диагностикой собственных векторов и акустических наклонов. Локальный повторный расчёт всех метрик воспроизводит кластерный в пределах rtol=1e-11, atol=1e-12.

## Сопоставимость и проверки

Одна исходная DPA4-Neo-OMat24-v20260805, SHA256 `fd7f34ae28f921201e4a0328ddf56179892752084f8363e7796038201471c989`. Единый split 694/149/149; N200 и validation149 имеют одинаковые ID у всех четырёх кандидатов. Пересечений их консервативных групп нет. Все 10 000 записей порядка optimizer batches совпадают побайтно; каждый запуск получил 10 000 frame exposures и 2 201 898 atom exposures. Метки нового test149 не использованы в сравнении, настройке или выборе checkpoint. Старый benchmark49 остаётся архивным.

Группы заданы точным числом Ti/Al, совместно для всех исходных семейств. Исходные жидкостные траектории утрачены; независимость условна относительно задокументированного MD+relaxation с сохранением состава. Это более грубая группировка, чем реальные родительские траектории, и одновременно проверка переноса между составами.

Пройдены zero-step, проверки маски, неизменности замороженных весов, live/merged на всех десяти checkpoint, native/export и финальные E/F, E/strain–virial, permutation/PBC/rotation QC. PASS означает численную корректность, а не доказанную физическую точность или MD-устойчивость.

## Validation: 149 структур, 23 группы

Ниже RMSE. Энергии считаются на атом с одинаковым весом структуры. E aligned использует только два элементных сдвига, найденных на N200 train отдельно для каждой модели. Силы объединены по атомам и компонентам; напряжения — по шести симметричным компонентам, σ=−sym(V)/Ω. Это не stochastic mini-batch MAE из training log.

| XC | Модель | E raw, meV/atom | E aligned, meV/atom | F, eV/Å | σ, GPa |
|---|---|---:|---:|---:|---:|
'''
    for xc in ['PBE','r2SCAN']:
        for method in ['original','lora16','full']:
            m=d['baseline'][xc]['metrics']['validation'] if method=='original' else cases[xc+'-'+method]['metrics']['metrics']['validation']
            text+=f"| {xc} | {NAMES[method]} | {m['E_raw_meV_atom']['RMSE']:.3f} | {m['E_aligned_meV_atom']['RMSE']:.3f} | {m['F_component_eV_A']['RMSE']:.5f} | {m['stress_component_GPa']['RMSE']:.5f} |\n"
    text+='''
LoRA имеет меньшие validation RMSE энергии после выравнивания и сил для обоих функционалов. Преимущество по силовой RMSE относительно full FT составляет 2,52% для PBE и 2,41% для r²SCAN. Full FT имеет меньшие точечные ошибки напряжений. Все четыре кандидата улучшают эти три метрики относительно исходной FM. Большая raw-ошибка исходной FM относительно r²SCAN включает элементный энергетический сдвиг; сравнение по aligned E отделяет его от остаточной ошибки.

Парный bootstrap: 2000 повторов, одинаковые 23 консервативные группы resampled для обеих моделей. Δ=RMSE(LoRA)−RMSE(full); отрицательное значение благоприятно для LoRA. Сдвиги энергий фиксированы по train. Интервалы условны для seed 10 и этих данных; они не оценивают разброс обучения между seeds.

| XC | Метрика | Δ | 95% интервал Δ |
|---|---|---:|---:|
'''
    for xc,b in d['paired_validation'].items():
        for key,label in [('E_aligned_meV_atom','E aligned, meV/atom'),('F_component_eV_A','F, eV/Å'),('stress_component_GPa','σ, GPa')]:
            v=b['delta'][key];text+=f"| {xc} | {label} | {v['point']:+.6f} | [{v['CI95'][0]:+.6f}; {v['CI95'][1]:+.6f}] |\n"
    text+='''
Для энергии и сил интервалы не пересекают ноль. Для напряжений PBE интервал включает ноль; преимущество full FT здесь пока неубедительно. Для r²SCAN интервал напряжений благоприятен для full FT.

Validation содержит 133 liquid/relaxation, 7 γ-TiAl, 7 β-Ti/bcc и только 2 α-Ti/hcp. Средняя метрика преимущественно характеризует жидкостный пул. По F RMSE LoRA лучше на liquid/relaxation, а full FT немного лучше на α/β-кристаллических поднаборах; выборки кристаллов малы.

| XC | Метод | Train E aligned, meV/atom | Train F, eV/Å | Validation F p95 / p99 / max, eV/Å |
|---|---|---:|---:|---:|
'''
    for c in cases.values():
        tr=c['metrics']['metrics']['train'];v=c['metrics']['metrics']['validation']['F_component_eV_A']
        text+=f"| {c['XC']} | {NAMES[c['method']]} | {tr['E_aligned_meV_atom']['RMSE']:.3f} | {tr['F_component_eV_A']['RMSE']:.5f} | {v['abs_q95']:.3f} / {v['abs_q99']:.3f} / {v['abs_max']:.3f} |\n"
    text+='''
Full FT сильнее снижает train-ошибку сил, но имеет больший разрыв train–validation. LoRA улучшает p95/p99, хотя максимальная ошибка компоненты силы у неё немного выше. По одному seed это не устанавливает универсальное преимущество регуляризации.

## Фононы на одинаковых DFT-геометриях α-Ti

Для каждого XC использованы его фиксированная DFT-ячейка, supercell 4×4×3, смещения ±0.01 Å, масса Ti 47.867, одна обработка central-pair/mean-drift/symfc/ASR. ML-релаксация не выполнялась. Сравнение частотно упорядоченных ветвей относится к общему пути Γ–K–M–Γ–A (604 точки), а устойчивость дополнительно проверена на сетке 50³. DOS: σ=0.05 THz, интеграл 3 моды/атом.

| XC | Модель | Spectral sorted RMSE, THz | Overlap-matched RMSE, THz | Γ doublet, THz | Γ c-singlet, THz | Мнимые моды на 50³ |
|---|---|---:|---:|---:|---:|---:|
'''
    for xc,ss in physics.items():
        for method in ['DFT','original','lora16','full']:
            v=ss[method];g=v['Gamma'];sr=f"{v['sorted_RMSE_THz']:.3f}" if method!='DFT' else '—';mr=f"{v['overlap_matched_RMSE_THz']:.3f}" if method!='DFT' else '—'
            text+=f"| {xc} | {NAMES.get(method,method)} | {sr} | {mr} | {np.mean(g['doublet_basal_THz']):.3f} | {g['singlet_c_THz']:.3f} | {v['projection']['ASR']['imaginary_modes_weighted']} |\n"
    text+='''
У обеих адаптаций отсутствуют частоты ниже −0.001 THz на проверенной mesh50³, причём уже до central/ASR. Остатки порядка 10⁻⁷ THz — численный ноль. У исходной FM минимум −1.239 THz для PBE и −1.346 THz для r²SCAN, около H вне показанного пути. Отсутствие мнимых мод на этой сетке не является отдельной проверкой конечной температуры или долговременной MD.

Full FT точнее по общей spectral RMSE, overlap-сопоставлению и Γ-doublet. LoRA точнее по Γ c-singlet: ошибки −0.075 THz для обоих XC против −0.339 THz (PBE) и −0.293 THz (r²SCAN) у full FT. Относительно исходной FM full FT ухудшает этот синглет, одновременно улучшая спектр в целом. У PBE исходный синглет ближе к DFT, чем у обеих адаптаций; у r²SCAN LoRA улучшает синглет относительно исходной FM. Это локальный конфликт метрик, а не доказательство глобального forgetting.

| XC | Метод | Относит. ошибка динамических матриц | Относит. ошибка FC | DOS L1, modes/atom | (ωsinglet/ωDFT)² |
|---|---|---:|---:|---:|---:|
'''
    for xc,ss in physics.items():
        for method in ['original','lora16','full']:
            v=ss[method];text+=f"| {xc} | {NAMES[method]} | {v['relative_dynamical_matrix_Frobenius_error']:.4f} | {v['relative_FC_Frobenius_error']:.4f} | {v['DOS_L1_modes_per_atom']:.4f} | {v['singlet_eigenvalue_ratio_to_DFT']:.4f} |\n"
    text+='''
Для вырожденных групп сравнивались проекторы подпространств (порог разницы частот 10⁻⁴ THz); минимальная fidelity ≈1. Для одиночных мод минимум overlap у пилотов 0.943–0.981. Overlap assignment выполняется отдельно в каждом q, без заявления о непрерывном tracking ветви. Частотно упорядоченный RMSE не интерпретируется как ошибка идентифицированной моды. Отношение квадратов частот относится только к собственному значению синглета, а не к полной жёсткости кристалла.

Акустические фазовые скорости ниже взяты при q=0.0005×направление, в m/s. Три акустические ветви упорядочены по скорости; Γ–K в этом пределе совпадает с Γ–M в приведённой точности.

| XC | Модель | Γ–M: v₁ / v₂ / v₃ | Γ–A: v₁ / v₂ / v₃ |
|---|---|---:|---:|
'''
    for xc,ss in physics.items():
        for method in ['DFT','original','lora16','full']:
            ac=ss[method]['acoustics'];vals=[' / '.join(f'{v:.0f}' for v in ac[k]['phase_velocity_m_s'][0]) for k in ['Gamma-M','Gamma-A']]
            text+=f"| {xc} | {NAMES.get(method,method)} | {vals[0]} | {vals[1]} |\n"
    text+='''
Проверены q-масштабы 0.0005, 0.001, 0.002, 0.004; у пилотов максимальное относительное изменение между двумя малыми q <4×10⁻⁶. Это сходимость производной при фиксированных FC. Full FT уступает LoRA по поперечной скорости вдоль c: для r²SCAN ошибки около −16.6% и −12.2% соответственно. Сходимость DFT по cutoff, электронной k-сетке и supercell этим не установлена. Различия обучающего и phonon-DFT протоколов описаны в отчётах 09 и 10.

![LoRA и full FT на общей геометрии](n200-pilot-comparison-20261106/phonons-LoRA-full.png)

## Стоимость и параметры

| XC | Метод | Обучаемых параметров | Training wall, min | GPU allocation с оценкой, min | Peak allocated GPU, GiB |
|---|---|---:|---:|---:|---:|
'''
    for c in cases.values():
        t=c['training'];text+=f"| {c['XC']} | {NAMES[c['method']]} | {c['parameter_counts']['trainable']:,} | {t['training_seconds']/60:.2f} | {c['allocated_GPU_seconds']/60:.2f} | {t['peak_cuda_allocated_bytes']/2**30:.3f} |\n"
    gpu_hours=sum(c['allocated_GPU_seconds'] for c in cases.values())/3600
    text+=f'''
Все четыре задачи использовали NVIDIA A100-PCIE-40GB, по одной последовательно; суммарное выделенное GPU-время **{gpu_hours:.3f} GPU·h** включает экспорт, оценку и фононную постобработку. На каждый optimizer step подаётся одна структура. Число nominal epochs не означает равномерного просмотра каждой структуры: фактическая выборка записана в training-exposures.jsonl.

LoRA обучает 325 158 параметров: 285 696 адаптерных и 39 462 исходных, согласно установленной политике. Full FT обучает 1 124 686 параметров. Сокращение числа обучаемых параметров ≈71%, но в этом измерении LoRA примерно на 7% медленнее и уменьшает peak allocated GPU-память лишь на 6.5%. Преимущества по скорости обучения здесь нет. Peak allocated не равно зарезервированной PyTorch памяти или памяти по nvidia-smi; reserved сохранён в JSON.

## Решение по пилоту

Технический screening пройден для обеих адаптаций. Единого победителя нет: LoRA предпочтительна по validation E/F и c-singlet, full FT — по спектру в целом, doublet и точечной stress RMSE; PBE stress-различие пока статистически неубедительно. Для r²SCAN DOS L1 почти совпадает, несмотря на заметную разницу spectral RMSE.

Оба режима следует сохранить как кандидатов для повторов на не менее трёх training seeds и сопоставимого полного train694. Это следующие запланированные опыты, не запущенные в данном статусном запросе. Gate FT как воспроизводимый выбор режима ещё не закрыт. Фононы здесь служат physical development benchmark; они не независимый test после выбора по ним. К дорогой MD по этим результатам не переходим.

## Воспроизведение и артефакты

- [Полное сравнение JSON](../runs/dpa4-n200-pilot-20261106-02/comparison-v1/comparison.json), включая MAE/RMSE, хвосты, элементы, семейства, bootstrap и hashes.
- [Манифест 687 файлов](../runs/dpa4-n200-pilot-20261106-02/collected-pilots/COLLECTION_MANIFEST.json); здесь же Slurm accounting и пути checkpoints/exports на UMT.
- [Дополнительная физическая диагностика](../runs/dpa4-n200-pilot-20261106-02/pilot-physical-diagnostics-v1/summary.json).
- [PDF рисунка](n200-pilot-comparison-20261106/phonons-LoRA-full.pdf).

Финальные merged weights сохранены локально в `runs/dpa4-n200-pilot-20261106-02/collected-pilots/training-plan01/cases/<case>/merged/10000.pt`. Native optimizer checkpoints и frozen exports остались на UMT, их hashes включены в манифест. Исходные данные и прежние результаты не перезаписаны. Git commit отсутствует: исходный репозиторий unborn без author identity; точные исходники выполненных программ и контрольные суммы сохранены в каталоге результатов.

| Модель | SHA256 финальных merged weights |
|---|---|
'''
    for name,c in cases.items():text+=f"| {name} | `{c['model_sha256']}` |\n"
    report=ROOT/'reports/12_n200_pilot_comparison.md';report.write_text(text,encoding='utf8')
    files=[report,OUT/'phonons-LoRA-full.png',OUT/'phonons-LoRA-full.pdf',RUN/'comparison-v1/comparison.json']
    (OUT/'SHA256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+'\n')
    print(report)

if __name__=='__main__':main()
