from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/dpa4-n200-pilot-20261106-02';DATA=RUN/'collected/harmonics';OUT=ROOT/'reports/original-harmonics-20261106'
def main():
    OUT.mkdir(exist_ok=True);summary=json.loads((DATA/'summary.json').read_text())['results'];modes=json.loads((DATA/'mode_diagnostics.json').read_text())['results'];ref=np.load(RUN/'package/reference_figure.npz')
    q=ref['qpoints'];x=ref['path_coordinate'];ticks=[];labels=[]
    for label,v in [(r'$\Gamma$',[0,0,0]),('K',[1/3,1/3,0]),('M',[.5,0,0]),('A',[0,0,.5])]:
        for i in np.where(np.max(np.abs(q-v),axis=1)<1e-9)[0]:ticks.append(float(x[i]));labels.append(label)
    tl=sorted(set(zip(ticks,labels)));ticks=[v[0] for v in tl];labels=[v[1] for v in tl]
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,2,figsize=(11,11),gridspec_kw={'width_ratios':[3.3,1.2]},constrained_layout=True)
    for row,xc in enumerate(['PBE','PBEsol','r2SCAN']):
        ax,ad=axes[row];high=[]
        for name,color,style,label in [('DFT','#233d59','-','DFT reference'),('original','#ce5b21','--','Original DPA4')]:
            freq=np.load(DATA/xc/name/'bands.npz')['frequencies'];dos=np.load(DATA/xc/name/'dos.npz');high.append(float(np.max(freq)))
            for b in range(6):ax.plot(x,freq[:,b],color=color,ls=style,lw=1.1,label=label if b==0 else None)
            ad.plot(dos['dos'],dos['frequency'],color=color,ls=style,lw=1.2)
        ax.set(xticks=ticks,xticklabels=labels,xlim=(float(x.min()),float(x.max())),ylabel='Frequency (THz)',title=xc+' DFT geometry')
        ymin=min(-1.7,summary[xc]['original']['mesh_min_THz']-.2);ymax=max(high)+.3
        for a in [ax,ad]:a.set_ylim(ymin,ymax);a.axhline(0,color='#707070',lw=.7);a.grid(axis='y',alpha=.18)
        for t in ticks:ax.axvline(t,color='#dddddd',lw=.6,zorder=0)
        ax.legend(frameon=False,fontsize=9,loc='lower left');ad.set(xlabel='DOS (modes/atom/THz)');ad.tick_params(labelleft=False)
        ax.text(.99,.025,f"Sorted-band RMSE = {summary[xc]['sorted_band_RMSE_THz']:.3f} THz",ha='right',va='bottom',transform=ax.transAxes,fontsize=9)
        ax.text(.99,.09,f"Minimum off this path (50³ mesh): {summary[xc]['original']['mesh_min_THz']:.2f} THz",ha='right',va='bottom',transform=ax.transAxes,fontsize=9,color='#a3441b')
    fig.suptitle('Original DPA4-Neo on matched alpha-Ti DFT geometries\n4×4×3; ±0.01 Å; identical force processing; DOS mesh50³, σ=0.05 THz',fontsize=13)
    for ext in ['png','pdf']:fig.savefig(OUT/('original-DPA4-vs-DFT.'+ext),dpi=200)
    plt.close(fig)
    text='''# Фононы исходной DPA4: H0

Использован исходный DPA4-Neo-OMat24-v20260805 с SHA256 `fd7f34ae28f921201e4a0328ddf56179892752084f8363e7796038201471c989`. Каждому функционалу соответствует его собственная фиксированная DFT-ячейка; геометрии не релаксировались моделью. Job20988449 завершён за52с. Новых DFT:0.

DFT-спектры воспроизведены из сохранённых сил с максимальной разницей менее2.6×10⁻⁷ ТГц относительно архивного Figure2. Одинаковы supercell4×4×3, ±0.01Å, масса47.867, central-pair/mean-drift, symfc и ASR, DOS mesh50³ и σ0.05ТГц. Интеграл DOS =3 моды/атом. Исходные силы и raw/central/ASR силовые константы сохранены отдельно.

| Reference cell | Γ doublet DFT / FM, THz | Γ singlet DFT / FM, THz | Sorted spectral RMSE, THz | Mesh minimum FM, THz |
|---|---:|---:|---:|---:|
'''
    for xc,s in summary.items():
        g=modes[xc]['Gamma'];text+=f"| {xc} | {np.mean(g['DFT']['optical_doublet_basal_THz']):.3f} / {np.mean(g['original']['optical_doublet_basal_THz']):.3f} | {g['DFT']['optical_singlet_c_THz']:.3f} / {g['original']['optical_singlet_c_THz']:.3f} | {s['sorted_band_RMSE_THz']:.3f} | {s['original']['mesh_min_THz']:.3f} |\n"
    text+='''
Исходная FM имеет мнимые ветви на всех трёх фиксированных DFT-геометриях; обработанный DFT-reference их не имеет на данной сетке. Численная воспроизводимость H0 пройдена, но это не означает физическую точность исходной модели. Малый остаток ASR сам по себе не устраняет неустойчивость.

Отдельный job20988452 проверил raw, central и ASR силовые константы. Большие отрицательные частоты FM сохраняются на всех стадиях практически без изменения: обработка их не создала. Минимум находится около H, q=[0.34,0.32,-0.5], вне показанного архивного пути Γ–K–M–Γ–A. Малые raw DFT остатки порядка0.004ТГц около Γ сохранены в diagnostics; после central-pair/ASR они исчезают.

Для r²SCAN исходный optical c-singlet5.439ТГц близок к DFT5.189ТГц, но doublet2.930ТГц заметно ниже DFT4.587ТГц. Архивный адаптированный checkpoint имел singlet около4.221ТГц и меньшую общую spectral RMSE около0.259ТГц. Следовательно, адаптация могла улучшить спектр в целом, ухудшив эту конкретную моду. Старый checkpoint не является независимым baseline нового train/test-разбиения; это только согласованная физическая диагностика.

В `mode_diagnostics.json` отдельно сохранены максимальное overlap-сопоставление собственных векторов при каждом q, projectors для вырожденных подпространств, различия динамических матриц и акустические фазовые скорости на четырёх малых q. Sorted RMSE не выдаётся за ошибку конкретной отслеживаемой ветви. Сходимость по малому q при фиксированных FC не устанавливает сходимость по supercell, cutoff и электронной k-сетке.

Фононы, используемые при выборе настройки пилота, относятся к physical development benchmark. Проверки нового test149 и физического benchmark имеют разные роли.

![Фононный baseline](original-harmonics-20261106/original-DPA4-vs-DFT.png)
'''
    (ROOT/'reports/10_original_harmonics.md').write_text(text,encoding='utf8');print(OUT)
if __name__=='__main__':main()
