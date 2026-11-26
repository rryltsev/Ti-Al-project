"""Independently aggregate the four completed fixed-final GRACE pilot models."""
from pathlib import Path
import datetime,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prepare_grace_pilot import ROOT,RUN

DATA=RUN/'collected-final';OUT=RUN/'comparison-final';FIG=ROOT/'reports/grace-n200-final'
def load(p):return json.loads(p.read_text(encoding='utf8'))
def dump(p,d):p.write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def recompute(rows,reported,subsets):
    assert len(rows)==349 and len({r['id'] for r in rows})==349
    rr={part:[r for r in rows if r['split']==part] for part in ['train','validation']}
    assert {r['id'] for r in rr['train']}==set(subsets['subsets']['200'])
    assert {r['id'] for r in rr['validation']}==set(subsets['validation_ids'])
    assert not {r['id'] for r in rows}.intersection(subsets['test_ids'])
    train=rr['train'];design=np.array([r['fractions'] for r in train]);assert np.linalg.matrix_rank(design)==2
    target=np.array([(r['reference_energy']-r['energy'])/r['natoms'] for r in train])
    beta=np.linalg.lstsq(design,target,rcond=None)[0]
    assert np.allclose(beta,[reported['offset_eV_atom'][s] for s in ['Al','Ti']],rtol=1e-10,atol=1e-12)
    results={}
    for part,records in rr.items():
        e=np.array([(r['energy']-r['reference_energy'])/r['natoms'] for r in records]);x=np.array([r['fractions'] for r in records])
        assert all(r['force_sq']>=0 and r['stress_sq']>=0 for r in records)
        result=dict(E_raw_RMSE_meV_atom=float(np.sqrt(e@e/len(e))*1000),E_aligned_RMSE_meV_atom=float(np.linalg.norm(e+x@beta)/np.sqrt(len(e))*1000),
            F_RMSE_eV_A=float(np.sqrt(sum(r['force_sq'] for r in records)/(3*sum(r['natoms'] for r in records)))),
            stress_RMSE_GPa=float(np.sqrt(sum(r['stress_sq'] for r in records)/(6*len(records)))))
        for k,v in result.items():assert np.isfinite(v) and np.isclose(v,reported['metrics'][part][k],rtol=1e-11,atol=1e-12),(part,k)
        results[part]=result
    return results

def phonons(folder,xc,summary,reference):
    bands=np.load(folder/'bands.npz');assert np.array_equal(bands['qpoints'],reference['qpoints'])
    freq=bands['frequencies'];dft=reference[xc+'_DFT_bands'];assert freq.shape==dft.shape==(604,6)
    error=freq-dft;rmse=float(np.sqrt(np.mean(error**2)))
    assert np.isclose(rmse,summary['sorted_band_RMSE_to_DFT_THz'],atol=1e-12)
    mesh=np.load(folder/'mesh50.npz');assert mesh['weights'].sum()==125000
    imaginary=int(np.sum(mesh['weights'][:,None]*(mesh['frequencies']<-.001)))
    assert imaginary==summary['model']['imaginary_modes_weighted']
    dos=np.load(folder/'dos.npz');area=float(np.trapezoid(dos['dos'],dos['frequency']));assert abs(area-3)<1e-5
    gamma=np.array(summary['model']['gamma_THz']);assert np.max(abs(gamma-freq[0]))<1e-5
    return dict(band_RMSE_THz=rmse,band_max_error_THz=float(np.max(abs(error))),gamma_doublet_THz=float(np.mean(gamma[3:5])),gamma_singlet_THz=float(gamma[5]),
        gamma_doublet_abs_error_THz=float(abs(np.mean(gamma[3:5]-dft[0,3:5]))),gamma_singlet_abs_error_THz=float(abs(gamma[5]-dft[0,5])),
        imaginary_modes_weighted=imaginary,mesh_min_THz=float(mesh['frequencies'].min()),DOS_integral=area)

def main():
    OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
    hashes=load(DATA/'COLLECTION_SHA256.json')
    for name,h in hashes.items():assert sha(DATA/name)==h,name
    remote=load(RUN/'final-remote-verification.json');assert remote['status']=='PASS'
    plan=load(RUN/'package/plan.json');status=load(RUN/'latest-status.json');subsets=load(ROOT/'datasets/nested-train-v1-20261106/subsets.json')
    ref=np.load(RUN/'package/reference_figure.npz');cases={};baseline={};checks=0
    for xc in ['PBE','r2SCAN']:
        metrics=recompute(load(DATA/f'baseline/{xc}-original-rows.json'),load(DATA/f'baseline/{xc}-original-metrics.json'),subsets)
        h=load(DATA/'baseline/harmonics/summary.json')['results'][xc]
        baseline[xc]=dict(metrics=metrics,phonons=phonons(DATA/'baseline/harmonics'/xc,xc,h,ref))
    for i,c in enumerate(plan['cases']):
        name=c['name'];w=DATA/'cases'/name;xc=c['XC'];t=load(w/'training-summary.json')
        assert t['status']=='PASS' and t['steps']==10000 and t['epochs']==50 and not t['test_access']
        assert sha(w/'training-summary.json')==remote['cases'][name]['training_summary_sha256']
        assert t['hashes']==remote['cases'][name]['checkpoint']
        assert sha(w/'seed/10/model.yaml')==remote['cases'][name]['model_yaml_sha256']
        for f in ['evaluation-summary.json','numerical-QC.json','zero-step.json','harmonics/summary.json','checkpoint-path-check.json']:assert load(w/f)['status']=='PASS',(name,f)
        assert load(w/'validation-selection.json')['best_checkpoint_reload_skipped']
        inv=load(w/'parameter-inventory.json');expected=21111690 if c['method']=='restricted' else 21764956
        assert inv['trainable']==expected
        steps=[load(p) for p in sorted((w/'checks').glob('*.json'))]
        assert {v['steps'] for v in steps}=={1,*range(1000,10001,1000)}
        assert all(v['status']=='PASS' and not v['frozen_changed'] and v['optimizer_iterations']==v['steps'] for v in steps)
        checks+=len(steps)
        if c['method']=='restricted':assert len(steps[-1]['changed_tensors'])==12
        metrics=recompute(load(w/'per-structure.json'),load(w/'metrics.json'),subsets)
        h=load(w/'harmonics/summary.json')['results'][xc]
        ph=phonons(w/'harmonics'/xc,xc,h,ref)
        row=next(r.split('|') for r in remote['accounting'].splitlines() if r.startswith('20990597_'+str(i)+'|'));assert row[1]=='COMPLETED' and row[3]=='0:0'
        cases[name]=dict(case=c,metrics=metrics,phonons=ph,checkpoint_hashes=t['hashes'],trainable=expected,
            training_seconds=t['training_seconds'],GPU_job_seconds=int(row[2]),GPU_peak_GiB=t['GPU_memory']['peak']/2**30,checks=len(steps))
    paired={}
    for xc in ['PBE','r2SCAN']:
        f=cases[f'{xc}-full-N200-seed10'];r=cases[f'{xc}-restricted-N200-seed10']
        a=load(DATA/f'cases/{xc}-full-N200-seed10/checks/10000.json');b=load(DATA/f'cases/{xc}-restricted-N200-seed10/checks/10000.json')
        assert len(a['first5_batch_hashes'])==5 and a['first5_batch_hashes']==b['first5_batch_hashes']
        paired[xc]={k:dict(full=v,restricted=r['metrics']['validation'][k],full_reduction_percent=100*(1-v/r['metrics']['validation'][k])) for k,v in f['metrics']['validation'].items()}
        paired[xc]['band_RMSE_THz']=dict(full=f['phonons']['band_RMSE_THz'],restricted=r['phonons']['band_RMSE_THz'],full_reduction_percent=100*(1-f['phonons']['band_RMSE_THz']/r['phonons']['band_RMSE_THz']))
    result=dict(status='PASS',utc=status['utc'],cases=cases,baseline=baseline,paired=paired,checks=dict(collected_files=len(hashes),final_models=4,parameter_checks=checks,validation149_identical=True,train200_only_alignment=True,test_access=False,seed_count=1,seed=10),GPU_job_hours=sum(c['GPU_job_seconds'] for c in cases.values())/3600)
    dump(OUT/'comparison.json',result)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors={'full':'#167a9e','restricted':'#d47729','original':'#888888'}
    fig,axes=plt.subplots(2,4,figsize=(13,6),constrained_layout=True)
    for i,xc in enumerate(['PBE','r2SCAN']):
        for j,(key,title) in enumerate([('E_aligned_RMSE_meV_atom','Energy aligned (meV/atom)'),('F_RMSE_eV_A','Force (eV/Å)'),('stress_RMSE_GPa','Stress (GPa)'),('band_RMSE_THz','Phonon bands (THz)')]):
            ax=axes[i,j];ys=[]
            for method in ['full','restricted']:
                c=cases[f'{xc}-{method}-N200-seed10'];ys.append(c['phonons'][key] if j==3 else c['metrics']['validation'][key])
            ax.bar([0,1],ys,color=[colors['full'],colors['restricted']],width=.6);ax.set_xticks([0,1],['Full','Restricted']);ax.set_title(xc+' · '+title);ax.set_ylim(0,max(ys)*1.22);ax.grid(axis='y',alpha=.2)
            for x,y in enumerate(ys):ax.text(x,y*1.025,f'{y:.4f}',ha='center',va='bottom')
    fig.suptitle('GRACE N=200 · validation 149 · fp64 · seed 10 · final epoch 50')
    fig.savefig(FIG/'validation.png',dpi=180);fig.savefig(FIG/'validation.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),gridspec_kw={'width_ratios':[2.1,1]},constrained_layout=True)
    for i,xc in enumerate(['PBE','r2SCAN']):
        ax,ad=axes[i];x=ref['path_coordinate'];breaks=np.r_[0,np.flatnonzero(np.linalg.norm(np.diff(ref['qpoints'],axis=0),axis=1)>.05)+1,len(x)]
        datasets={'DFT':ref[xc+'_DFT_bands'],'original':np.load(DATA/'baseline/harmonics'/xc/'bands.npz')['frequencies']}
        datasets.update({m:np.load(DATA/f'cases/{xc}-{m}-N200-seed10/harmonics/{xc}/bands.npz')['frequencies'] for m in ['full','restricted']})
        for method,values in datasets.items():
            color='black' if method=='DFT' else colors[method];style='--' if method in ['DFT','original'] else '-'
            for first,last in zip(breaks[:-1],breaks[1:]):
                for band in range(6):ax.plot(x[first:last],values[first:last,band],color=color,ls=style,lw=1,alpha=.6 if method=='original' else .9,label=method if first==0 and band==0 else None)
            if method=='DFT':freq,dos=ref['frequency_THz'],ref[xc+'_DFT_DOS']
            else:
                folder=DATA/'baseline/harmonics'/xc if method=='original' else DATA/f'cases/{xc}-{method}-N200-seed10/harmonics/{xc}'
                d=np.load(folder/'dos.npz');freq,dos=d['frequency'],d['dos']
            ad.plot(dos,freq,color=color,ls=style,lw=1,label=method)
        ax.set(title=xc+' · matched DFT geometry',xlabel='Archived band-path coordinate',ylabel='Frequency (THz)',ylim=(-.2,9));ax.legend(ncol=2,fontsize=8);ax.grid(alpha=.15)
        ad.set(xlabel='DOS (modes / atom / THz)',ylim=(-.2,9));ad.grid(alpha=.15)
    fig.savefig(FIG/'phonons.png',dpi=180);fig.savefig(FIG/'phonons.pdf');plt.close(fig)
    lines=['# GRACE N=200: завершённые full/restricted пилоты','',f'Снимок: {status["utc"]}. Все четыре задачи 20990597_0–3 и фононы 20990598_0–3 COMPLETED, ExitCode 0:0. Проверки обучения, export/numerical QC и гармоник PASS. Для каждой модели выполнено 10000 шагов / 50 эпох, seed 10, fp64.','',
    '| XC | Модель | E raw, meV/atom | E aligned, meV/atom | F, eV/Å | Stress, GPa | Фононы, THz |','|---|---|---:|---:|---:|---:|---:|']
    for xc in ['PBE','r2SCAN']:
        for method in ['original','full','restricted']:
            c=baseline[xc] if method=='original' else cases[f'{xc}-{method}-N200-seed10'];v=c['metrics']['validation']
            lines.append(f'| {xc} | {method} | {v["E_raw_RMSE_meV_atom"]:.4f} | {v["E_aligned_RMSE_meV_atom"]:.4f} | {v["F_RMSE_eV_A"]:.6f} | {v["stress_RMSE_GPa"]:.6f} | {c["phonons"]["band_RMSE_THz"]:.6f} |')
    lines+=['','Все значения — RMSE. Энергия: ошибка на атом с равным весом структур; F: компоненты с весом по числу атомов; stress: шесть компонент с равным весом структур. Исходная энергия original не калибрована; raw у FT уже включает исходное train-only выравнивание до обучения. E aligned дополнительно устраняет остаточный композиционный сдвиг по train200 финальной модели. Ни одна поправка не подгонялась по validation.','',
    '![Validation](grace-n200-final/validation.png)','','В этом seed full выигрывает по F, stress и общей ошибке фононных ветвей для обоих XC. Restricted немного лучше по E aligned и заметнее по raw E. Статистического вывода по seeds пока нет: выполнен только seed 10.']
    for xc in ['PBE','r2SCAN']:
        p=paired[xc];lines.append(f'- {xc}: full снижает F RMSE на {p["F_RMSE_eV_A"]["full_reduction_percent"]:.2f}%, stress на {p["stress_RMSE_GPa"]["full_reduction_percent"]:.2f}%, phonon RMSE на {p["band_RMSE_THz"]["full_reduction_percent"]:.2f}% относительно restricted.')
    lines+=['','## Оптические моды Γ','', '| XC | Модель | Дублет, THz | Синглет вдоль c, THz |','|---|---|---:|---:|']
    for xc in ['PBE','r2SCAN']:
        lines.append(f'| {xc} | DFT | {ref[xc+"_DFT_bands"][0,3:5].mean():.4f} | {ref[xc+"_DFT_bands"][0,5]:.4f} |')
        for method in ['original','full','restricted']:
            ph=(baseline[xc] if method=='original' else cases[f'{xc}-{method}-N200-seed10'])['phonons'];lines.append(f'| {xc} | {method} | {ph["gamma_doublet_THz"]:.4f} | {ph["gamma_singlet_THz"]:.4f} |')
    lines+=['','Restricted завышает Γ-синглет: ошибки около 1.06 THz (PBE) и 1.48 THz (r²SCAN). Full: около 0.029 и 0.104 THz соответственно. Дублеты обоих FT остаются ниже DFT, особенно для r²SCAN. У всех четырёх моделей нет частот ниже −0.001 THz на mesh 50³; численные отрицательные акустические значения порядка 10⁻⁷ THz не считаются нестабильностью.','',
    '![Phonons](grace-n200-final/phonons.png)','','Фононы: те же фиксированные XC-specific DFT-ячейки, ±0.01 Å, central forces, symfc/ASR, mesh 50³. Ошибка ветвей — по отсортированным частотам в 604 q-точках, без сопоставления eigenvectors ветвь-к-ветви. Это диагностическое сравнение на development-структурах; оно не подтверждает сходимость DFT или стабильность длинной MD.','',
    '## Проверки и ресурсы','',f'Локально проверены SHA-256 всех {len(hashes)} собранных файлов, заново рассчитаны validation/train метрики из по-структурных сумм ошибок, ошибки bands, DOS-нормировка и число мнимых мод. На UMT проверены текущие хеши четырёх финальных checkpoint и экспортов, а также 53 исходных файла. Подтверждены {checks} проверки изменений параметров; frozen tensors не менялись. Full: 21764956 trainable, restricted: 21111690 (12 тензоров; около 97% полного числа).','',
    f'Суммарное время четырёх GPU jobs (обучение + оценка/export) — {result["GPU_job_hours"]:.4f} GPU·ч. Отдельно:','', '| XC | FT | Job, min | Training loop, min | Peak TF allocator, GiB |','|---|---|---:|---:|---:|']
    for c in cases.values():lines.append(f'| {c["case"]["XC"]} | {c["case"]["method"]} | {c["GPU_job_seconds"]/60:.2f} | {c["training_seconds"]/60:.2f} | {c["GPU_peak_GiB"]:.3f} |')
    lines+=['','Peak относится к TensorFlow allocator, а не всей памяти процесса. Сравнение времени предварительное: один запуск каждой конфигурации. Финальные бинарные модели остаются на UMT; локально собраны метрики, журналы, model.yaml, хеши и фононные массивы.','',
    'Выборка train200 и validation149 сверена с frozen split; test149 и архивные49 не использованы. Гарантия отсутствия общих родительских траекторий условна при утраченных liquid trajectories и опирается на документированную группировку. Для PBE происхождение исходного поля DFT energy остаётся ранее отмеченным ограничением; raw-аудит r²SCAN выполнен.','',
    'Следующий обоснованный шаг — повторы N=200 с seeds 20/30 для обоих методов и XC, прежде чем объявлять устойчивый выигрыш и переходить к контролю N=694. В этом шаге новые обучения не запускались.','',
    'Машиночитаемое сравнение: `runs/grace-n200-fp64-pilot-20261109-02/comparison-final/comparison.json`. Полный сбор: `collected-final/COLLECTION_SHA256.json`; remote provenance: `final-remote-verification.json`.']
    report=ROOT/'reports/22_grace_n200_results.md';report.write_text('\n'.join(lines)+'\n',encoding='utf8')
    for f in ['state.json','protocol/experiment_matrix.yaml']:
        backup=OUT/(Path(f).name+'.before');
        if not backup.exists():backup.write_bytes((ROOT/f).read_bytes())
    state=load(ROOT/'state.json');state.update(status='GRACE_fp64_N200_pilot_complete',stage='GRACE_N200_seed10_comparison',updated_utc=status['utc'],active_training_jobs=0,completed_training_jobs=28,fully_qualified_training_jobs=28)
    state['grace_pilot'].update(status='complete',completed_cases=4,comparison=str(OUT/'comparison.json'),B0='PASS',H0='PASS',optimizer='PASS')
    state['current_gates'].update(GRACE_B0='PASS',GRACE_H0='PASS',GRACE_optimizer='PASS',GRACE_N200_final_QC='PASS',GRACE_N200_harmonics='PASS')
    state['next_actions']=['Replicate GRACE N200 full/restricted for PBE/r2SCAN with seeds20/30 before claiming seed-robust superiority.','Then evaluate full-train N694 controls under the fixed split; preserve held-out test149 and archival49.']
    if 'reports/22_grace_n200_results.md' not in state['reports']:state['reports'].append('reports/22_grace_n200_results.md')
    dump(ROOT/'state.json',state)
    matrix=load(ROOT/'protocol/experiment_matrix.yaml')
    for d in matrix['training_designs']:
        if d['architecture']=='GRACE-2L' and d['train_size']==200:
            name=f'{d["functional"]}-{d["adaptation"]}-N200-seed10';d.update(state='completed_qualified',launch_ready=False,blocked_by=[],results=str(OUT/'comparison.json'),final_checkpoint_hashes=cases[name]['checkpoint_hashes'],next_step='Replicate seeds20/30; one seed is not evidence of seed-robust superiority.')
    dump(ROOT/'protocol/experiment_matrix.yaml',matrix)
    (ROOT/'reports/summary.md').write_text('# Ti–Al FM: текущее состояние\n\nGRACE N200 fp64: все 4 пилота seed10 и фононы завершены, PASS. Full лучше F/stress/phonons, restricted немного лучше aligned E; разброс seeds ещё не измерен.\n\n[Результаты GRACE N200](22_grace_n200_results.md) · [DPA4 N694 и A100 precision](18_final_n694_grace_a100.md).\n',encoding='utf8')
    print(json.dumps(dict(status='PASS',checks=result['checks'],paired=paired,GPU_job_hours=result['GPU_job_hours'],report=str(report)),indent=2))

if __name__=='__main__':main()
