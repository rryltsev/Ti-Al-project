"""Verify collected final models and publish the completed seed/precision comparison."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import datetime,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analyze_completed_expansion as previous

ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/dpa4-n694-seeds-20261107-01'
OLD=ROOT/'runs/dpa4-replicates-full-20261106-02';G=ROOT/'runs/grace-precision-20261107-02'
OUT=RUN/'comparison-20261108';FIG=ROOT/'reports/final-n694-grace-20261108'
def load(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')

def main():
    OUT.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
    agg=load(RUN/'collected/aggregates/N694/summary.json');assert agg['status']=='PASS' and not agg['failures']
    plan=load(RUN/'package/plan.json');oldplan=load(OLD/'package/plan.json')
    for run,cc in [(RUN,plan['cases']),(OLD,[c for c in oldplan['cases'] if c['N']==694])]:
        previous.RUN=run
        with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(previous.verify_metrics,cc))
    cases={};checkpoints=0;validation=[]
    for name,c in agg['cases'].items():
        if c['case']['N']!=694:continue
        run=OLD if c['case']['seed']==10 else RUN;folder=run/'collected/training-plan01/cases'/name
        for f in ['training-summary.json','zero-step.json','parameter-update-check.json','evaluation/summary.json','evaluation/harmonics/summary.json','evaluation/numerical-QC/summary.json','evaluation/physical-diagnostics/summary.json']:
            assert load(folder/f)['status']=='PASS',(name,f)
        mh=sha(folder/'merged/34700.pt');assert mh==c['model_sha256']==load(folder/'evaluation/summary.json')['model_sha256']==load(folder/'evaluation/numerical-QC/summary.json')['sha256']
        checks=[load(p) for p in (folder/'checks').glob('*.json')];assert len(checks)==35 and all(x['status']=='PASS' for x in checks);checkpoints+=len(checks)
        assert not load(folder/'parameter-update-check.json')['frozen_changed']
        rows=list(map(json.loads,(folder/'evaluation/metrics/per_structure.jsonl').read_text().splitlines()));validation.append(sorted(r['id'] for r in rows if r['split']=='validation'))
        assert len(rows)==843 and {r['split'] for r in rows}=={'train','validation'}
        cases[name]=dict(c,local_folder=str(folder),checkpoint_checks=len(checks))
    assert len(cases)==12 and all(v==validation[0] and len(v)==149 for v in validation)
    grouped={k:v for k,v in agg['seed_summary'].items() if k.endswith('N694')};paired={}
    for xc in ['PBE','r2SCAN']:
        paired[xc]={}
        for metric in cases[f'{xc}-full-N694-seed10']['values']:
            ds=[cases[f'{xc}-lora16-N694-seed{s}']['values'][metric]-cases[f'{xc}-full-N694-seed{s}']['values'][metric] for s in [10,20,30]]
            paired[xc][metric]=dict(mean=float(np.mean(ds)),sample_SD=float(np.std(ds,ddof=1)),values=ds,LoRA_lower=sum(x<0 for x in ds),full_lower=sum(x>0 for x in ds))
        for method in ['lora16','full']:
            cc=[cases[f'{xc}-{method}-N694-seed{s}'] for s in [10,20,30]]
            assert len({c['exposures_sha256'] for c in cc})==3
            if method=='lora16':assert len({c['initial_adapters_sha256'] for c in cc})==3
            for metric,s in grouped[f'{xc}-{method}-N694'].items():
                a=[c['values'][metric] for c in cc];assert np.isclose(np.mean(a),s['mean']) and np.isclose(np.std(a,ddof=1),s['sample_SD'])
    for seed in [10,20,30]:assert len({c['exposures_sha256'] for c in cases.values() if c['case']['seed']==seed})==1
    gpu=load(G/'collected/comparison/gpu/summary.json');assert gpu['status']=='PASS'
    for name,h in load(G/'gpu-collection.json')['files'].items():assert sha(G/'collected'/name)==h
    preds={p:load(G/f'collected/results/gpu/{p}-block0/panel-predictions.json') for p in ['fp32','fp64']}
    labels=load(G/'package/inputs/panel-labels.json');values={'E_meV_atom':[],'F_eV_A':[],'stress_GPa':[]}
    for sid,l in labels.items():
        a,b=preds['fp32'][sid],preds['fp64'][sid]
        values['E_meV_atom'].append(1000*(a['energy']-b['energy'])/l['natoms']);values['F_eV_A'].extend((np.array(a['forces'])-b['forces']).ravel());values['stress_GPa'].extend((np.array(a['stress'])-b['stress'])*160.21766208)
    for k,v in values.items():assert abs(np.sqrt(np.mean(np.square(v)))-gpu['precision_delta'][k]['RMSE'])<1e-14
    for xc in ['PBE','r2SCAN']:
        bands={p:np.load(G/f'collected/comparison/gpu/harmonics/{xc}/{p}/bands.npz')['frequencies'] for p in ['fp32','fp64']}
        assert abs(np.sqrt(np.mean((bands['fp32']-bands['fp64'])**2))-gpu['harmonics'][xc]['fp32_minus_fp64']['band_RMSE_THz'])<1e-14
    accounting=load(RUN/'collected/COLLECTION_MANIFEST.json')['accounting'];gpurows=[r.split('|') for r in accounting.splitlines()[1:] if r.startswith('20989407_') and '.' not in r.split('|')[0]]
    result=dict(status='PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=cases,seed_summary=grouped,paired_LoRA_minus_full=paired,GPU_grace=gpu,
        checks=dict(N694_models=12,checkpoint_checks=checkpoints,local_validation_metrics_reproduced=12,identical_validation149=True,seed_independence=True,no_test=True,GRACE_GPU_prediction_phonon_recomputed=True),
        additional_N694_GPU_hours=sum(int(r[2]) for r in gpurows)/3600)
    dump(OUT/'comparison.json',result)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    metrics=['E_aligned_meV_atom','F_component_eV_A','stress_component_GPa','phonon_sorted_RMSE_THz'];titles=['Validation E aligned (meV/atom)','Validation force RMSE (eV/Å)','Validation stress RMSE (GPa)','Sorted-band RMSE (THz)']
    fig,axs=plt.subplots(2,4,figsize=(13,6),constrained_layout=True)
    for i,xc in enumerate(['PBE','r2SCAN']):
        for j,(metric,title) in enumerate(zip(metrics,titles)):
            ax=axs[i,j]
            for x,method,color in [(0,'lora16','#1874a2'),(1,'full','#c56b28')]:
                s=grouped[f'{xc}-{method}-N694'][metric];ax.errorbar(x,s['mean'],yerr=s['sample_SD'],fmt='o',color=color,capsize=5);ax.scatter(np.array([x-.06,x,x+.06]),s['values'],s=14,color=color,alpha=.7)
            ax.set_xticks([0,1],['LoRA','Full']);ax.set_title(xc+'\n'+title);ax.grid(axis='y',alpha=.2)
    fig.savefig(FIG/'n694-seeds.png',dpi=180);fig.savefig(FIG/'n694-seeds.pdf');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,3.9),constrained_layout=True);x=np.arange(4)
    for p,color in [('fp32','#1874a2'),('fp64','#c56b28')]:
        rr=[v[p] for v in gpu['timing'].values()];y=np.array([r['median_ms'] for r in rr]);lo=np.array([r['p10_ms'] for r in rr]);hi=np.array([r['p90_ms'] for r in rr]);axs[0].errorbar(x,y,yerr=[y-lo,hi-y],fmt='o-',capsize=3,label=p,color=color)
        axs[1].plot(x,[r['GPU_peak_bytes']/2**30 for r in rr],'o-',label=p,color=color)
    for ax in axs:ax.set_xticks(x,list(gpu['timing']));ax.legend();ax.grid(alpha=.2)
    axs[0].set_ylabel('Total ASE E/F/stress latency (ms)');axs[0].set_title('A100: median and P10–P90');axs[1].set_ylabel('TensorFlow allocator peak (GiB)');axs[1].set_title('A100 memory during measured calls')
    fig.savefig(FIG/'grace-a100.png',dpi=180);fig.savefig(FIG/'grace-a100.pdf');plt.close(fig)
    report='reports/18_final_n694_grace_a100.md';lines=['# N=694: три seeds; GRACE fp32/fp64 на A100','',f'Собрано {result["utc"]} UTC. Все задания завершены. N694: 12 моделей seeds10/20/30, 420 checkpoint checks, numerical/physical QC PASS. Все 12 validation-метрик воспроизведены локально из сохранённых предсказаний. Новые восемь повторов потребовали {result["additional_N694_GPU_hours"]:.3f} GPU·ч.','',
        '## DPA4: validation и фононы','', 'Среднее ± sample SD по трём training seeds. Validation149 общий; energy alignment подгоняется только на train каждой модели. Ошибки E/F/stress — RMSE. Фононы: одинаковые DFT cells, ±0.01Å, central/ASR protocol.','',
        '| XC | Метод | E aligned, meV/atom | F, eV/Å | Stress, GPa | Phonon, THz |','|---|---|---:|---:|---:|---:|']
    for xc in ['PBE','r2SCAN']:
        for method in ['lora16','full']:
            row=grouped[f'{xc}-{method}-N694'];ss=[f'{row[k]["mean"]:.6f} ± {row[k]["sample_SD"]:.6f}' for k in metrics]
            lines.append('| '+xc+' | '+method+' | '+' | '.join(ss)+' |')
    lines+=['','LoRA даёт меньший force RMSE во всех трёх seeds для обоих XC: среднее преимущество около0.65% PBE и0.79% r²SCAN. Full FT даёт меньшую ошибку всего фононного спектра во всех трёх seeds: улучшение относительно LoRA около15.1% PBE и21.3% r²SCAN; разброс full FT также меньше.','',
        'По aligned E full FT лучше в2/3 seeds для каждого XC; средние преимущества0.043 и0.034 meV/atom сопоставимы с seed-разбросом, поэтому убедительного энергетического победителя нет. По stress full FT лучше в3/3 PBE и2/3 r²SCAN; среднее r²SCAN различие мало относительно разброса. Сохраняется компромисс между force fitting и глобальной гармоникой. При приоритете фононов full FT — предпочтительный DPA4-кандидат.','',
        'Raw E RMSE (без alignment), mean±SD: PBE LoRA2.278±0.653, full1.651±0.372; r²SCAN LoRA2.221±1.275, full1.676±0.395 meV/atom. Полные значения каждого seed, Γ doublet/singlet и парные разности сохранены в comparison.json. Проверки Γ-мод и акустики доступны в individual physical diagnostics; глобальный phonon RMSE не заменяет их.','',
        'Все12 моделей не дают мнимых мод ниже−0.001 THz в сохранённой ASR mesh50³ диагностике. Три seeds оценивают техническую воспроизводимость, а не неопределённость по всем возможным train/test наборам.','',
        '![N694 seed comparison](final-n694-grace-20261108/n694-seeds.png)','', '## GRACE на A100','',
        'GPU benchmark20989431 и анализ20989432 завершены (3м42с и20с). Native TensorPotential0.6.1 / TensorFlow2.20.0, TPModel/ASE diverse engine, main XLA включён, UQ/TF32 выключены. Два свежих процесса на точность, ABBA; 5 warmup и30 измерений на размер в каждом процессе. Полный latency включает подготовку соседей и синхронизацию результатов. FP32 — контролируемое преобразование закреплённого fp64 checkpoint; все44 тензора проверены после cast.','',
        '| Размер | fp32, ms | fp64, ms | Ускорение | Peak TF fp32, GiB | Peak TF fp64, GiB |','|---|---:|---:|---:|---:|---:|']
    for name,r in gpu['timing'].items():lines.append(f'| {name} | {r["fp32"]["median_ms"]:.2f} | {r["fp64"]["median_ms"]:.2f} | {r["fp32_speedup"]:.3f}× | {r["fp32"]["GPU_peak_bytes"]/2**30:.3f} | {r["fp64"]["GPU_peak_bytes"]/2**30:.3f} |')
    lines+=['','Peak относится к TensorFlow allocator в измеряемом интервале после сброса peak counters; включает живые параметры и кэши ранее пройденных размеров. Это не полный объём процесса из nvidia-smi. Память почти вдвое меньше у fp32, но ускорение полного ASE-вызова всего1.05–1.25×. На864 атомах model+sync занимает23.23ms/fp32 против35.50ms/fp64; подготовка соседей и прочие CPU-издержки уменьшают выигрыш полного вызова. Холодные вызовы с компиляцией8–14с исключены из steady timings.','',
        '| Разность fp32−fp64 | RMSE | Максимум |','|---|---:|---:|']
    for name,v in gpu['precision_delta'].items():lines.append(f'| {name} | {v["RMSE"]:.6g} | {v["max_abs"]:.6g} |')
    lines+=['','Проверены16 диагностических train-структур (8 геометрий × 2 XC), без validation/test подбора. Ни один результат не получен повторным использованием ASE result cache.','', '| XC | Phonon fp32−fp64 RMSE, THz | Максимум, THz |','|---|---:|---:|']
    for xc,r in gpu['harmonics'].items():lines.append(f'| {xc} | {r["fp32_minus_fp64"]["band_RMSE_THz"]:.6g} | {r["fp32_minus_fp64"]["band_max_abs_THz"]:.6g} |')
    lines+=['','![A100 performance and memory](final-n694-grace-20261108/grace-a100.png)','',
        '## Дальнейший режим','',
        'Для массового inference — fp32: меньше память, умеренное ускорение, различия сил и фононов на проверенном panel малы. Для строгих finite-difference и эталонных фононных проверок — fp64. На A100 fp64 не оказался вдвое медленнее: при216 атомах полный вызов дольше примерно на6%, при864 — на25%. Это измерения TensorFlow/ASE; LAMMPS/Kokkos throughput и длинная MD ещё не проверены.','',
        'Точность обучения этим inference benchmark не установлена. Первый GRACE reference pilot разумно сохранить в fp64, затем проверить fp32 коротким matched optimizer/profile запуском перед расширением: текущий тест не содержит backprop и Adam state. Обязательны fresh optimizer, проверка frozen tensors, GRACE-specific B0/H0 и train-only energy-reference alignment (особенно r²SCAN). Новые обучения GRACE данным запросом не отправлялись.','',
        'Исходный и derived checkpoints сохранены. Test149 и архивные49 не использовались в настройке. Все numerical значения и QC evidence: runs/dpa4-n694-seeds-20261107-01/comparison-20261108/comparison.json.']
    text='\n'.join(lines)+'\n'
    import re
    text=re.sub(r'(?<=[а-яА-ЯёЁ])(?=\d)', ' ',text)
    (ROOT/report).write_text(text,encoding='utf8')
    state=load(ROOT/'state.json');matrix=load(ROOT/'protocol/experiment_matrix.yaml')
    for p,name in [(ROOT/'state.json','state-before-final.json'),(ROOT/'protocol/experiment_matrix.yaml','matrix-before-final.json')]:
        if not (OUT/name).exists():(OUT/name).write_bytes(p.read_bytes())
    for name,c in cases.items():
        cc=c['case'];d=next(d for d in matrix['training_designs'] if d['architecture']=='DPA4-Neo' and d['train_size']==694 and d['training_seed']==cc['seed'] and d['functional']==cc['XC'] and d['adaptation']==cc['method'])
        d.update(state='completed',training_steps=34700,numerical_QC='PASS',model_sha256=c['model_sha256'],result=str(OUT/'comparison.json'),next_step='Three-seed N694 comparison complete; retain force/phonon tradeoff.')
    for d in matrix['training_designs']:
        if d['architecture']=='GRACE-2L':
            d['blocked_by']=[b for b in d['blocked_by'] if b!='GRACE_precision_benchmark']
            d['precision_selection']='inference_fp32_reference_diagnostics_fp64'
            d['training_precision_status']='fp64_reference_pilot_proposed_pending_B0_H0_optimizer_profile'
    state.update(updated_utc=result['utc'],stage='N694_three_seed_and_GRACE_A100_completed',status='all_DPA4_24_cases_completed_GRACE_A100_PASS',completed_training_jobs=24,fully_qualified_training_jobs=24,active_training_jobs=0,n694_final_comparison=str(OUT/'comparison.json'))
    state['grace_precision_benchmark'].update(GPU='PASS',selected_precision='fp32_for_bulk_inference_fp64_for_reference_checks',training_precision='fp64_reference_pilot_pending_optimizer_profile')
    state['n694_repeats_status_counts']=dict(fully_completed=8,running=0,pending=0)
    state['grace_preparation']['precision_selection']='A100 inference benchmark complete; fp32 bulk inference, fp64 reference checks; fp64 reference training pilot proposed, not launched'
    state['current_gates'].update(FT='N200_and_N694_three_seed_PASS; LoRA_force_full_phonon_tradeoff',GRACE_precision='A100_inference_PASS; training_precision_not_benchmarked')
    state['next_actions']=['GRACE B0/H0 and train-only energy alignment; fresh-optimizer/frozen-mask and timing/memory profile before N200 training.','Use full FT as DPA4 harmonic-priority candidate; retain LoRA force-priority comparison. Keep held-out test closed.']
    if report not in state['reports']:state['reports'].append(report)
    matrix['screening_summary'].update(completed_DPA4_N694_cases=12,active_DPA4_N694_cases=0,latest_result=report,GRACE_GPU_precision='PASS')
    dump(ROOT/'state.json',state);dump(ROOT/'protocol/experiment_matrix.yaml',matrix)
    (ROOT/'reports/summary.md').write_text('# Ti–Al FM: текущее состояние\n\n8 ноября 2026: N200 и N694 завершены по трём seeds, всего24 DPA4 модели; QC PASS. LoRA лучше по силам, full FT — по глобальным фононам. GRACE A100 fp32/fp64 benchmark завершён: fp32 ускоряет полный ASE-вызов в1.05–1.25 раза и примерно вдвое уменьшает TF GPU memory; численные различия малы. Training GRACE пока не запускался.\n\n[Итоговое сравнение](18_final_n694_grace_a100.md).\n',encoding='utf8')
    print(json.dumps(dict(status='PASS',N694_models=12,checkpoint_checks=checkpoints,GPU_grace='PASS',report=report)))

if __name__=='__main__':main()
