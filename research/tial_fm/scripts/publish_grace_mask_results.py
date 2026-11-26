"""Verify collected mask ablations independently and publish the seed-10 comparison."""
from pathlib import Path
import datetime
import numpy as np
import matplotlib.pyplot as plt
from compare_grace_n200 import ROOT, load, dump, sha, recompute, phonons

RUN = ROOT/'runs/grace-n200-mask-ablation-20261109-01'
PARENT = ROOT/'runs/grace-n200-fp64-pilot-20261109-02'
OUT = RUN/'comparison-local'
FIG = ROOT/'reports/grace-mask-ablation-final'
METHODS = ['full', 'restricted', 'restricted_mlp', 'restricted_radial']
LABELS = ['Full', 'Restricted', '+ MLP', '+ Radial']
COLORS = ['#167a9e', '#d47729', '#39966a', '#8356a5']


def main():
    OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
    aggregate = load(RUN/'collected-final/comparison/summary.json')
    assert aggregate['status'] == 'PASS'
    subsets = load(ROOT/'datasets/nested-train-v1-20261106/subsets.json')
    reference = np.load(RUN/'package/reference_figure.npz')
    counts = {}; cases = {}; folders = {}; batches = {}; checks = 0
    for run in [PARENT, RUN]:
        data = run/'collected-final'
        hashes = load(data/'COLLECTION_SHA256.json')
        for name, expected in hashes.items():
            assert sha(data/name) == expected, name
        counts[run.name] = len(hashes)
        remote = load(run/'final-remote-verification.json')
        assert remote['status'] == 'PASS'
        for c in load(run/'package/plan.json')['cases']:
            name = c['name']; xc = c['XC']; w = data/'cases'/name
            folders[name] = w
            t = load(w/'training-summary.json'); v = remote['cases'][name]
            assert t['status'] == 'PASS' and t['steps'] == 10000 and t['epochs'] == 50 and not t['test_access']
            assert sha(w/'training-summary.json') == v['training_summary_sha256']
            assert sha(w/'seed/10/model.yaml') == v['model_yaml_sha256']
            assert t['hashes'] == v['checkpoint'] == aggregate['cases'][name]['checkpoint_hashes']
            for f in ['evaluation-summary.json', 'numerical-QC.json', 'zero-step.json', 'checkpoint-path-check.json', 'harmonics/summary.json']:
                assert load(w/f)['status'] == 'PASS', (name, f)
            assert load(w/'validation-selection.json')['best_checkpoint_reload_skipped']
            steps = [load(p) for p in (w/'checks').glob('*.json')]
            assert {s['steps'] for s in steps} == {1, *range(1000, 10001, 1000)}
            assert all(s['status'] == 'PASS' and not s['frozen_changed'] and s['optimizer_iterations'] == s['steps'] for s in steps)
            checks += len(steps)
            batch = next(s for s in steps if s['steps'] == 10000)['first5_batch_hashes']
            assert len(batch) == 5
            if xc in batches: assert batch == batches[xc]
            batches[xc] = batch
            metrics = recompute(load(w/'per-structure.json'), load(w/'metrics.json'), subsets)
            h = load(w/'harmonics/summary.json')['results'][xc]
            ph = phonons(w/'harmonics'/xc, xc, h, reference)
            for part, values in metrics.items():
                for key, value in values.items():
                    assert np.isclose(value, aggregate['cases'][name]['metrics'][part][key], rtol=1e-11, atol=1e-12)
            for key, value in aggregate['cases'][name]['phonons'].items():
                assert np.isclose(value, ph[key], rtol=1e-11, atol=1e-12), (name, key)
            inv = load(w/'parameter-inventory.json')
            expected = c.get('expected_trainable', 21764956 if c['method'] == 'full' else 21111690)
            assert inv['trainable'] == expected
            cases[name] = dict(case=c, metrics=metrics, phonons=ph, checkpoint_hashes=t['hashes'],
                               trainable=expected, training_seconds=t['training_seconds'], checks=len(steps))
    result = dict(status='PASS', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), cases=cases,
                  comparisons=aggregate['comparisons'], checks=dict(collected_files=counts, models=len(cases),
                  parameter_checks=checks, identical_first5_batches_within_XC=True, frozen_split_checked=True,
                  train_only_alignment=True, test_access=False, seed=10, seed_count=1))
    assert len(cases) == 8 and checks == 88
    dump(OUT/'comparison.json', result)
    plt.rcParams.update({'font.size':9, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), constrained_layout=True)
    specifications = [('E_aligned_RMSE_meV_atom', 'Energy aligned (meV/atom)'),
                      ('F_RMSE_eV_A', 'Force (eV/Å)'), ('stress_RMSE_GPa', 'Stress (GPa)'),
                      ('band_RMSE_THz', 'Phonon bands (THz)')]
    for i, xc in enumerate(['PBE', 'r2SCAN']):
        for j, (key, title) in enumerate(specifications):
            ax = axes[i,j]
            ys = [(cases[f'{xc}-{m}-N200-seed10']['phonons'] if j == 3 else cases[f'{xc}-{m}-N200-seed10']['metrics']['validation'])[key] for m in METHODS]
            ax.bar(range(4), ys, color=COLORS, width=.7)
            ax.set_xticks(range(4), LABELS, rotation=20, ha='right')
            ax.set(title=xc+' · '+title, ylim=(0,max(ys)*1.22)); ax.grid(axis='y', alpha=.2)
            for x,y in enumerate(ys): ax.text(x,y*1.025,f'{y:.4f}',ha='center',va='bottom',fontsize=8)
    fig.suptitle('GRACE · train 200 / validation 149 · fp64 · seed 10 · final epoch 50')
    for ext in ['png','pdf']: fig.savefig(FIG/f'validation.{ext}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(13,8),gridspec_kw={'width_ratios':[2.1,1]},constrained_layout=True)
    x = reference['path_coordinate']
    breaks = np.r_[0,np.flatnonzero(np.linalg.norm(np.diff(reference['qpoints'],axis=0),axis=1)>.05)+1,len(x)]
    for i, xc in enumerate(['PBE','r2SCAN']):
        ax, ad = axes[i]
        for m, label, color in zip(['DFT']+METHODS, ['DFT']+LABELS, ['black']+COLORS):
            style = '--' if m == 'DFT' else '-'
            if m == 'DFT':
                freq = reference[xc+'_DFT_bands']; df, dd = reference['frequency_THz'], reference[xc+'_DFT_DOS']
            else:
                folder = folders[f'{xc}-{m}-N200-seed10']/'harmonics'/xc
                freq = np.load(folder/'bands.npz')['frequencies']
                dos = np.load(folder/'dos.npz'); df, dd = dos['frequency'],dos['dos']
            for start,end in zip(breaks[:-1],breaks[1:]):
                for band in range(6):
                    ax.plot(x[start:end],freq[start:end,band],color=color,ls=style,lw=1,alpha=.85,label=label if start==0 and band==0 else None)
            ad.plot(dd,df,color=color,ls=style,lw=1)
        ax.set(title=xc+' · matched DFT geometry',xlabel='Archived band-path coordinate',ylabel='Frequency (THz)',ylim=(-.2,9)); ax.legend(ncol=3,fontsize=8); ax.grid(alpha=.15)
        ad.set(xlabel='DOS (modes / atom / THz)',ylim=(-.2,9)); ad.grid(alpha=.15)
    for ext in ['png','pdf']: fig.savefig(FIG/f'phonons.{ext}',dpi=180)
    plt.close(fig)
    lines = ['# GRACE N200: full, restricted, restricted + MLP, restricted + radial','',
             f'Проверено {result["utc"]}. Повторы 20990790_1–3 COMPLETED, ExitCode 0:0; сохранён успешный PBE + MLP 20990675_0. Фононы 20990676_0–3 и сравнение 20990677 COMPLETED. Все восемь моделей: 10000 шагов / 50 эпох, fp64, seed 10. Optimizer, маски, numerical QC, export и фононы PASS.','',
             '| XC | FT | E raw, meV/atom | E aligned, meV/atom | F, eV/Å | Stress, GPa | Bands, THz |',
             '|---|---|---:|---:|---:|---:|---:|']
    for xc in ['PBE','r2SCAN']:
        for m in METHODS:
            c=cases[f'{xc}-{m}-N200-seed10'];v=c['metrics']['validation']
            lines.append(f'| {xc} | {m} | {v["E_raw_RMSE_meV_atom"]:.4f} | {v["E_aligned_RMSE_meV_atom"]:.4f} | {v["F_RMSE_eV_A"]:.6f} | {v["stress_RMSE_GPa"]:.6f} | {c["phonons"]["band_RMSE_THz"]:.6f} |')
    lines += ['', 'Все значения — RMSE. Energy: равный вес структур, на атом; F: все компоненты с весом по числу атомов; stress: шесть компонент с равным весом структур. Raw FT уже включает исходную train-only поправку до обучения; aligned дополнительно устраняет остаточный композиционный сдвиг финальной модели, оценённый строго по train200.', '',
              '![Validation](grace-mask-ablation-final/validation.png)', '',
              'В seed10 добавление энергетической MLP не улучшает ни одну из четырёх метрик относительно restricted для обоих XC. Изменения сил и stress очень малы; без повторов нельзя отличить слабый эффект от разброса.', '',
              'Размораживание радиальных сетей снижает F RMSE на 2.03% (PBE) / 2.24% (r²SCAN), ошибку фононов на 15.87% / 16.81%. Цена: aligned E хуже на 4.66% / 1.98%, stress хуже на 2.83% / 3.69%. Full остаётся лучшим по F, stress и общей ошибке фононов для обоих XC; исходный restricted — лучший по raw и aligned E.', '',
              '## Оптические моды в Γ', '', '| XC | FT | Дублет, THz | Синглет, THz | Ошибка синглета, THz |', '|---|---|---:|---:|---:|']
    for xc in ['PBE','r2SCAN']:
        dft=reference[xc+'_DFT_bands'][0]
        lines.append(f'| {xc} | DFT | {dft[3:5].mean():.4f} | {dft[5]:.4f} | — |')
        for m in METHODS:
            p=cases[f'{xc}-{m}-N200-seed10']['phonons']
            lines.append(f'| {xc} | {m} | {p["gamma_doublet_THz"]:.4f} | {p["gamma_singlet_THz"]:.4f} | {p["gamma_singlet_abs_error_THz"]:.4f} |')
    assert all(c['phonons']['imaginary_modes_weighted']==0 for c in cases.values())
    lines += ['', 'Радиальные сети уменьшают завышение Γ-синглета: PBE 1.058 → 0.542 THz, r²SCAN 1.481 → 0.874 THz; full даёт 0.029 / 0.104 THz. MLP это отклонение не исправляет. Это согласуется с важной ролью радиального представления в наблюдаемой ошибке кривизны, но не доказывает единственный механизм её возникновения.', '',
              '![Phonons](grace-mask-ablation-final/phonons.png)', '',
              'У всех восьми моделей нет мод ниже −0.001 THz на mesh50³. Сравнение на фиксированных XC-specific DFT-ячейках: central ±0.01 Å, symfc/ASR, 604 q-точки, шесть отсортированных частот в каждой точке; eigenvector tracking не выполнялся. DOS нормирована до трёх мод на атом. Это development-диагностика, не проверка сходимости DFT или длительной MD.', '',
              '## Контроль эксперимента', '',
              f'Локально проверены SHA-256 {sum(counts.values())} файлов двух сборов; validation/train метрики независимо пересчитаны из по-структурных ошибок, фононы — из bands/mesh/DOS. Проверены восемь финальных checkpoint по remote provenance и {checks} проверки параметров; frozen tensors не менялись. Первые пять batches совпадают между четырьмя масками внутри каждого XC, train200/validation149 сверены с frozen split. Test149 и архивные49 не использованы.', '',
              'Restricted обучает I2/reducing_, rho/reducing_, I1/reducing_: 21111690 номинальных параметров. +MLP добавляет LinMLPOut2ScalarTarget_ (1088 параметров); +radial добавляет R_Linear_ и R1_Linear_ (33152). Full: 21764956. Номинальная доля restricted около 97% не отражает эффективную адаптацию двух элементов: базовые 12 тензоров имеют 89-element axis; Al/Ti-блоки содержат 474420 коэффициентов. Поэлементный аудит изменённых строк отдельно не выполнялся.', '',
              'Все научные настройки совпадают с parent pilot: seed10, источник, train-only alignment, Adam, LR 1e-3→1e-4, batch1, loss и 50 эпох. После Disk quota exceeded три запуска начаты заново с исходных выровненных весов и нового Adam. Изменена только политика дисковой записи: отключены неиспользуемые best-validation checkpoint, сохранены регулярные optimizer checkpoint и финальные модели, добавлены quota guards. Выбор во всех моделях — фиксированная финальная эпоха50; best validation не загружается. История неудачных попыток сохранена в отчёте24.', '',
              'Ограничения: один seed каждой конфигурации, поэтому статистической устойчивости различий пока нет. При утраченных liquid trajectories гарантия группового разделения условна и основана на документированном протоколе. Для PBE остаётся ранее отмеченное ограничение происхождения поля DFT energy; raw-аудит r²SCAN выполнен.', '',
              'Рекомендация: проверить seeds20/30 для full, restricted и restricted + radial на обоих XC. +MLP пока имеет низкий приоритет. Затем контроль N694; на основании одного seed исключать варианты окончательно преждевременно. Новые обучения в этом шаге не запускались.', '',
              'Машиночитаемый результат: `runs/grace-n200-mask-ablation-20261109-01/comparison-local/comparison.json`. Сбор: `collected-final/COLLECTION_SHA256.json`; актуальная проверка моделей: `final-remote-verification.json`.']
    report='reports/25_grace_mask_ablation_results.md'
    (ROOT/report).write_text('\n'.join(lines)+'\n',encoding='utf8')
    for f in ['state.json','protocol/experiment_matrix.yaml']:
        backup=OUT/(Path(f).name+'.before')
        if not backup.exists(): backup.write_bytes((ROOT/f).read_bytes())
    state=load(ROOT/'state.json')
    state.update(status='GRACE_mask_ablation_complete',stage='GRACE_N200_four_masks_seed10_comparison',updated_utc=result['utc'],active_training_jobs=0,completed_training_jobs=32,fully_qualified_training_jobs=32)
    state['grace_mask_ablation'].update(status='complete',completed_cases=4,optimizer='PASS',harmonics='PASS',comparison='PASS',results=str(OUT/'comparison.json'),report=report)
    state['current_gates'].update(GRACE_mask_optimizer='PASS',GRACE_mask_final_QC='PASS',GRACE_mask_harmonics='PASS',GRACE_mask_comparison='PASS')
    state['next_actions']=['Replicate GRACE N200 full/restricted/restricted_radial for both XC with seeds20/30; MLP lower priority based on seed10 only.','Then full-train N694 controls; preserve test149 and archival49.']
    if report not in state['reports']: state['reports'].append(report)
    dump(ROOT/'state.json',state)
    matrix=load(ROOT/'protocol/experiment_matrix.yaml')
    for d in matrix['training_designs']:
        if d['architecture']=='GRACE-2L' and d['adaptation'] in ['restricted_mlp','restricted_radial'] and d['train_size']==200:
            name=f'{d["functional"]}-{d["adaptation"]}-N200-seed10'
            d.update(state='completed_qualified',blocked_by=[],launch_ready=False,results=str(OUT/'comparison.json'),final_checkpoint_hashes=cases[name]['checkpoint_hashes'])
    dump(ROOT/'protocol/experiment_matrix.yaml',matrix)
    (ROOT/'reports/summary.md').write_text('# Ti–Al FM: текущее состояние\n\nGRACE N200 fp64: все 8 моделей seed10, фононы и сравнение завершены, PASS. Full лучше по F/stress/phonons; restricted лучше по энергии. +Radial улучшает силы/фононы относительно restricted, но уступает full; +MLP улучшений не показал. Разброс seeds пока не измерен.\n\n[Четыре маски GRACE](25_grace_mask_ablation_results.md) · [Исходные full/restricted](22_grace_n200_results.md) · [DPA4 N694 и A100 precision](18_final_n694_grace_a100.md).\n',encoding='utf8')
    print(result['checks']); print(ROOT/report)


if __name__=='__main__': main()
