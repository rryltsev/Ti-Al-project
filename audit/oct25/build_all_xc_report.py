"""Assemble a complete, self-contained checkpoint audit only after all checks finish."""
from pathlib import Path
import json,html,hashlib
HERE=Path(__file__).resolve().parent;OUT=HERE/'all_xc_audit';DEST=OUT/'analysis'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
cfg=read(OUT/'protocol.json');metrics=read(DEST/'metrics.json');prop=read(DEST/'properties.json');dos=read(DEST/'phonons.json')
extra=read(DEST/'extra_test_metrics.json');integrity=read(DEST/'input_integrity.json');matching=read(DEST/'cross_functional_geometry_matching.json')
assert not any(metrics['missing_predictions'].values()),metrics['missing_predictions']
assert not prop['missing'] and not extra['missing']
assert len(dos['available_curves'])==91,len(dos['available_curves'])
assert all(set(v)=={'train','validation','test'} for v in metrics['groups'].values())
tasks=read(OUT/'tasks.json');assert all((OUT/'results'/str(t['id'])/'summary.json').exists() for t in tasks)
cpu_seconds=0;elapsed=[]
for folder in [OUT,OUT/'extra_tests']:
    jid=read(folder/'submission.json')['job_id'];n=len(read(folder/'tasks.json'))
    records={s.split('|')[0]:s.split('|') for s in (folder/'accounting.txt').read_text().splitlines()}
    for i in range(n):
        r=records[jid+'_'+str(i)];assert r[1:3]==['COMPLETED','0:0'],r
        cpu_seconds+=int(r[5]);elapsed.append(r[3])
rows=[]
for key,m in cfg['models'].items():
    g=metrics['groups'][key]
    rows.append([m['functional'],f"{m['step']/1000:g}k",f"{g['train']['E_RMSE_meV_atom']:.3f}",
                 f"{g['validation']['E_RMSE_meV_atom']:.3f}",f"{g['test']['E_RMSE_meV_atom']:.3f}",
                 f"{g['test']['F_RMSE_eV_A']:.5f}",f"{g['test']['S_RMSE_GPa']:.4f}"])
late=[]
for key in ['PBE-15000','PBE-20000','PBE-23400','PBEsol-20000','r2SCAN-20000']:
    r=prop['liquids'][key];baseline=r['baseline'];rr=r['rows']
    volume=max(abs(100*(prop['crystals'][key][phase]['equilibrium']['V_A3_atom']/prop['crystals'][baseline][phase]['equilibrium']['V_A3_atom']-1)) for phase in ['Ti_hcp','Ti_bcc','Al_fcc'])
    gap=lambda k:1000*(prop['crystals'][k]['Ti_bcc']['equilibrium']['E_eV_atom']-prop['crystals'][k]['Ti_hcp']['equilibrium']['E_eV_atom'])
    late.append({'key':key,'baseline':baseline,'max_mixing_energy_change_kJ_mol':max(abs(z['mixing_energy_change_kJ_mol']['mean']) for z in rr),
                 'max_pressure_change_GPa':max(abs(z['delta_P_GPa']['mean']) for z in rr),
                 'max_relative_force_change_percent':100*max(z['relative_delta_F_RMS'] for z in rr),
                 'max_equilibrium_volume_change_percent':volume,'bcc_hcp_gap_change_meV_atom':gap(key)-gap(baseline),
                 'DOS_fixed_vs_article':dos['vs_article'][key+'/fixed/MP40/0.05'],
                 'DOS_equilibrium_vs_article':dos['vs_article'][key+'/equilibrium/MP40/0.05']})
summary={'status':'complete','models':15,'main_tasks':len(tasks),'extra_tasks':10,'new_cpu_core_hours':cpu_seconds/3600,
 'test_common':20,'test_additional_PBE':12,'test_additional_PBEsol':11,
 'new_train_validation_predictions':sum(cfg['datasets'][x]['counts']['train']+cfg['datasets'][x]['counts']['validation'] for x in ['PBEsol','r2SCAN'])*5,
 'liquid_frames_per_checkpoint':250,'checkpoint_changes':late,
 'recommendation':'Prioritize PBE sensitivity MD comparing 15k/20k to article 5k. PBEsol 20k is an energy-accuracy candidate, but no blanket property rerun is justified by the small paired changes. Retain r2SCAN final unless subsequent same-functional DFT evidence changes the assessment. No automatic model substitution.'}
(DEST/'summary.json').write_text(json.dumps(summary,indent=2))
lines=['АУДИТ CHECKPOINT DPA4: PBE, PBEsol, r²SCAN — 25.10.2026',
       'Все запрошенные CPU-расчёты завершены. Модели статьи и прежние MD-результаты не заменены.',
       '', '1. Охват и воспроизводимость',
       'Проверены все 15 сохранённых checkpoint: 5k, 10k, 15k, 20k и финальный для каждого функционала.',
       'Финальные шаги: PBE 23400, PBEsol 23100, r²SCAN 23200; модели статьи: PBE 5k, PBEsol 23.1k, r²SCAN 23.2k.',
       'Общий test set: 20 одинаковых жидких конфигураций, каждая модель сравнивается со своим DFT-функционалом.',
       'Дополнительно: 12 готовых PBE и 11 PBEsol конфигураций; всего 32/31 соответственно. Для r²SCAN пока 20.',
       'Полные train/validation: PBE 850/149, PBEsol 849/149, r²SCAN 845/148.',
       'E(V) для α-Ti, β-Ti и fcc Al; α-Ti c/a оптимизировано при каждом объёме; кубическая симметрия β-Ti/Al сохранена.',
       'Гармонические DOS α-Ti: общая DFT-геометрия и собственная равновесная ячейка каждого checkpoint; суперячейка 9³, смещение 0.01 Å.',
       'Сравнение энергий, сил, давлений: x_Al=0,0.1,0.3,0.5,1 при 2000 K, 50 кадров на состав, на собственном архивном ансамбле каждого функционала.',
       f"Новые расчёты: 63 основные и 10 дополнительных задач, {cpu_seconds/3600:.2f} выделенных ядро-часа; GPU не использовались.",
       'Предыдущие результаты PBE повторно использованы после сверки хешей моделей, геометрий и DFT-меток.',
       '', '2. Learning curves и целостность данных',
       'Конфиги подтверждают одинаковые основные настройки: LoRA rank=16, alpha=16, 50 эпох, seed=10, HybridMuon, cosine LR 1e-4 -> 1e-6, веса E/F/virial=20/20/5.',
       'В loss включён f_use_norm: логируемая MAE сил использует нормы векторов; это не компонентная RMSE сил из parity-анализа.',
       'Разрыв train/validation сам по себе не критерий остановки. Важны абсолютные validation-ошибки и их устойчивость по checkpoint.',
       'Кривые шумны; на рисунке показаны сырые значения и скользящие медианы по 11 точкам. Падение test/validation-ошибок позднее 5k не поддерживает универсальный тезис о переобучении после 5k.',
       'Полная validation-выборка PBE: RMSE сил 0.06669 (5k), 0.06487 (15k), 0.06472 (20k), 0.06468 (23.4k) эВ/Å. Разрыв с train у 5k действительно меньше, но абсолютная validation-ошибка выше; это не обосновывает early stopping на 5k.',
       'Для PBEsol между 20k и 23.1k энергетическая RMSE растёт и на train (0.890 -> 1.664), и на validation (0.995 -> 1.843 мэВ/атом). Поэтому ухудшение энергии последнего checkpoint нельзя объяснять одним только классическим переобучением; точность сил практически не меняется.',
       'Разбиения train/validation между функционалами НЕ одинаковы. При приближённом сопоставлении с PBE:',
       f"  PBEsol: {matching['PBEsol']['matched']} сопоставленных структур, {matching['PBEsol']['different_split']} в другой части набора.",
       f"  r²SCAN: {matching['r2SCAN']['matched']} сопоставленных структур, {matching['r2SCAN']['different_split']} в другой части набора.",
       f"  PBEsol/r²SCAN: {integrity['pairwise_exact_geometry_matching']['PBEsol/r2SCAN']['identical_geometries']} точно совпадающих геометрий, {integrity['pairwise_exact_geometry_matching']['PBEsol/r2SCAN']['different_split']} распределены по-разному.",
       'Поэтому нельзя интерпретировать различие validation-кривых функционалов как сравнение на одних и тех же структурах.',
       'Точных совпадений геометрий между test и train/validation, а также между train и validation внутри функционала не найдено. Это не доказывает независимость исходных траекторий.',
       '', '3. Метрики: без подгонки энергетических сдвигов и без удаления точек',
       'XC | checkpoint | E RMSE train | E RMSE val | E RMSE test | F RMSE test | stress RMSE test',
       'Энергия: мэВ/атом; силы: эВ/Å, декартовы компоненты; напряжения: ГПа, шесть независимых компонент.']
lines+=[' | '.join(r) for r in rows]
lines+=['', 'Парный bootstrap по текущим 20 конфигурациям:',
         'PBE 15k и 20k улучшают E/F/stress относительно 5k на текущем общем тесте.',
         'PBEsol 20k снижает энергетическую ошибку относительно 23.1k; различия ошибок сил и напряжений в пределах парной неопределённости.',
         'r²SCAN 20k и 23.2k имеют практически одинаковые ошибки энергий и сил; оснований откатываться к ранним checkpoint по этим данным нет.',
         'Доверительные интервалы условны на текущем наборе конфигураций, не учитывают независимые обучающие запуски и корреляции между исходными траекториями.',
         '', '4. Дополнительные тестовые структуры']
for key in ['PBE-5000','PBE-15000','PBE-20000','PBE-23400','PBEsol-20000','PBEsol-23100']:
    g=extra['groups'][key];lines.append(f"{key}: E RMSE на всех готовых = {g['all_available']['E_RMSE_meV_atom']:.3f}; на дополнительных кристаллах = {g['extra_crystals']['E_RMSE_meV_atom']:.3f} мэВ/атом.")
lines+=['Дополнительные наборы разных функционалов различаются по покрытию. Их средние ошибки не используются как основной межфункциональный benchmark.',
         '', '5. Фононные DOS',
         'Для всех 15 checkpoint рассчитаны спектры при двух геометриях. Проверены сетки MP40x40x30, MP50³, Gamma40x40x30 и ширины 0.05/0.10/0.15 THz.',
         'Сохранённый DFT-DOS относится к PBE. Близость PBEsol/r²SCAN к этой кривой не является ошибкой относительно собственного функционала.',
         'Для выбора checkpoint PBE фононный критерий не совпадает с энергетическим: 15k лучше по интегральным спектральным метрикам, 20k лучше по энергиям тестовых структур.',
         'Относительная L2-разница с PBE-DFT, ширина 0.05 THz, MP40x40x30; общая/собственная ячейка:']
for key in cfg['models']:
    a=dos['width_checks']['0.05'][key+'/fixed/MP40']['L2_relative'];b=dos['width_checks']['0.05'][key+'/equilibrium/MP40']['L2_relative']
    lines.append(f'{key}: {a:.4f} / {b:.4f}')
lines+=['Фононы DFT построены в суперячейке 4x4x3, MLIP — 9³; неизвестен исходный полный DFT q-grid. Фиксация ячейки не устраняет эти ограничения.',
         'Гармонические фононы α-Ti не определяют свободную энергию β-фазы или температуру плавления.',
         '', '6. Изменения относительно checkpoint статьи',
         'Максимумы ниже относятся только к пяти проверенным составам и трём кристаллическим структурам.']
for r in late:
    lines.append(f"{r['key']} vs {r['baseline']}: |ΔE_mix|≤{r['max_mixing_energy_change_kJ_mol']:.4f} кДж/моль, |ΔP|≤{r['max_pressure_change_GPa']:.4f} ГПа, RMS ΔF/F≤{r['max_relative_force_change_percent']:.3f}%, |ΔV0/V0|≤{r['max_equilibrium_volume_change_percent']:.4f}%, Δ(Eβ−Eα)={r['bcc_hcp_gap_change_meV_atom']:+.3f} мэВ/атом.")
lines+=['ΔE_mix — изменение средней потенциальной энергии с вычитанием линейного вклада чистых компонентов на прежнем ансамбле. Это не новая равновесная энтальпия смешения.',
         'Давление конфигурационное; в парной разности кинетический вклад сокращается. Для каждого checkpoint использованы одинаковые кадры внутри функционала.',
         'Блочные SEM получены из 5 и 10 временных блоков одной траектории; независимые повторы не выполнены.',
         'Соответствие MD-моделей проверено по сохранённым энергиям/силам/напряжениям: все три checkpoint статьи воспроизводят архив с численной точностью.',
         '', '7. Решение о дальнейших расчётах',
         'PBE: приоритетная контрольная MD. Сравнить 15k и 20k с существующим 5k: сначала плотность и энтальпия при 2000 K для Ti, Al, Ti0.7Al0.3, Ti0.5Al0.5; затем уточнить плавление для выбранного checkpoint.',
         'Не выбирать 15k единственным победителем только по DOS и не выбирать 20k только по энергии: аудит выявил компромисс. Единый критерий выбора следует зафиксировать до сравнения с экспериментальными MD-свойствами.',
         'PBEsol: 20k — кандидат по энергетическим ошибкам, но парные изменения смешения и сил малы. Полный пересчёт всех свойств сейчас не обоснован; при необходимости достаточно ограниченной проверки плотности/смешения.',
         'r²SCAN: сохранить финальный 23.2k. Сравнение с 20k служит оценкой поздней чувствительности; массовый пересчёт по текущим данным не требуется.',
         'Если checkpoint заменён, метрики и все связанные MD-результаты должны относиться к тому же файлу модели. Нельзя подставить метрики 20k к свойствам, рассчитанным 23.1k.',
         'После использования текущего test set для выбора checkpoint он становится диагностическим набором разработки. Новые, ещё не использованные тестовые конфигурации следует сохранить для финальной проверки.',
         'Новая MD и замена моделей в статье в рамках этого аудита не выполнялись.',
         '', 'Файлы: learning_curves, checkpoint_metrics, parity_E/F/S, test_energy_residuals, extra_hcp_test_residuals, eos_all_checkpoints, liquid_checkpoint_sensitivity, phonon_dos_all_checkpoints (PNG/PDF).',
         'Машиночитаемые результаты: metrics.json, extra_test_metrics.json, properties.json, phonons.json, summary.json; исходные результаты и протоколы в родительской папке.']
text='\n'.join(lines)+'\n';(DEST/'REPORT_RU.txt').write_text(text,encoding='utf-8')
def table(headers,body):return '<table><thead><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in r)+'</tr>' for r in body)+'</tbody></table>'
images=[('learning_curves','Learning curves'),('checkpoint_metrics','Полные train/validation и общий test set'),
        ('test_energy_residuals','Систематические ошибки энергии'),('extra_hcp_test_residuals','Дополнительные DFT-тесты α-Ti'),
        ('phonon_dos_all_checkpoints','Фононные DOS'),('eos_all_checkpoints','Кривые E(V)'),
        ('liquid_checkpoint_sensitivity','Чувствительность жидких свойств'),('parity_E','Parity: энергии'),('parity_F','Parity: силы'),('parity_S','Parity: напряжения')]
page='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>TiAl — аудит checkpoint</title><style>body{font:16px/1.55 system-ui,sans-serif;max-width:1180px;margin:32px auto;padding:0 20px;color:#213244;background:#fafbfc}h1,h2{line-height:1.2}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border-bottom:1px solid #dce1e6;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#e9eef3}img{width:100%;height:auto;background:white}details{margin:22px 0;padding:14px;background:white;border:1px solid #dce1e6}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;font:14px/1.6 ui-monospace,monospace}a{color:#246a9f}.note{padding:18px;background:#e9f1f7}</style><body><h1>TiAl: аудит 15 checkpoint</h1>'''
page+=f'<p>25.10.2026 · CPU · {cpu_seconds/3600:.2f} ядро-часа новых расчётов · 20 общих DFT-тестов</p>'
page+='<p class="note">Приоритет — проверить чувствительность MD для PBE (15k и 20k против 5k). Для PBEsol и r²SCAN аудит не обосновывает автоматический полный пересчёт всех свойств. DFT-эталон фононов относится только к PBE.</p>'
page+='<h2>Ошибки на полных наборах</h2><p>E: мэВ/атом; F: эВ/Å; S: ГПа. Ошибки каждого функционала относительно собственного DFT.</p>'
page+=table(['Функционал','Checkpoint','E train','E validation','E test','F test','S test'],rows)
page+='<p><a href="REPORT_RU.txt">Полный текст отчёта</a> · <a href="summary.json">Сводные данные JSON</a></p>'
for stem,title in images:page+=f'<details><summary>{html.escape(title)}</summary><p><a href="{stem}.pdf">PDF</a></p><img src="{stem}.png" alt="{html.escape(title)}"></details>'
page+='<details><summary>Интерпретация, контроль качества и ограничения</summary><pre>'+html.escape(text)+'</pre></details></body></html>'
(DEST/'REPORT.html').write_text(page,encoding='utf-8')
manifest={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in DEST.iterdir() if p.is_file() and p.name!='report_manifest.json'}
(DEST/'report_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(summary,indent=2))
