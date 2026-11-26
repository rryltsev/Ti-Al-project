from pathlib import Path
import json,time,hashlib
p=Path(__file__).resolve().parent;root=p.parents[3];camp=root/'audit/oct29/pbe15k_ti_temperature'
read=lambda f:json.loads(f.read_text(encoding='utf-8'))
x=read(p/'results.json');scan=read(p/'scan.json');jk=read(p/'grouped_errors.json');val=read(p/'numerical_validation.json')
for T in ['1873','2000']:
 rows=scan[T]['old20'];best=min(rows,key=lambda a:a['criterion_zscore'])
 x['temperatures'][T]['old_cutoff_sampled_minimum']={'fcut_THz':best['fcut']/1e12,'cutoff_z':best['criterion_zscore']}
x['boundary_note']='Threshold crossings are numerical approximations. Use interior 1.23-1.26 THz to avoid rounding at Z=2 boundaries. Tiny optimizer/truncation changes at boundaries do not alter this interior interval.'
x['practical_interior_THz']=[1.23,1.26]
for T in ['1873','2000']:
 rows={k:[r for r in scan[T][k] if 1.23e12<=r['fcut']<=1.26e12] for k in x['primary_groups']}
 assert all(max(r['cost_zscore'],r['criterion_zscore'])<2 for rr in rows.values() for r in rr)
(p/'results.json').write_text(json.dumps(x,indent=2)+'\n')
labels={'old20':'Старые 20 пс','new40':'Новые полные 40 пс','new_split20':'Новые 2 x 20 пс вместе','pooled20':'Объединённые 20-пс сегменты'}
lines=['Общий диапазон cutoff: старые, новые и объединённые данные PBE 15000','1 ноября 2026. Уточнение отдельного анализа pbe15k_legacy_pooling_v2.','',
'Данные неизменны: 25 старых + 10 новых реплик при 1873 K; 20 старых + 10 новых при 2000 K.',
'Проверяются два односторонних условия, как в protocol.json: cost Z < 2 и cutoff Z < 2.',
'Это не проверка всех условий остановки. N, neff, последовательная устойчивость и',
'согласие разных длительностей требуют отдельных проверок. Отрицательный Z не заменялся на |Z|.','',
'Скан: старый поднабор 0.6–8 ТГц с шагом 0.05; все шесть вариантов 0.8–1.8 ТГц',
'с шагом 0.01; границы пересечения уточнены численно, характеристики внутри',
'пересечений проверены с шагом 0.001. За пределами плотного скана проверены',
'дополнительные точки, в которых старый поднабор проходит.',
'Настройки Lorentz/CV2L и нормировка сохранены. Численные границы приблизительны;',
'значение ровно на границе соответствует Z=2 и не входит в строгое условие Z<2.','',
'Основной набор условий: старые 20 пс, новые полные 40 пс, новые два 20-пс окна',
'совместно и объединённый спектр должны одновременно проходить.','']
for T in ['1873','2000']:
 e=x['temperatures'][T]
 lines.append(f'{T} K: основные интервалы внутри исследованного диапазона: '+str(e['primary_intervals']))
 lines.append(f'{T} K: дополнительно обе новые половины по отдельности: '+str(e['strict_intervals']))
lines+=['','Общий для обеих температур основной интервал: примерно 1.2205–1.2654 ТГц.',
'Практический внутренний участок с запасом от округления границ: 1.23–1.26 ТГц.',
'Общего для обеих температур интервала с одновременным прохождением каждой',
'20-пс половины отдельно не найдено. При 1873 K первая половина имеет cutoff Z',
'примерно 2.50–2.81 внутри основного общего интервала.',
'Низкочастотное прохождение при 1873 K около 0.6–0.66 ТГц находится у нижней',
'границы сканирования и использует только около 13–14 эффективных частотных',
'точек 20-пс спектра; оно не является общим с 2000 K и не решает проблему.','',
'Изменение в общем основном интервале 1.2205–1.2654 ТГц.',
'Ниже все eta и стандартные ошибки в мПа·с. Ошибка в этой таблице формальная STACIE.']
for T,e in x['primary_shared_ranges'][0]['temperatures'].items():
 lines+=['',T+' K:']
 for k in labels:
  r=e[k];lines.append(f"  {labels[k]}: eta {r['eta_min']:.6f}–{r['eta_max']:.6f}; SE {r['formal_se_min']:.6f}–{r['formal_se_max']:.6f}; размах eta {r['eta_span_pct_of_midpoint']:.4f}%.")
lines+=['','В пределах собственного основного интервала каждой T (выше 1 ТГц):']
for T,e in x['temperatures'].items():
 for rr in e['primary_ranges']:
  if rr['interval'][0]<1:continue
  lines.append(f'{T} K, cutoff {rr["interval"][0]:.4f}–{rr["interval"][1]:.4f} ТГц:')
  for k in labels:
   r=rr['summaries'][k];lines.append(f"  {labels[k]}: eta {r['eta_min']:.6f}–{r['eta_max']:.6f}; SE {r['formal_se_min']:.6f}–{r['formal_se_max']:.6f}; размах eta {r['eta_span_pct_of_midpoint']:.4f}%.")
lines+=['','Проверка группировки ошибок в трёх внутренних точках общего интервала:',
str(jk['cutoffs_THz'])+' ТГц.',
'При удалении новой реплики обе её половины и все каналы удаляются вместе.',
'При объединении сохраняются исходные суммарные веса старой и новой групп.']
for T,rows in jk['temperatures'].items():
 lines.append(T+' K:')
 for k in ['old_se','new_split_se','pooled_se']:
  vv=[a['old_block_1'][k] for a in rows];lines.append(f'  Jackknife {k}: {min(vv):.6f}–{max(vv):.6f}.')
 vv=[a['new40_se'] for a in rows];lines.append(f'  Jackknife новых полных 40 пс: {min(vv):.6f}–{max(vv):.6f}.')
 vv=[a['old_block_5']['pooled_se'] for a in rows];lines.append(f'  Pooled SE, старые реплики удаляются соседними пятёрками: {min(vv):.6f}–{max(vv):.6f}.')
lines+=['Эти диапазоны jackknife относятся к трём опорным точкам, не являются точными',
'непрерывными экстремумами. Меньшая ошибка при блокировке старых 1873 K',
'не доказывает большей точности: остаётся всего 5 блоков; для 2000 K — 4 блока.','',
'Запас диагностики и ограничения:',
'При 2000 K минимум cutoff Z старых данных около 1.95; окно едва ниже порога 2.',
'Изменение eta внутри найденного узкого окна гораздо меньше статистической ошибки.',
'Однако окно найдено после просмотра тех же данных, поэтому не является',
'независимым подтверждением отсутствия систематики или гарантией покрытия ошибок.',
'В общем интервале neff около 25.5–26.5 у 20-пс спектров и 50.6–52.4 у 40-пс,',
'что меньше рабочего порога neff>=60. Также остаётся проблема первой половины',
'1873 K. Не менять по этому скану физический протокол и критерии остановки.',
'Сравнение трёх random seeds (1,42,123) в точках 1.23/1.25/1.26 ТГц для всех',
'шести вариантов не изменило ни eta, ни cutoff Z; критерии конечны, ошибок fit нет.','',
'Объединённые ориентиры в этом окне: около 3.200 и 2.830 мПа·с.',
'Формальные SE около 0.084 и 0.077; с групповыми проверками разумно помнить',
'SE около 0.082 для 1873 K и около 0.097 для 2000 K при блокировании старых стартов.',
'Это отдельный анализ, официальные новые 40-пс точки и рисунок 7 не заменены.','',
'Файлы: results.json (границы/диапазоны), scan.json (все fits), grouped_errors.json,',
'numerical_validation.json, common_cutoff.png/pdf; исходные скрипты в родительском каталоге.',
'Описание статистики/cutoff STACIE: https://molmod.github.io/stacie/theory/cutoff.html']
(p/'report_RU.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
manifest={f.name:sha(f) for f in p.iterdir() if f.is_file() and f.name!='manifest.json'}
manifest['source_scripts']={f.name:sha(f) for f in [p.parent/'scan_common_cutoff.py',p.parent/'check_common_errors.py',p.parent/'validate_common_cutoff.py']}
(p/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
m=read(camp/'monitor.json')
m['legacy_pooling_audit']['common_cutoff_followup']={'unix':time.time(),'report':str(p/'report_RU.txt'),'primary_shared_THz':x['primary_shared_between_temperatures'],'practical_interior_THz':[1.23,1.26],'strict_shared_including_each_half':False,'limiting_diagnostic':'1873 K new first20 cutoff Z 2.50-2.81 in common main interval; old2000 has only a small margin below 2. neff is also below 60.','pooled_eta_ranges_mPa_s':{T:[e['pooled20']['eta_min'],e['pooled20']['eta_max']] for T,e in x['primary_shared_ranges'][0]['temperatures'].items()},'production_protocol_changed':False}
tmp=camp/'monitor.json.tmp';tmp.write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');tmp.replace(camp/'monitor.json')
print('REPORT_AND_MONITOR_SAVED')

