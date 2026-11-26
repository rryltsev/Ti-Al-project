from pathlib import Path
import sys,json,datetime
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
sys.path.insert(0,str(ROOT/'audit/melting'))
from summarize_melting import order,read_dump,slab_width,THRESHOLDS,np,plt
run=BASE/'pbe15k_melting/attempt2';latest=json.loads((run/'latest-snapshot.json').read_text());out=Path(latest['directory']);meta=json.loads((out/'snapshot.json').read_text())
cases={};results={}
for key,d in meta['dumps'].items():
 rows=[]
 for step,cell,xyz,pe in read_dump(Path(d['local_file'])):
  q,qb,bonds=order(xyz,cell);bins=np.minimum((xyz[:,2]/cell[2]*36).astype(int),35)
  profile=np.bincount(bins,weights=qb,minlength=36)/np.maximum(np.bincount(bins,minlength=36),1)
  rows.append((step*.001,(qb[:,None]>THRESHOLDS).mean(axis=0),slab_width(profile),profile,(bonds>=7).mean()))
 a={'time':np.array([r[0] for r in rows]),'fraction':np.array([r[1] for r in rows]),'width':np.array([r[2] for r in rows]),'profile':np.array([r[3] for r in rows]),'bonds_fraction':np.array([r[4] for r in rows])}
 cases[key]=a;np.savez_compressed(out/(key.replace('/','_')+'_order.npz'),**a)
 results[key]={'time_ps':a['time'][-1],'solid_fraction_first_last':a['fraction'][[0,-1],1].tolist(),'width_fraction_first_last':a['width'][[0,-1],0].tolist(),'thresholds':THRESHOLDS.tolist(),'fraction_first_last_all_thresholds':a['fraction'][[0,-1]].tolist(),'bond_fraction_first_last':a['bonds_fraction'][[0,-1]].tolist()}
old={}
for T in [1700,1800]:
 with np.load(ROOT/f'audit/melting/pbe_{T}_001.npz') as f:old[T]={'time':f['step']*.001,'fraction':f['solid_fraction'][:,1],'width':np.array([slab_width(p)[0] for p in f['profile']])}
 idx=np.argmin(abs(old[T]['time']-8));results[f'5k_{T}K']={'time_ps':float(old[T]['time'][idx]),'solid_fraction_first_last':old[T]['fraction'][[0,idx]].tolist(),'width_fraction_first_last':old[T]['width'][[0,idx]].tolist()}
fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for key,a in cases.items():
 label='15k '+key.split('/')[0][1:];axs[0].plot(a['time'],a['fraction'][:,1],'o-',ms=3,label=label);axs[1].plot(a['time'],a['width'][:,0],'o-',ms=3,label=label)
for T,a in old.items():
 axs[0].plot(a['time'],a['fraction'],'--',label=f'5k {T} K');axs[1].plot(a['time'],a['width'],'--',label=f'5k {T} K')
for ax in axs:ax.set_xlim(0,10);ax.set_ylim(0,1);ax.set_xlabel('Time after preparation (ps)');ax.legend(fontsize=8);ax.grid(alpha=.2)
axs[0].set_ylabel('Solid-like fraction (qbar6 > 0.25)');axs[1].set_ylabel('Largest solid slab width / Lz');fig.suptitle('PRELIMINARY: first 8–8.5 ps of planned 50 ps; 5k has no 1850 K trajectory')
fig.savefig(out/'early_interfaces.png',bbox_inches='tight');fig.savefig(out/'early_interfaces.pdf',bbox_inches='tight');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,(key,a) in zip(axs,cases.items()):
 mesh=ax.pcolormesh(a['time'],(np.arange(36)+.5)/36,a['profile'].T,shading='nearest',vmin=.08,vmax=.4,cmap='viridis')
 ax.plot(a['time'],a['width'][:,1],color='white',lw=1);ax.plot(a['time'],a['width'][:,2],color='white',lw=1);ax.set_title('15k '+key.split('/')[0]);ax.set_xlabel('Time (ps)');ax.set_ylabel('z / Lz')
fig.colorbar(mesh,ax=axs,label='Mean qbar6 in z slabs');fig.suptitle('PRELIMINARY spatial order profiles; white = smoothed qbar6=0.25 boundaries')
fig.savefig(out/'early_interface_maps.png',bbox_inches='tight');plt.close(fig)
# Timing: stage-specific measured speeds, remaining first case plus complete next case.
timing={}
for T in [1700,1850]:
 states={s['path'].split('/')[1]:s for s in meta['stages'] if s['path'].startswith(f'T{T}K/')};seconds=sum(states[k]['summary']['wall_seconds'] for k in ['equil','heat','cool']);prep_rate=seconds/40000
 # stage start from captured scheduler metadata; snapshots are ~seconds apart, adequate for ETA.
 status=json.loads((run/'status.json').read_text());progress=next(v for k,v in status['metadata'].items() if k.startswith('progress-') and v.get('temperature_K')==T)
 step=states['melt']['last_complete_step'];rate=(meta['unix']-progress['unix'])/step
 first_remaining=(50000-step)*rate;all_remaining=first_remaining+seconds+50000*rate
 timing[str(T)]={'last_step':step,'seconds_per_step_production':rate,'first_case_remaining_minutes':first_remaining/60,'all_remaining_hours':all_remaining/3600}
results['timing']=timing;results['snapshot_UTC']=datetime.datetime.fromtimestamp(meta['unix'],datetime.timezone.utc).isoformat();results['status']='partial: no temperature completed yet'
def clean(v):
 if isinstance(v,dict):return {k:clean(a) for k,a in v.items()}
 if isinstance(v,list):return [clean(a) for a in v]
 if isinstance(v,float) and not np.isfinite(v):return None
 return v
results=clean(results)
(out/'analysis.json').write_text(json.dumps(results,indent=2,allow_nan=False));print(json.dumps(results,indent=2))
eta=datetime.datetime.fromtimestamp(meta['unix']+max(v['all_remaining_hours'] for v in timing.values())*3600,datetime.timezone(datetime.timedelta(hours=5)))
report=f'''Предварительный анализ PBE 15k, задача 20960770.
Снимок данных: {results['snapshot_UTC']}. Ни одна температура ещё не завершена; подготовка 40 ps выполнена для 1700 и 1850 K.
1700 K: проанализированы 0–8 ps из плановых 50 ps. Доля qbar6>0.25 увеличилась с 0.621 до 0.979: жидкий слой исчезает, ячейка практически закристаллизовалась.
1850 K: проанализированы 0–8.5 ps. Доля qbar6>0.25 уменьшилась с 0.454 до 0: кристаллический слой исчезает. Направления изменений сохраняются для всех четырёх порогов 0.20/0.25/0.30/0.35 и дополнительного критерия согласованных связей.
Границы определены по сглаженному профилю qbar6 вдоль z/Lz; после исчезновения одной фазы координаты границ не определяются. Null в JSON означает отсутствие определимой границы, а не нулевую ширину.
Сравнение с 5k: при тех же 1700 K за 8 ps доля выросла с 0.496 до 0.622. У 15k рост быстрее, но начальный кристаллический слой уже толще (0.621 против 0.496); одна реализация не позволяет интерпретировать разницу скоростей как сдвиг Tm. У 5k нет 1850 K: показанная кривая 1800 K — качественный ориентир, не сравнение при одинаковой температуре.
Пока наблюдения совместимы с широкой рабочей скобкой 1700–1850 K. Это не окончательная оценка и не доверительный интервал. Для сравнения с прежними 1750–1775 K нужны следующие две температуры 1750 и 1800 K и полные траектории.
Фактическая скорость production: {timing['1700']['seconds_per_step_production']:.4f} и {timing['1850']['seconds_per_step_production']:.4f} s/шаг. Первые две температуры ожидаются примерно через 60–63 min от времени снимка, вся серия через 3.2–3.3 h. Расчётный центр ETA: {eta:%Y-%m-%d %H:%M} UTC+5; практический запас до 12:00 по Екатеринбургу при сохранении текущей скорости.
Расчёты не изменялись и не останавливались. Исходные данные снимка, SHA256 и полные кадры сохранены в этой папке; анализ выполнен прежним кодом audit/melting (14 соседей, Lechner–Dellago qbar6).
'''
(out/'REPORT_RU.txt').write_text(report,encoding='utf-8')
