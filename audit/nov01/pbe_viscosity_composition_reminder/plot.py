from pathlib import Path
import os,sys,json,hashlib
ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[2]
sys.path.insert(0,str(WORK/'audit/oct27/viscosity_5k_vs_15k/python_deps'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sources=[WORK/'audit/oct25/viscosity/viscosity_merged.json',WORK/'manuscript/supporting_data/pbe15k_viscosity_N15.json',WORK/'audit/nov01/pbe15k_legacy_pooling_v2/temperature_plot/plot_data.json']
five=[dict(x_Al=round(1-r['c_Ti'],6),eta_mPa_s=r['eta_mPa_s'],eta_std_mPa_s=r['eta_std_mPa_s'],n_replicas=r['n_replicas'],production_ps=20) for r in read(sources[0]) if r['T']==1873 and r['dft']=='pbe']
fifteen=[dict(x_Al=r['x_Al'],eta_mPa_s=r['windows']['40ps']['eta_mPa_s'],eta_std_mPa_s=r['windows']['40ps']['eta_std_mPa_s'],n_replicas=r['n_replicas'],production_ps=40) for r in read(sources[1])['points']]
ti=next(r for r in read(sources[2])['pbe15k'] if r['T']==1873)
fifteen.append(dict(x_Al=0,eta_mPa_s=ti['eta_mPa_s'],eta_std_mPa_s=ti['eta_std_mPa_s'],kind='exploratory_pooled_old_new',cutoff_THz=ti['cutoff_THz']))
five.sort(key=lambda r:r['x_Al']);fifteen.sort(key=lambda r:r['x_Al'])
assert len(five)==11 and len(fifteen)==4
comparison=[]
for r in fifteen:
 b=next(p for p in five if abs(p['x_Al']-r['x_Al'])<1e-8)
 comparison.append(dict(x_Al=r['x_Al'],eta5=b['eta_mPa_s'],se5=b['eta_std_mPa_s'],eta15=r['eta_mPa_s'],se15=r['eta_std_mPa_s'],delta_pct=100*(r['eta_mPa_s']/b['eta_mPa_s']-1)))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':12,'axes.titlesize':14,'pdf.fonttype':42,'ps.fonttype':42})
fig,ax=plt.subplots(figsize=(8.2,5.5))
fig.subplots_adjust(left=.11,right=.97,bottom=.23,top=.87)
blue='#2265A0';gray='#737B83'
ax.errorbar([r['x_Al'] for r in five],[r['eta_mPa_s'] for r in five],yerr=[r['eta_std_mPa_s'] for r in five],fmt='s--',color=gray,mfc='white',ms=6,lw=1.4,capsize=3.5,label='PBE 5k: 20 пс',zorder=3)
ax.plot([r['x_Al'] for r in fifteen],[r['eta_mPa_s'] for r in fifteen],color=blue,lw=1.8,zorder=2)
alloy=[r for r in fifteen if r['x_Al']>0]
ax.errorbar([r['x_Al'] for r in alloy],[r['eta_mPa_s'] for r in alloy],yerr=[r['eta_std_mPa_s'] for r in alloy],fmt='o',color=blue,ms=7,capsize=4,lw=1.5,label='PBE 15k: сплавы, 15 × 40 пс',zorder=5)
ax.errorbar([0],[ti['eta_mPa_s']],yerr=[ti['eta_std_mPa_s']],fmt='D',color=blue,ms=7,capsize=4,lw=1.5,label='PBE 15k: чистый Ti, объединение',zorder=6)
ax.set(title='Вязкость Ti–Al при T = 1873 K',xlabel=r'Атомная доля алюминия $x_{\mathrm{Al}}$',ylabel=r'Вязкость $\eta$, мПа·с',xlim=(-.025,1),ylim=(.45,5.65))
ax.set_xticks(np.arange(0,1.01,.1));ax.set_yticks(np.arange(1,5.51,1))
ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:.1f}'.replace('.',',')))
ax.grid(axis='y',alpha=.35,color='#CCD4DA',lw=.7)
ax.spines[['top','right']].set_visible(False)
ax.legend(loc='lower left',frameon=False,fontsize=9.5)
peak=max(five,key=lambda r:r['eta_mPa_s'])
ax.annotate('5k: 5,02',(peak['x_Al'],peak['eta_mPa_s']),xytext=(-13,19),textcoords='offset points',ha='right',color=gray,fontsize=10)
peak=max(fifteen,key=lambda r:r['eta_mPa_s'])
ax.annotate('15k: 4,73',(peak['x_Al'],peak['eta_mPa_s']),xytext=(14,9),textcoords='offset points',ha='left',color=blue,fontsize=10)
fig.text(.11,.10,'Точки сплавов 15k сохранены на N = 15. Чистый Ti: новая оценка 3,200 ± 0,084 мПа·с.',fontsize=8.8,color='#46515B')
fig.text(.11,.061,'Погрешности — стандартные ошибки; линии соединяют рассчитанные составы.',fontsize=9,color='#59636C')
fig.text(.11,.023,'Сетка составов у 15k разреженная: точное положение максимума по ней не определяется.',fontsize=8.8,color='#59636C')
for e in ['png','pdf','svg']:fig.savefig(ROOT/f'viscosity_composition_1873K_5k_15k.{e}',dpi=230)
plt.close(fig)
result=dict(temperature_K=1873,checkpoint5k=five,checkpoint15k=fifteen,comparison= comparison,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},note='Original composition series remains stopped. Saved N15 alloy results; pure-Ti endpoint uses the separately requested exploratory pooled estimate. No official article data modified.')
(ROOT/'plot_data.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'comparison':comparison,'files':[str(ROOT/f'viscosity_composition_1873K_5k_15k.{e}') for e in ['png','pdf']]},indent=2))
