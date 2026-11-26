"""Requested pure-Ti temperature plot; separate from the production monitor."""
from pathlib import Path
import sys, os, json, hashlib
ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[3]
sys.path.insert(0,str(WORK/'audit/oct27/viscosity_5k_vs_15k/python_deps'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
base=WORK/'audit/oct29/pbe15k_ti_temperature'
scan_path=ROOT.parent/'common_cutoff/scan.json'
archived_path=base/'before/viscosity_current_oct29.json'
latest_path=base/'collected/analysis/latest_Ti_2100K.json'
scan=read(scan_path)
pbe=[]
for T,requested in [(1873,3.200),(2000,2.830)]:
    r=min(scan[str(T)]['pooled20'],key=lambda r:abs(r['fcut']/1e12-1.25))
    assert abs(r['fcut']/1e12-1.25)<1e-10
    assert round(r['acint']*1000,3)==requested
    pbe.append(dict(T=T,eta_mPa_s=requested,eta_std_mPa_s=r['acint_std']*1000,
                    unrounded_eta_mPa_s=r['acint']*1000,cutoff_THz=1.25,
                    kind='pooled_old_new_20ps',old_replicas=25 if T==1873 else 20,
                    new_40ps_replicas=10,checkpoint_step=15000))
latest=read(latest_path)
r=latest['points'][0]['windows']['40ps']
pbe.append(dict(T=2100,eta_mPa_s=r['eta_mPa_s'],eta_std_mPa_s=r['eta_std_mPa_s'],
                kind='unchanged_new_full_40ps',n_replicas=latest['n_replicas'],checkpoint_step=15000))
assert latest['n_replicas']==15
refs=[r for r in read(archived_path) if r['dft']=='scan' and r['c_Ti']==1]
assert [r['T'] for r in pbe]==[1873,2000,2100]
assert len(refs)==4
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':12,
    'axes.titlesize':15,'legend.fontsize':9.5,'pdf.fonttype':42,'ps.fonttype':42})
fig,ax=plt.subplots(figsize=(8.2,5.65))
fig.subplots_adjust(left=.115,right=.97,top=.88,bottom=.23)
blue='#2265A0';orange='#C96B2C'
T=np.linspace(1850,2120,200)
eta=.61*np.exp(27000/(8.31*T))
ax.fill_between(T,eta*.897,eta*1.103,color='#B8C0C7',alpha=.23,zorder=0)
refline,=ax.plot(T,eta,color='#51585E',ls='--',lw=1.3,label='Ishikawa: корреляция*',zorder=1)
refs.sort(key=lambda r:r['T'])
scanline=ax.errorbar([r['T'] for r in refs],[r['eta_mPa_s'] for r in refs],
    yerr=[r['eta_std_mPa_s'] for r in refs],fmt='o-',color=orange,lw=1.4,ms=6,
    capsize=4,label=r'r$^2$SCAN',zorder=3)
ax.plot([r['T'] for r in pbe],[r['eta_mPa_s'] for r in pbe],color=blue,lw=1.6,zorder=2)
pooled=ax.errorbar([r['T'] for r in pbe[:2]],[r['eta_mPa_s'] for r in pbe[:2]],
    yerr=[r['eta_std_mPa_s'] for r in pbe[:2]],fmt='D',color=blue,ms=7.5,
    capsize=4,elinewidth=1.4,label='PBE 15k: объединённые оценки',zorder=5)
r=pbe[2]
retained=ax.errorbar([r['T']],[r['eta_mPa_s']],yerr=[r['eta_std_mPa_s']],
    fmt='o',color=blue,mfc='white',mew=1.6,ms=7.5,capsize=4,elinewidth=1.4,
    label='PBE 15k: 40 пс, N = 15',zorder=5)
for r in pbe:
    ax.annotate(f"{r['eta_mPa_s']:.3f}".replace('.',','),(r['T'],r['eta_mPa_s']),
                xytext=(0,-26),textcoords='offset points',ha='center',va='top',
                color=blue,fontsize=10,fontweight='bold')
ax.set(title='Вязкость чистого титана',xlabel='Температура, K',
       ylabel=r'Вязкость $\eta$, мПа·с',xlim=(1840,2130),ylim=(2.18,4.4))
ax.set_xticks([1873,1941,2000,2100])
ax.set_yticks(np.arange(2.25,4.26,.25))
from matplotlib.ticker import FuncFormatter
ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:.2f}'.replace('.',',')))
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y',color='#D8DEE3',lw=.65,alpha=.7)
ax.set_axisbelow(True)
ax.legend(handles=[pooled,retained,scanline,refline],loc='upper right',frameon=False,
          handlelength=2.4,labelspacing=.55)
fig.text(.115,.095,'1873 и 2000 K: объединение старых и новых траекторий; cutoff = 1,25 ТГц.',
         fontsize=9,color='#414950')
fig.text(.115,.058,'Погрешности — формальные стандартные ошибки. Объединённые оценки предварительные.',
         fontsize=8.5,color='#59636C')
fig.text(.115,.021,'* Ранее использованная корреляция Ishikawa; полоса ±10,3%. Линии соединяют точки.',
         fontsize=8.5,color='#59636C')
for ext in ['png','pdf','svg']:
    fig.savefig(ROOT/f'ti_viscosity_temperature_pbe15k.{ext}',dpi=240)
plt.close(fig)
data=dict(description='User-requested temperature plot with pooled 1873/2000 K estimates, no 5k or superseded historical PBE points.',
    pbe15k=pbe,r2scan=refs,formal_errors=True,pooled_results_exploratory=True,
    production_protocol_changed=False,source_sha256={str(p):sha(p) for p in [scan_path,archived_path,latest_path]},
    ishikawa=dict(prefactor_mPa_s=.61,activation_J_mol=27000,R_J_mol_K=8.31,relative_band=.103,
        provenance='Retained unchanged from the existing article plotting script; provisional transcription.'))
(ROOT/'plot_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'files':[str(ROOT/f'ti_viscosity_temperature_pbe15k.{e}') for e in ['png','pdf','svg']],
                  'pbe15k':pbe},ensure_ascii=True,indent=2))
