"""Retain only the 2000 K composition curve in manuscript Figure 6."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8'))
thermo=read('audit/oct20/thermo_reprocessed.json')
refs=read('audit/oct20/reference_values.json')['mixing_digitized_kJmol']
stat=read('audit/oct25/pbe_full_enthalpy_gpu/analysis/statistics.json')
pbe=sorted(stat['windows']['production']['mixing']['15000'].values(),key=lambda r:r['x_Al'])
plt.rcParams.update({'font.size':8,'axes.labelsize':9,'axes.titlesize':9,'legend.fontsize':7,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(3.4,2.8),layout='constrained')
for model,color,label in [('pbe','#2878B5','PBE 15k'),('sol','#39835B','PBEsol'),('scan','#D55E00',r'r$^2$SCAN')]:
    if model=='pbe':
        x=[r['x_Al'] for r in pbe]; y=[r['H']['mean'] for r in pbe]; e=[r['H']['sem'] for r in pbe]
    else:
        g=sorted([r for r in thermo if r['T']==2000 and r['dft']==model],key=lambda r:r['x_Al'])
        x=[r['x_Al'] for r in g]; y=[r['dH_external_pressure'] for r in g];e=[r['dH_external_sem_5ps'] for r in g]
    ax.errorbar(x,y,yerr=e,fmt='o-',color=color,ms=3,lw=1,capsize=2,label=label)
ax.scatter([float(x) for x in refs],list(refs.values()),c='black',marker='s',s=14,label='Calorimetry (reproduced)',zorder=5)
ax.set(xlabel=r'$x_{\mathrm{Al}}$',ylabel=r'$\Delta H_{\mathrm{mix}}$ (kJ mol$^{-1}$)',xlim=(-.02,1.02),ylim=(-36,2),title='2000 K')
ax.legend(frameon=False,loc='upper center')
fig.savefig(ROOT/'manuscript/figures/mixing_enthalpy.pdf',bbox_inches='tight')
fig.savefig(OUT/'mixing_enthalpy.png',dpi=220,bbox_inches='tight')
plt.close(fig)
