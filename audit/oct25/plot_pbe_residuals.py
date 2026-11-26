from pathlib import Path
import sys, json, os
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'audit/oct20/deps'))
os.environ['MPLBACKEND']='Agg'
import matplotlib.pyplot as plt
data=json.loads((HERE/'pbe_energy_diagnostic/sensitivity.json').read_text())
rows=data['per_configuration_meV_atom']
fig,ax=plt.subplots(figsize=(7.5,4.4),layout='constrained')
for role,color,label in [('PBE','#276FBF','PBE fine-tuned'),('pretrained','#D97917','Pretrained')]:
    ax.scatter([r['x_Ti'] for r in rows],[r[role] for r in rows],s=42,
               edgecolors=color,facecolors=color if role=='PBE' else 'none',label=label,zorder=3)
for sid,offset in [('009',(-35,5)),('015',(-8,10)),('005',(-16,9)),('020',(5,5)),('003',(-28,-16)),('008',(6,-3)),('016',(-15,8))]:
    r=next(r for r in rows if r['id'].endswith(sid))
    ax.annotate(sid,(r['x_Ti'],r['PBE']),xytext=offset,textcoords='offset points',fontsize=9,color='#276FBF')
ax.axhline(0,color='.4',ls='--',lw=.8)
ax.set(xlabel=r'Ti atomic fraction $x_{\mathrm{Ti}}$',
       ylabel='Model − PBE DFT energy (meV/atom)',xlim=(-.04,1.05),ylim=(-7.5,20),
       title='Energy residuals on the frozen common20 liquid subset')
ax.spines[['top','right']].set_visible(False)
ax.legend(loc='upper left',frameon=False)
fig.savefig(HERE/'pbe_energy_diagnostic/residuals_vs_composition.png',dpi=200)
