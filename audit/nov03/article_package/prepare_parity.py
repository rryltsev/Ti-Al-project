"""Publication-size parity figure from the exact frozen 49-structure benchmark."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[3]
MON=ROOT/'testset/monitor'
MAN=ROOT/'manuscript'
OUT=Path(__file__).resolve().parent
sources={}
def read(p):
    sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())
m=read(MAN/'supporting_data/test49_metrics.json')
b=read(OUT/'test49_input_snapshot.json')
records={r['metadata']['structure_id']:r for r in b['records']}
assert all(s in records for s in m['common_ids'])
records=[records[s] for s in m['common_ids']]
families={'liquid':('o','Liquid pool'),'POSCAR_alpha-Ti-hcp_orthogonal':('s','hcp'),
    'POSCAR_beta-Ti-bcc':('^','bcc'),'POSCAR_gamma_AlTi_L1_0':('D',r'$\gamma$-TiAl')}
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'legend.fontsize':7,
    'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axs=plt.subplots(2,3,figsize=(7.2,5.1),layout='constrained')
checks={}
for col,xc in enumerate(['PBE','PBEsol','r2SCAN']):
    for cache,kind,color,label in [('pretrained','pretrained_metrics','#e88726','Pretrained'),
        ('PBE-20000' if xc=='PBE' else xc,'metrics','#2878B5','Fine-tuned')]:
        er=[];fr=[];sr=[]
        for r in records:
            sid=r['metadata']['structure_id'];pred=read(MON/'predictions'/cache/(sid+'.json'))
            expected=m['models'][xc]['sha256'] if kind=='metrics' else m['pretrained_model']['sha256']
            assert pred['model_sha256']==expected
            assert pred['poscar_sha256']==r['metadata']['poscar_sha256']
            assert pred['symbols']==r['symbols']
            ref=r['labels'][xc];n=len(r['symbols'])
            e0=ref['energy_eV']/n;e1=pred['energy_eV']/n
            f0=np.asarray(ref['forces_eV_A']);f1=np.asarray(pred['forces_eV_A'])
            s0=np.asarray(ref['stress_eV_A3'])[np.triu_indices(3)]*160.21766208
            s1=-np.asarray(pred['virial_eV'])[np.triu_indices(3)]/ref['volume_A3']*160.21766208
            er.append((e1-e0)*1000);fr.extend((f1-f0).ravel());sr.extend(s1-s0)
            marker=families[r['metadata']['family']][0]
            axs[0,col].scatter(e0,e1,s=21 if kind=='pretrained_metrics' else 12,
                facecolors='none' if kind=='pretrained_metrics' else color,edgecolors=color,marker=marker,linewidths=.65,zorder=3)
            axs[1,col].scatter(f0.ravel(),f1.ravel(),s=.7,c=color,alpha=.20,rasterized=True)
        checks[xc+'/'+kind]={}
        for key,error in [('energy_meV_atom',er),('force_eV_A',fr),('stress_GPa',sr)]:
            error=np.array(error);computed={'mae':float(np.mean(np.abs(error))),'rmse':float(np.sqrt(np.mean(error**2)))}
            for stat,value in computed.items(): assert np.isclose(value,m[kind][xc]['all'][key][stat],rtol=1e-10,atol=1e-10),(xc,kind,key,stat)
            checks[xc+'/'+kind][key]=computed
    for row in range(2):
        ax=axs[row,col];lo=min(ax.get_xlim()[0],ax.get_ylim()[0]);hi=max(ax.get_xlim()[1],ax.get_ylim()[1])
        ax.plot([lo,hi],[lo,hi],'--',c='.5',lw=.65,zorder=0)
        ax.set(xlim=(lo,hi),ylim=(lo,hi),aspect='equal')
        title={'PBE':'PBE','PBEsol':'PBEsol','r2SCAN':r'r$^2$SCAN'}[xc]
        ax.set_title(f'({chr(97+row*3+col)}) {title}')
        ax.set_xlabel('DFT energy (eV/atom)' if row==0 else 'DFT force (eV/Å)')
        if col==0: ax.set_ylabel('Model energy (eV/atom)' if row==0 else 'Model force (eV/Å)')
axs[0,0].legend(handles=[Line2D([],[],c='#2878B5',marker='o',ms=4,ls='',label='Fine-tuned'),
    Line2D([],[],c='#e88726',marker='o',mfc='none',ms=4,ls='',label='Pretrained')],loc='lower right',frameon=False,fontsize=6.8)
fig.legend(handles=[Line2D([],[],c='.4',marker=marker,ls='',ms=4,label=label) for marker,label in families.values()],
    loc='outside lower center',ncol=4,frameon=False)
fig.savefig(MAN/'figures/testset_parity.pdf',bbox_inches='tight')
fig.savefig(OUT/'test49_parity.png',dpi=200,bbox_inches='tight')
(MAN/'supporting_data/test49_parity_sources.json').write_text(json.dumps({'source_hashes':sources,'verified_metrics':checks},indent=2)+'\n')
print('All 36 MAE/RMSE values reproduce the frozen benchmark; figure saved.')
