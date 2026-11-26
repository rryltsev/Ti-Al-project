"""Merge all EOS/liquid screens; changes are relative to each article checkpoint."""
from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'all_xc_audit';DEST=OUT/'analysis';DEST.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'audit/oct20/deps'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ase.units import GPa
read=lambda p:json.loads(Path(p).read_text())
cfg=read(OUT/'protocol.json');tasks=read(OUT/'tasks.json');paths={};eos={};pred={}
for key,path in [('PBE-5000',HERE/'pbe_property_screen/5000'),('PBE-15000',HERE/'pbe_property_screen/retry15k/15000')]:
    meta=read(path/'summary.json');assert meta['model_sha256']==cfg['models'][key]['sha256']
    paths[key]=path;eos[key]=read(path/'eos.json');pred[key]=np.load(path/'predictions.npz')
for t in tasks:
    if t['kind']!='properties':continue
    path=OUT/'results'/str(t['id'])
    if not (path/'summary.json').exists():continue
    meta=read(path/'summary.json');assert meta['status']=='passed' and meta['model_sha256']==cfg['models'][t['key']]['sha256']
    paths[t['key']]=path;eos[t['key']]=read(path/'eos.json');pred[t['key']]=np.load(path/'liquid.npz')
def stat(a):
    a=np.array(a);assert len(a)==50
    se={str(n):float(a.reshape(n,-1).mean(axis=1).std(ddof=1)/np.sqrt(n)) for n in [5,10]}
    return {'mean':float(a.mean()),'sem':max(se.values()),'sem_by_blocks':se,'first_half':float(a[:25].mean()),'second_half':float(a[25:].mean())}
result={'keys':list(eos),'missing':[k for k in cfg['models'] if k not in eos],'crystals':eos,'liquids':{},'archive_checks':{}}
for xc,ds in cfg['liquids'].items():
    baseline=next(k for k,m in cfg['models'].items() if m['functional']==xc and m['article_checkpoint'])
    if baseline not in pred:continue
    rows=read(OUT/ds['rows_file']);archive=np.load(OUT/(xc+'-archive.npz'));p0=pred[baseline]
    check=[]
    for x in [0,.1,.3,.5,1]:
        rr=[r for r in rows if r['x_Al']==x];ee=[];ff=[];pp=[]
        for r in rr:
            i=r['i'];n=r['n_atoms'];V=n*r['V_A3_atom']
            ee.append(1000*(float(p0[f'energy_{i}'])-float(archive[f'energy_{i}']))/n)
            ff.extend((p0[f'forces_{i}']-archive[f'forces_{i}']).ravel())
            st=archive[f'stress_{i}'];pold=-np.mean(st[:3])/GPa if st.shape==(6,) else -np.trace(st)/(3*GPa)
            pp.append(np.trace(p0[f'virial_{i}'])/(3*V*GPa)-pold)
        check.append({'x_Al':x,'max_E_difference_meV_atom':float(max(abs(np.array(ee)))),
                      'F_difference_RMS_eV_A':float(np.sqrt(np.mean(np.array(ff)**2))),
                      'max_P_difference_GPa':float(max(abs(np.array(pp))))})
    result['archive_checks'][xc]={'baseline':baseline,'by_composition':check}
    for key in pred:
        if cfg['models'][key]['functional']!=xc:continue
        pp=pred[key];rrr=[]
        for x in [0,.1,.3,.5,1]:
            rr=[r for r in rows if r['x_Al']==x];ee=[];pr=[];df=[];fbase=[]
            for r in rr:
                i=r['i'];n=r['n_atoms'];V=n*r['V_A3_atom']
                ee.append(1000*(float(pp[f'energy_{i}'])-float(p0[f'energy_{i}']))/n)
                pr.append(float(np.trace(pp[f'virial_{i}']-p0[f'virial_{i}'])/(3*V*GPa)))
                df.extend((pp[f'forces_{i}']-p0[f'forces_{i}']).ravel());fbase.extend(p0[f'forces_{i}'].ravel())
            rms=float(np.sqrt(np.mean(np.array(df)**2)));rel=rms/float(np.sqrt(np.mean(np.array(fbase)**2)))
            rrr.append({'x_Al':x,'delta_E_meV_atom':stat(ee),'delta_P_GPa':stat(pr),'delta_F_RMS_eV_A':rms,'relative_delta_F_RMS':rel})
        for r in rrr:
            x=r['x_Al'];a=r['delta_E_meV_atom'];ti=rrr[0]['delta_E_meV_atom'];al=rrr[-1]['delta_E_meV_atom']
            val=a['mean']-(1-x)*ti['mean']-x*al['mean'];sem=np.sqrt(a['sem']**2+(1-x)**2*ti['sem']**2+x*x*al['sem']**2)
            r['mixing_energy_change_kJ_mol']={'mean':float(val*.09648533212) if 0<x<1 else 0.,'sem':float(sem*.09648533212) if 0<x<1 else 0.}
        result['liquids'][key]={'baseline':baseline,'rows':rrr}
result['limitations']='Paired predictions on each functional article-model ensemble, not new equilibrium MD. Blocks from one trajectory per state. Comparing functional ensembles is not a common-geometry cross-functional energy comparison.'
(DEST/'properties.json').write_text(json.dumps(result,indent=2))
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
colors=['#2874a6','#20916f','#8e54ad','#d38c20','#bd4545'];xcs=['PBE','PBEsol','r2SCAN']
fig,axs=plt.subplots(3,3,figsize=(12,9),layout='constrained')
for j,xc in enumerate(xcs):
    keys=[k for k,m in cfg['models'].items() if m['functional']==xc]
    for i,phase in enumerate(['Ti_hcp','Ti_bcc','Al_fcc']):
        for color,key in zip(colors,keys):
            if key not in eos:continue
            ref='Ti_hcp' if phase.startswith('Ti') else phase;e0=eos[key][ref]['equilibrium']['E_eV_atom'];curve=eos[key][phase]['curve']
            axs[i,j].plot([r['V_A3_atom'] for r in curve],[1000*(r['E_eV_atom']-e0) for r in curve],'-',lw=1.2,color=color,label=f"{cfg['models'][key]['step']/1000:g}k")
        axs[i,j].set(xlabel='Volume (Å³/atom)',ylabel='E − phase reference (meV/atom)',title=xc+' / '+phase);axs[i,j].grid(alpha=.2)
        if i==0:axs[i,j].legend(fontsize=8)
fig.suptitle('EOS: Ti energies relative to each checkpoint hcp minimum; Al relative to fcc minimum',fontsize=11)
fig.savefig(DEST/'eos_all_checkpoints.png',dpi=180);fig.savefig(DEST/'eos_all_checkpoints.pdf');plt.close(fig)
fig,axs=plt.subplots(3,3,figsize=(12,9),layout='constrained')
for j,xc in enumerate(xcs):
    keys=[k for k,m in cfg['models'].items() if m['functional']==xc]
    for color,key in zip(colors,keys):
        if key not in result['liquids']:continue
        rr=result['liquids'][key]['rows'];xx=[r['x_Al'] for r in rr];lab=f"{cfg['models'][key]['step']/1000:g}k"
        for i,field in enumerate(['mixing_energy_change_kJ_mol','delta_P_GPa']):
            axs[i,j].errorbar(xx,[r[field]['mean'] for r in rr],yerr=[r[field]['sem'] for r in rr],fmt='o-',ms=3,lw=1,capsize=2,c=color,label=lab)
        axs[2,j].plot(xx,[100*r['relative_delta_F_RMS'] for r in rr],'o-',ms=3,lw=1,c=color,label=lab)
    for i,label in enumerate(['Mixing-energy change (kJ/mol)','Pressure change (GPa)','Force difference / baseline RMS (%)']):
        axs[i,j].set(xlabel=r'$x_{Al}$',ylabel=label,title=xc if i==0 else None);axs[i,j].grid(alpha=.2);axs[i,j].axhline(0,lw=.5,c='gray')
        if i==0:axs[i,j].legend(fontsize=8)
fig.suptitle('Changes relative to article checkpoints: PBE 5k; PBEsol 23.1k; r²SCAN 23.2k\nOwn-functional 2000 K liquid ensembles; energy/pressure bars are block SEM',fontsize=11)
fig.savefig(DEST/'liquid_checkpoint_sensitivity.png',dpi=180);fig.savefig(DEST/'liquid_checkpoint_sensitivity.pdf');plt.close(fig)
print(json.dumps({'completed':len(eos),'missing':result['missing'],'archive_checks':result['archive_checks']},indent=2))
