"""Matched thermodynamic definitions and block uncertainties for archived/new MD."""
from pathlib import Path
import json,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'pbe_control_md';DEST=OUT/'analysis';DEST.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'audit/oct20/deps'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
read=lambda p:json.loads(Path(p).read_text())
cfg=read(OUT/'protocol.json');tasks=read(OUT/'tasks.json');raw={'5000':read(OUT/'baseline_5k.json'),'15000':{},'20000':{}}
missing=[]
for t in tasks:
    p=OUT/'results'/str(t['id']);state=cfg['states'][t['state']]
    if not (p/'summary.json').exists():missing.append(t['id']);continue
    meta=read(p/'summary.json');assert meta['status']=='completed'
    assert meta['model_sha256']==cfg['models'][str(t['checkpoint'])]['sha256']
    rows=[json.loads(s) for s in (p/'production-thermo.jsonl').read_text().splitlines()]
    assert rows[-1]['step']==cfg['production_steps'] and len(rows)==3001
    raw[str(t['checkpoint'])][state['composition']]={'x_Al':state['x_Al'],'rows':rows}
def stat(a):
    a=np.asarray(a);blocks={str(n):[float(x.mean()) for x in np.array_split(a,n)] for n in [5,10]}
    sem={n:float(np.std(b,ddof=1)/np.sqrt(len(b))) for n,b in blocks.items()}
    half=np.array_split(a,2)
    return dict(mean=float(a.mean()),sem=max(sem.values()),sem_by_nblocks=sem,block_means=blocks,
                second_minus_first_half=float(half[1].mean()-half[0].mean()),n=len(a))
fields=['Epot_eV_atom','Ekin_eV_atom','H_external_eV_atom','H_instantaneous_eV_atom','H_legacy_eV_atom','T_K','P_GPa','V_A3_atom','rho_g_cm3']
groups={};mix={};factor=96.4853321233
for step,states in raw.items():
    groups[step]={};mix[step]={}
    for name,d in states.items():
        rows=[r for r in d['rows'] if r['step']>0]
        groups[step][name]={'x_Al':d['x_Al'],**{f:stat([r[f] for r in rows]) for f in fields}}
    g=groups[step]
    if not {'Ti','Al'}.issubset(g):continue
    for name,d in g.items():
        x=d['x_Al'];mix[step][name]={'x_Al':x}
        for field in ['H_external_eV_atom','Epot_eV_atom','H_instantaneous_eV_atom','H_legacy_eV_atom']:
            if x in [0,1]:v=err=0.
            else:
                v=(d[field]['mean']-(1-x)*g['Ti'][field]['mean']-x*g['Al'][field]['mean'])*factor
                err=(d[field]['sem']**2+(1-x)**2*g['Ti'][field]['sem']**2+x*x*g['Al'][field]['sem']**2)**.5*factor
            mix[step][name][field]={'mean_kJ_mol':v,'sem_kJ_mol':err}
changes={}
for step in ['15000','20000']:
    changes[step]={}
    for name,d in groups[step].items():
        b=groups['5000'][name];dd={}
        for field in ['rho_g_cm3','Epot_eV_atom','H_external_eV_atom','T_K']:
            dd[field]={'delta':d[field]['mean']-b[field]['mean'],'sem_independent':(d[field]['sem']**2+b[field]['sem']**2)**.5}
        if name in mix[step]:
            f='H_external_eV_atom';a=mix[step][name][f];b=mix['5000'][name][f]
            dd['mixing_enthalpy']={'delta_kJ_mol':a['mean_kJ_mol']-b['mean_kJ_mol'],
              'sem_independent_kJ_mol':(a['sem_kJ_mol']**2+b['sem_kJ_mol']**2)**.5}
        changes[step][name]=dd
result={'status':'complete' if not missing else 'partial','missing_tasks':missing,'groups':groups,'mixing':mix,'changes_vs_5k':changes,
 'definition':'H=(Epot+Ekin+P_external V)/N. Positive compression pressure. Initial production frame excluded for all models.',
 'uncertainty':'Maximum SEM from 5/10 contiguous blocks. Propagation assumes independent runs; common pure endpoints correlate mixing values across compositions. No independent replicas yet.',
 'decision':'Pending complete MD, temperature/pressure and drift assessment. No automatic model selection.'}
(DEST/'comparison.json').write_text(json.dumps(result,indent=2))
fig,axs=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
for step in ['5000','15000','20000']:
    names=sorted(groups[step],key=lambda n:groups[step][n]['x_Al'])
    if not names:continue
    x=[groups[step][n]['x_Al'] for n in names]
    axs[0].errorbar(x,[groups[step][n]['rho_g_cm3']['mean'] for n in names],yerr=[groups[step][n]['rho_g_cm3']['sem'] for n in names],fmt='o-',capsize=3,label=str(int(step)//1000)+'k')
    for ax,field in zip(axs[1:],['H_external_eV_atom','Epot_eV_atom']):
        if not mix[step]:continue
        ax.errorbar(x,[mix[step][n][field]['mean_kJ_mol'] for n in names],yerr=[mix[step][n][field]['sem_kJ_mol'] for n in names],fmt='o-',capsize=3,label=str(int(step)//1000)+'k')
for ax,title,y in zip(axs,['Density','Mixing enthalpy: external pressure','Mixing potential energy'],['Density (g/cm³)','Mixing enthalpy (kJ/mol)','Mixing energy (kJ/mol)']):
    ax.set(title=title,xlabel='Al atomic fraction',ylabel=y);ax.grid(alpha=.2);ax.legend()
fig.suptitle('PBE checkpoint control MD, 2000 K, 1 bar; block SEM'+(' — INCOMPLETE' if missing else ''))
for ext in ['png','pdf']:fig.savefig(DEST/('comparison.'+ext),dpi=180)
print(json.dumps({'status':result['status'],'missing_tasks':missing,'baseline_5k':{n:{'rho':g['rho_g_cm3']['mean'],'Hmix':mix['5000'][n]['H_external_eV_atom'],'legacy_Hmix':mix['5000'][n]['H_legacy_eV_atom']} for n,g in groups['5000'].items()}},indent=2))
