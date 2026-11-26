"""Audit known/completed PBE15k densities; incrementally update manuscript Fig. 3."""
from pathlib import Path
import csv,hashlib,json,os,re,shutil,sys
sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).parent
RUN=HERE/'pbe15k_ti_density_temperature';OUT=RUN/'analysis';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'audit/oct20/deps'));os.environ['MPLBACKEND']='Agg'
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as student
from ase.io.trajectory import Trajectory

SOURCES={}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def track(p):
    p=Path(p);SOURCES[str(p)]=sha(p);return p
def load(p):return json.loads(track(p).read_text(encoding='utf-8'))
def save(p,d):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8');tmp.replace(p)
cfg=load(RUN/'protocol.json');modelsha=cfg['models']['15000']['sha256']

def stats(a,dt=.01,blocks=(3,5,6)):
    a=np.asarray(a);assert np.isfinite(a).all();checks={}
    for ps in blocks:
        n=round(ps/dt)
        if len(a)%n==0 and len(a)//n>=4:
            b=a.reshape(-1,n).mean(axis=1);checks[str(ps)]=dict(sem=float(b.std(ddof=1)/np.sqrt(len(b))),blocks=len(b),means=b.tolist())
    key=max(checks,key=lambda k:checks[k]['sem']);sem=checks[key]['sem'];df=checks[key]['blocks']-1
    mean=float(a.mean());half=float(student.ppf(.975,df)*sem)
    return dict(mean=mean,sem=sem,ci95=[mean-half,mean+half],selected_block_ps=float(key),block_checks=checks,samples=len(a))

def primary(folder,state,source_id,protocol):
    meta=load(folder/'summary.json');assert meta['status']=='completed' and meta['model_sha256']==modelsha
    assert meta['protocol_sha256']==sha(protocol)
    raw=track(folder/'production-thermo.jsonl')
    rows=[json.loads(x) for x in raw.read_text().splitlines()]
    assert [r['step'] for r in rows]==list(range(0,30001,10))
    r=rows[1:];rho=np.array([d['rho_g_cm3'] for d in r]);T=np.array([d['T_K'] for d in r]);N=state['N']
    with Trajectory(str(track(folder/'production.traj'))) as tr:
        assert len(tr)==301 and all(len(a)==N and np.all(a.numbers==22) for a in tr)
        calc_rho=np.array([a.get_masses().sum()/a.get_volume()*1.66053906660 for a in tr])
        np.testing.assert_allclose(calc_rho,[d['rho_g_cm3'] for d in rows[::10]],rtol=0,atol=1e-10)
        frac=np.array([a.get_scaled_positions(wrap=True) for a in tr]);cells=np.array([a.cell.array for a in tr])
    delta=np.diff(frac,axis=0);delta-=np.round(delta)
    cart=np.einsum('tni,tij->tnj',delta,.5*(cells[1:]+cells[:-1]));cart-=cart.mean(axis=1,keepdims=True)
    xyz=np.concatenate([np.zeros_like(cart[:1]),np.cumsum(cart,axis=0)])
    msd10=float(np.mean(np.sum((xyz[100:]-xyz[:-100])**2,axis=-1)))
    q=dict(T_ASE_K=float(T.mean()),T_3N_minus_3_K=float(T.mean()*N/(N-1)),
           P_GPa=stats([d['P_GPa'] for d in r]),MSD_10ps_A2=msd10,
           rho_second_half_minus_first=float(rho[1500:].mean()-rho[:1500].mean()),
           rho_last10_minus_full=float(rho[-1000:].mean()-rho.mean()),
           rho_100fs_minus_10fs=float(rho[9::10].mean()-rho.mean()))
    q['warnings']=[]
    if abs(q['T_3N_minus_3_K']/state['T_K']-1)>.02:q['warnings'].append('Temperature differs from nominal by >2%')
    if not q['P_GPa']['ci95'][0]<=cfg['P_bar']*.0001<=q['P_GPa']['ci95'][1]:
        q['warnings'].append('Mean pressure differs from target outside block 95% interval; density SEM excludes protocol bias')
    if abs(q['rho_second_half_minus_first'])>.002*rho.mean():q['warnings'].append('Density half-window drift exceeds 0.2%; inspect equilibration')
    if state['phase']!='liquid' and msd10>5:q['warnings'].append('Substantial diffusion from crystalline initialization; inspect melting/phase change')
    return dict(id=source_id,checkpoint=15000,model_sha256=modelsha,x_Al=0.,composition='Ti',T_K=state['T_K'],
        phase_initial=state['phase'],kind='homogeneous_NPT',N=N,P_bar=1.,production_ps=30.,
        rho=stats(rho),qc=q,source=str(folder),job_id=meta['job_id'])

registry=[];pure=[]
# Consolidate both complete composition series, not only their pure-Ti endpoints.
for temp,base,summary in [
    (1873,ROOT/'audit/oct26/pbe15k_density_1873',ROOT/'audit/oct26/pbe15k_density_1873/analysis/statistics.json'),
    (2000,ROOT/'audit/oct25/pbe_full_enthalpy_gpu',ROOT/'audit/oct25/pbe_full_enthalpy_gpu/analysis/statistics.json')]:
    data=load(summary);group=data['results']['15000'] if temp==1873 else data['windows']['production']['groups']['15000']
    oldcfg=load(base/'protocol.json');assert oldcfg['models']['15000']['sha256']==modelsha
    rawbase=base/'collected' if temp==1873 else base
    p=primary(rawbase/'15000/results/0',dict(N=200,T_K=temp,phase='liquid'),f'prior-liquid-{temp}',base/'protocol.json')
    assert abs(p['rho']['mean']-group['Ti']['rho']['mean'])<1e-11
    pure.append(p)
    for name,d in group.items():
        if name=='Ti':registry.append(p);continue
        registry.append(dict(id=f'prior-{name}-{temp}',checkpoint=15000,model_sha256=modelsha,
            x_Al=d['x_Al'],composition=name,T_K=temp,phase_initial='liquid',kind='homogeneous_NPT',N=200,
            P_bar=1.,production_ps=30.,rho=d['rho'],source=str(summary)))

# Append complete temperature points; do not read active/truncated trajectories.
completed=[]
for s in cfg['states']:
    folder=RUN/'collected/results'/s['id']
    if (folder/'summary.json').exists():
        pre=load(RUN/'collected/preflight.json');assert pre['status']=='passed' and pre['model_sha256']==modelsha and 'A100' in pre['device']['name']
        p=primary(folder,s,'new-'+s['id'],RUN/'protocol.json');pure.append(p);registry.append(p);completed.append(s['id'])

# Coexistence-derived late single-phase data are ancillary, not pooled with homogeneous NPT.
ancillary=[]
for temp in [1700,1750,1775,1800,1850]:
    base=ROOT/('audit/oct26/pbe15k_melting_1775' if temp==1775 else 'audit/oct26/pbe15k_melting/attempt2')
    pcfg=load(base/'protocol.json');assert pcfg['checkpoint']['sha256']==modelsha
    path=track(base/f'collected/T{temp}K/melt/log.lammps');lines=path.read_text().splitlines()
    header=None;rows=[]
    for line in lines:
        fields=line.split()
        if fields[:2]==['Step','Time']:header=fields;continue
        if header and len(fields)==len(header):
            try:rows.append(dict(zip(header,map(float,fields))))
            except ValueError:pass
    assert rows and rows[-1]['Time'] in [50,100] and all(r['Atoms']==1296 for r in rows)
    end=rows[-1]['Time'];late=[r for r in rows if end-10<r['Time']<=end]
    assert len(late)==100
    dat=track(base/f'collected/T{temp}K/melt/final.data').read_text()
    mass=float(dat.split('Masses',1)[1].split('Atoms',1)[0].split()[1])
    assert abs(mass-47.867)<1e-6
    summary=load(base/'analysis/summary.json')['15k' if temp==1775 else f'15k_{temp}']
    f=summary['mean_last10_fraction'];assert f>.95 if temp<1775 else f<.05
    p=dict(id=f'coexistence-late-{temp}',checkpoint=15000,model_sha256=modelsha,composition='Ti',x_Al=0.,T_K=temp,
        kind='coexistence_late_single_phase',phase_initial='solid-liquid',phase_late='solid-like' if temp<1775 else 'liquid-like',
        N=1296,P_bar=0.,production_ps=10.,window_ps=[end-10,end],solid_fraction_last10=f,
        rho=stats([r['Atoms']*mass*1.66053906660/r['Volume'] for r in late],.1,(1,2)),
        T_mean_K=float(np.mean([r['Temp'] for r in late])),P_mean_GPa=float(np.mean([r['Press'] for r in late]))/10000,
        source=str(path),protocol='LAMMPS aniso NPT; Tdamp=0.1 ps, Pdamp=1 ps; last 10 ps after interface disappearance')
    registry.append(p);ancillary.append(p)
eospath=ROOT/'audit/oct25/pbe_property_screen/retry15k/15000/eos.json';eos=load(eospath)
eosmeta=load(eospath.parent/'summary.json')
assert eosmeta['status']=='passed' and eosmeta['model_sha256']==modelsha and eosmeta['train_infos']['step']==14999
for phase,key in [('alpha','Ti_hcp'),('beta','Ti_bcc')]:
    v=eos[key]['equilibrium']['V_A3_atom']
    registry.append(dict(id='static-'+phase,checkpoint=15000,model_sha256=modelsha,composition='Ti',x_Al=0.,T_K=0,
        kind='static_equilibrium',phase_initial=phase,rho={'mean':47.867*1.66053906660/v},V_A3_atom=v,source=str(eospath),
        note='Static constrained lattice; not a finite-temperature MD point or proof of phase stability.'))

notes=[
 'Density is mean instantaneous mass/volume. Primary NPT averages exclude t=0 and use 3000 samples at 10 fs.',
 'Primary SEM: maximum among 3, 5, 6 ps block estimates; not a total uncertainty or checkpoint-only error.',
 'Old 5k temperature-series values are preserved as published in the original CSV. Recording cadence differs from 15k.',
 'Phase marker denotes starting structure, not a structural identification of the final phase. MSD is a mobility diagnostic.',
 'Coexistence late windows use different cell size, pressure and thermostat/barostat; retained separately, not joined to main curves.',
 'Static EOS and phonon-relaxation volume duplicate the same 0 K equilibrium; no duplicate finite-temperature point is created.',
 'Property-screen single-point evaluations on 5k geometries, canceled equilibration-only MD, and NVT viscosity runs do not supply independent equilibrated 15k NPT density points.',
 'The 2000 K raw scalar mean is 4.2605731066 g/cm3; an earlier rounded status message used 4.26061. Article precision 4.2606 is unchanged.'
]
save(OUT/'density_registry_15k.json',dict(model_sha256=modelsha,points=registry,notes=notes,completed_new=completed))
with (OUT/'density_registry_15k.csv').open('w',newline='',encoding='utf-8-sig') as fp:
    fields=['id','composition','x_Al','T_K','kind','phase_initial','phase_late','N','P_bar','production_ps','rho_g_cm3','sem_g_cm3','source']
    w=csv.DictWriter(fp,fieldnames=fields);w.writeheader()
    for p in registry:w.writerow({k:(p['rho']['mean'] if k=='rho_g_cm3' else p['rho'].get('sem') if k=='sem_g_cm3' else p.get(k)) for k in fields})

PROJECT=Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project')
styles={'pbe':('#2878B5','PBE 5k'),'sol':('#39835B','PBEsol'),'scan':('#D55E00',r'r$^2$SCAN')}
old={}
for model in styles:
    with track(PROJECT/f'MD-results/density_DPA4-neo/density_scan_{model}.csv').open() as fp:
        old[model]=[{'T':float(r['T_target_K']),'rho':float(r['density_g_cm3'])} for r in csv.DictReader(fp,delimiter=';')]
assert len(old['pbe'])==16 and any(r['T']==2000 for r in old['pbe'])
extended_xc=None
xc_path=ROOT/'manuscript/supporting_data/xc_liquid_density_extended.json'
if xc_path.exists():
    extended_xc=load(xc_path)
    assert extended_xc['completed_points']==14
    for key in ['sol','scan']:
        pts=[p for p in extended_xc['points'] if p['key']==key]
        assert sorted(p['T_K'] for p in pts)==[1873,2000,2100,2200,2300,2400,2500]
        assert all(p['production_ps']==90 and p['qc']['MSD_10ps_A2']>=5 and p['qc']['solid_like_fraction_last10']<=.2 for p in pts)
drefs=load(ROOT/'audit/density_reference/references.json');lref=drefs['liquid']
experimental=load(ROOT/'audit/density_reference/reiplinger2022_points.json')['points']
save(OUT/'source_manifest.json',SOURCES)
fingerprint=hashlib.sha256(json.dumps({'registry':registry,'old':old,'extended_xc':extended_xc,'refs':drefs,'points':experimental,'script_sha':sha(__file__)},sort_keys=True).encode()).hexdigest()
previous=json.loads((OUT/'figure_state.json').read_text()) if (OUT/'figure_state.json').exists() else {}
FIG=ROOT/'manuscript/figures';DATA=ROOT/'manuscript/supporting_data'
if previous.get('fingerprint')==fingerprint and all((FIG/n).exists() and sha(FIG/n)==v for n,v in previous.get('canonical_hashes',{}).items()):
    print('No new completed points; figure unchanged.',len(completed),'of',len(cfg['states']));sys.exit(0)
backup=RUN/'before-update';backup.mkdir(exist_ok=True)
for src in [FIG/'density_ti.pdf',FIG/'density_ti.png',ROOT/'manuscript/sections/results.tex']:
    if src.exists() and not (backup/src.name).exists():shutil.copy2(src,backup/src.name)
plt.rcParams.update({'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':7,
    'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
phases=[('alpha',300,1100,'o'),('beta',1200,1900,'^'),('liquid',2000,2500,'s')]

def main_curves(axs,models):
    for model in models:
        color,label=styles[model]
        for phase,lo,hi,marker in phases:
            ax=axs[1] if phase=='liquid' else axs[0];g=[r for r in old[model] if lo<=r['T']<=hi]
            if phase=='liquid' and model in ['sol','scan'] and extended_xc:
                pts=sorted([p for p in extended_xc['points'] if p['key']==model],key=lambda p:p['T_K'])
                ax.errorbar([p['T_K'] for p in pts],[p['rho']['mean'] for p in pts],
                    yerr=[p['rho']['sem'] for p in pts],color=color,marker=marker,ms=3.4,lw=1,
                    capsize=2,label=label)
                continue
            ax.plot([r['T'] for r in g],[r['rho'] for r in g],c=color,marker=marker,ms=3.4,lw=1,
                ls='--' if model=='pbe' else '-',mfc='white' if model=='pbe' else color,
                label=label if phase in ['alpha','liquid'] else None)
    for phase,lo,hi,marker in phases:
        g=sorted([p for p in pure if p['phase_initial']==phase],key=lambda p:p['T_K'])
        if not g:continue
        ax=axs[1] if phase=='liquid' else axs[0]
        # A gap means pending calculations, so only connect adjacent temperatures in the target grid.
        grid=sorted([p['T_K'] for p in cfg['states'] if p['phase']==phase]+([1873,2000] if phase=='liquid' else []))
        for a,b in zip(g[:-1],g[1:]):
            if grid.index(b['T_K'])==grid.index(a['T_K'])+1:ax.plot([a['T_K'],b['T_K']],[a['rho']['mean'],b['rho']['mean']],color=styles['pbe'][0],lw=1.3)
        ax.errorbar([p['T_K'] for p in g],[p['rho']['mean'] for p in g],yerr=[p['rho']['sem'] for p in g],
            color=styles['pbe'][0],fmt=marker,ms=4.4,capsize=2,lw=.9,zorder=5,
            label='PBE 15k' if phase in ['alpha','liquid'] or not any(p['phase_initial']=='alpha' for p in pure) else None)

def writefig(fig,name):
    for ext in ['pdf','png']:fig.savefig(OUT/(name+'.'+ext),bbox_inches='tight',dpi=240)
    plt.close(fig)

fig,axs=plt.subplots(1,2,figsize=(7.4,3.5),layout='constrained')
main_curves(axs,list(styles))
for phase in ['alpha','beta']:
    a=np.array(drefs['solid'][phase]);axs[0].plot(a[:,0],a[:,1],c='black',marker='D',ms=3,ls=':',lw=.9,label="Zinov'ev (1989)" if phase=='alpha' else None)
tt=np.linspace(*lref['plotted_range_K'],200);rr=(lref['c1_kg_m3']+lref['c2_kg_m3_K']*(tt-lref['Tm_K']))/1000;u=lref['expanded_relative_uncertainty']
axs[1].fill_between(tt,rr*(1-u),rr*(1+u),color='.88',zorder=0,label=r'$\pm2\%$ (95% coverage)')
axs[1].plot(tt,rr,c='black',lw=1.1,label='Ntonti et al. (2024)')
axs[1].errorbar([p['T_K'] for p in experimental],[p['rho_g_cm3'] for p in experimental],
    yerr=[[p['rho_g_cm3']-p['lower_g_cm3'] for p in experimental],[p['upper_g_cm3']-p['rho_g_cm3'] for p in experimental]],
    fmt='o',ms=2.6,mfc='white',mec='black',mew=.6,ecolor='.35',elinewidth=.55,capsize=1.2,zorder=4,label='Reiplinger & Brillo (2022)')
for T,label in [(1155,r'$\alpha$-$\beta$: 1155 K'),(1943,r'$T_m$: 1943 K')]:
    axs[0].axvline(T,color='.45',ls='-.',lw=.7,zorder=0);axs[0].text(T-25,4.82,label,rotation=90,ha='right',va='top',fontsize=7,color='.3')
axs[0].set(title=r'(a) $\alpha$- and $\beta$-Ti starting cells',xlim=(220,2010),ylim=(4.20,4.85))
axs[1].set(title='(b) Separately prepared liquid',xlim=(1835,2530),ylim=(3.87,4.85))
axs[0].legend(frameon=True,facecolor='white',edgecolor='none',framealpha=1,ncol=2,fontsize=7,loc='lower left');axs[1].legend(frameon=False,fontsize=6.7,loc='upper right',labelspacing=.28,handlelength=1.5)
for ax in axs:ax.set(xlabel='Temperature (K)',ylabel=r'Density (g cm$^{-3}$)')
writefig(fig,'density_ti')

# One-axis PBE checkpoint comparison, including ancillary coexistence estimates.
fig,ax=plt.subplots(figsize=(7.1,4.7),layout='constrained');main_curves([ax,ax],['pbe'])
for phase,marker,label in [('solid-like','D','15k: late coexistence, solid-like'),('liquid-like','d','15k: late coexistence, liquid-like')]:
    g=[p for p in ancillary if p['phase_late']==phase]
    ax.errorbar([p['T_K'] for p in g],[p['rho']['mean'] for p in g],yerr=[p['rho']['sem'] for p in g],fmt=marker,color='#8663A8',ms=4,capsize=2,label=label)
handles,labels=ax.get_legend_handles_labels();unique=dict(zip(labels,handles));ax.legend(unique.values(),unique.keys(),frameon=False,loc='upper right',fontsize=8)
ax.set(xlabel='Temperature (K)',ylabel=r'Density (g cm$^{-3}$)',title='Pure Ti: PBE checkpoint comparison',xlim=(230,2560))
ax.text(.02,.03,'Starting cells: circles = alpha; triangles = beta; squares = liquid\nCoexistence estimates use a different protocol and are not joined to the curves.',transform=ax.transAxes,fontsize=7,color='.3')
writefig(fig,'pbe_5k_15k_temperature_comparison')

# Preserve a version for each new set of completed temperatures.
version=OUT/'versions'/fingerprint[:12];version.mkdir(parents=True,exist_ok=True)
for name in ['density_ti.pdf','density_ti.png','pbe_5k_15k_temperature_comparison.pdf','pbe_5k_15k_temperature_comparison.png','density_registry_15k.json','density_registry_15k.csv']:
    shutil.copy2(OUT/name,version/name)
canonical_hashes={};locked=[]
for ext in ['pdf','png']:
    src=OUT/('density_ti.'+ext);tmp=FIG/('density_ti.'+ext+'.tmp');shutil.copyfile(src,tmp)
    try:
        tmp.replace(FIG/('density_ti.'+ext));canonical_hashes['density_ti.'+ext]=sha(src)
    except PermissionError:
        # Windows PDF viewers can deny both writes and deletion. Keep that open file intact.
        locked.append('density_ti.'+ext)
article_name='density_ti_'+fingerprint[:12]+'.pdf'
article_path=FIG/article_name
if not article_path.exists():shutil.copyfile(OUT/'density_ti.pdf',article_path)
assert sha(article_path)==sha(OUT/'density_ti.pdf')
canonical_hashes[article_name]=sha(article_path)
results_path=ROOT/'manuscript/sections/results.tex';results_text=results_path.read_text(encoding='utf-8')
pattern=r'(\\includegraphics\[width=\\textwidth\]\{figures/)density_ti(?:_[0-9a-f]{12})?\.pdf(\})'
updated,n=re.subn(pattern,lambda m:m[1]+article_name+m[2],results_text)
assert n==1, 'Expected exactly one Fig. 3 image reference'
if updated!=results_text:results_path.write_text(updated,encoding='utf-8')
shutil.copyfile(OUT/'density_registry_15k.json',DATA/'pbe15k_density_temperature.json')
save(OUT/'figure_state.json',dict(fingerprint=fingerprint,completed_new=completed,
    canonical_hashes=canonical_hashes,article_figure_pdf=str(article_path),locked_canonical_files=locked,version=str(version)))
report=['PBE 15k: плотность Ti и сплавов',f'Новых температурных точек завершено: {len(completed)}/15.',
    f'В реестре: {len(registry)} записей (основные NPT, поздние окна сосуществования, статические объёмы).','',
    'Основная температурная серия чистого Ti:']
for p in sorted(pure,key=lambda p:p['T_K']):report.append(f"{p['T_K']:4d} K, {p['phase_initial']}: {p['rho']['mean']:.7f} ± {p['rho']['sem']:.7f} г/см³ (SEM); MSD(10 пс)={p['qc']['MSD_10ps_A2']:.3f} Å²; предупреждения: {p['qc']['warnings']}")
report+=['','Дополнительные оценки из последних 10 пс траекторий сосуществования:']
for p in ancillary:report.append(f"{p['T_K']} K, {p['phase_late']}, N=1296: {p['rho']['mean']:.7f} ± {p['rho']['sem']:.7f} г/см³; доля solid-like={p['solid_fraction_last10']:.4f}")
report+=['','Ограничения и определения:']+notes
(OUT/'REPORT_RU.txt').write_text('\n'.join(report)+'\n',encoding='utf-8')
print('\n'.join(report[:12]));print('Updated Fig. 3 and PBE comparison:',fingerprint[:12])
