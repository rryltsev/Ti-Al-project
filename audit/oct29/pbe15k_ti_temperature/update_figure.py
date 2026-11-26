# User-selected article profile, 2026-11-01.
from pathlib import Path as _ProfilePath
import sys as _profile_sys, runpy as _profile_runpy
_profile_builder = _ProfilePath(__file__).resolve().parents[3] / 'audit/nov01/viscosity_article_combined/build_figure.py'
if _profile_builder.with_name('figure_profile.json').exists():
    _profile_sys.argv = [str(_profile_builder), '--publish'] + _profile_sys.argv[1:]
    _profile_runpy.run_path(str(_profile_builder), run_name='__main__')
    raise SystemExit(0)

"""Update article Fig.7 from archived estimates and completed new-Ti batches."""
from pathlib import Path
import sys,os,json,hashlib,time,subprocess
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[2]
sys.path.insert(0,str(WORK/'audit/oct27/viscosity_5k_vs_15k/python_deps'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache/matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
CFG=read(ROOT/'protocol.json')
archived=read(ROOT/'before/viscosity_current_oct29.json')
baseline=read(ROOT/'baseline_5k.json');known=read(ROOT/'known_15k.json')
sources=[ROOT/'before/viscosity_current_oct29.json',ROOT/'baseline_5k.json',ROOT/'known_15k.json',WORK/'audit/oct20/reference_values.json']
new=[];reports={}
for state in CFG['states']:
    f=ROOT/f"collected/analysis/latest_{state['id']}.json"
    if not f.exists():continue
    report=read(f)
    assert report['protocol_sha256']==sha(ROOT/'protocol.json') and report['checkpoint_sha256']==CFG['checkpoint']['sha256']
    row=report['points'][0];n=report['n_replicas'];T=state['temperature_K']
    assert report['temperature_K']==T and n>=5 and n%5==0
    assert (ROOT/f"collected/analysis/{state['id']}/batch_{n:04d}/summary.json").exists()
    new.append(dict(row['windows']['40ps'],T=T,n_replicas=n,checkpoint_step=15000,production_ps=40,stop_recommended=report['stop_recommended']))
    reports[state['id']]=report;sources.append(f)
new.sort(key=lambda r:r['T'])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':10,'axes.titlesize':10,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(8.8,3.65),layout='constrained')
def series(ax,rows,x,**style):
    rows=sorted(rows,key=x)
    ax.errorbar([x(r) for r in rows],[r['eta_mPa_s'] for r in rows],yerr=[r['eta_std_mPa_s'] for r in rows],capsize=3,lw=1.2,ms=5,**style)
for model,color,label in [('pbe','#2878B5','PBE 15k'),('scan','#D55E00',r'r$^2$SCAN')]:
    series(axs[0],[r for r in archived if r['dft']==model and r['T']==1873],lambda r:1-r['c_Ti'],fmt='o-',color=color,label=label)
refs=read(WORK/'audit/oct20/reference_values.json')
for c,(a,b,lo,hi) in refs['takeda_table3'].items():
    axs[0].scatter(1-float(c),10**(a+b/1873),s=26,marker='s',facecolors='black' if hi>=1873 else 'white',edgecolors='black',zorder=5)
axs[0].plot([],[],'ks',ms=4,label='Takeda fit');axs[0].plot([],[],'ks',mfc='white',ms=4,label='Fit extrapolated by 3 K')
series(axs[1],baseline,lambda r:r['T'],fmt='s:',color='#656565',mfc='white',label='PBE 5k, 20 ps')
series(axs[1],known,lambda r:r['T'],fmt='o--',color='#2878B5',mfc='white',label='PBE 15k, earlier 20 ps')
series(axs[1],[r for r in archived if r['dft']=='scan' and r['c_Ti']==1],lambda r:r['T'],fmt='o-',color='#D55E00',label=r'r$^2$SCAN')
if new:
    series(axs[1],new,lambda r:r['T'],fmt='D-',color='#23664C',label='PBE 15k, new 40 ps')
    for r in new:axs[1].annotate(f"N={r['n_replicas']}",(r['T'],r['eta_mPa_s']),xytext=(5,-14),textcoords='offset points',fontsize=7,color='#23664C')
t=np.linspace(1850,2120,150);eta=.61*np.exp(27000/(8.31*t))
axs[1].plot(t,eta,'k--',lw=1,label='Ishikawa fit*');axs[1].fill_between(t,eta*.897,eta*1.103,color='.7',alpha=.25,zorder=0)
axs[0].set(title='(a) Composition dependence, 1873 K',xlabel=r'$x_{\mathrm{Al}}$',ylabel=r'$\eta$ (mPa s)',xlim=(-.02,1.02),ylim=(0,8.3))
lo=min([2.1]+[r['eta_mPa_s']-2*r['eta_std_mPa_s'] for r in new]);hi=max([4.35]+[r['eta_mPa_s']+3*r['eta_std_mPa_s'] for r in new])
axs[1].set(title='(b) Pure Ti',xlabel='Temperature (K)',ylabel=r'$\eta$ (mPa s)',xlim=(1840,2130),ylim=(lo,hi))
for ax in axs:
    ax.grid(axis='y',alpha=.15);ax.spines[['top','right']].set_visible(False)
axs[0].legend(frameon=False,loc='lower left');axs[1].legend(frameon=False,loc='upper right',fontsize=7.3)
for suffix in ['png','pdf','svg']:
    fig.savefig(ROOT/f'figure7_current.{suffix}',dpi=220)
plt.close(fig)
pdf=ROOT/'figure7_current.pdf';(WORK/'manuscript/figures/viscosity.pdf').write_bytes(pdf.read_bytes())
caption=r'''\caption{STACIE viscosity versus composition at 1873~K (a) and temperature for pure Ti (b). The PBE~15k alloy estimates in (a) use 15 replicas of 40~ps each. Panel (b) retains the PBE~5k results for comparison (open squares, 20~ps) and the earlier PBE~15k pure-Ti exports (open circles, 20~ps; 25 replicas at 1873~K and 20 at 2000~K). New PBE~15k results, as completed, are shown separately as green diamonds from 40~ps trajectories, with their replica counts indicated; these estimates remain preliminary while sampling and diagnostic checks continue. Historical and new trajectory ensembles are not pooled. Error bars are estimator standard errors; lines connect sampled states. Black squares in (a) are values of the experimental correlations of Takeda \textit{et al.}\cite{Takeda2025}; open squares require a 3~K extrapolation. The dashed curve with an asterisk in (b) is a provisional transcription of the Ishikawa \textit{et al.}\cite{Ishikawa2012} correlation; shading denotes its reported 10.3\% uncertainty, not a specified confidence level.}'''
tex=WORK/'manuscript/sections/results.tex';text=tex.read_text(encoding='utf-8')
start=text.index('\\caption{STACIE viscosity versus composition')
end=text.index('\\label{fig:viscosity}',start)
updated=text[:start]+caption+'\n'+text[end:]
if updated!=text:tex.write_text(updated,encoding='utf-8')
provenance=dict(updated_unix=time.time(),figure=str(WORK/'manuscript/figures/viscosity.pdf'),baseline_5k=baseline,historical_15k=known,new_15k_40ps=new,new_analysis=reports,sources_sha256={str(p):sha(p) for p in sources},notes=['All error bars are estimator standard errors; systematic uncertainties are not included.','No pooling of historical and new data.','Experimental reference curves retained from the existing article.'])
for p in [ROOT/'figure7_data.json',WORK/'manuscript/supporting_data/viscosity_temperature_current.json']:
    p.write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if '--compile-article' in sys.argv:
    build=WORK/'manuscript/build/ti-temperature';build.mkdir(parents=True,exist_ok=True)
    # Compile independently of a PDF viewer holding the published file open on Windows.
    staging=build/'staging';staging.mkdir(exist_ok=True)
    completed=subprocess.run([str(WORK/'audit/latex-tools/tectonic.exe'),'-X','compile','--outdir',str(staging),'--outfmt','pdf','--untrusted','--keep-logs','main.tex'],cwd=WORK/'manuscript',capture_output=True,text=True)
    (ROOT/'article-compile.txt').write_text(completed.stdout+completed.stderr,encoding='utf-8')
    if completed.returncode:raise RuntimeError('Figure saved, but article compilation failed; see article-compile.txt')
    compiled=staging/'main.pdf';published=build/'main.pdf';locked=False
    assert compiled.is_file() and compiled.stat().st_size>0
    try:compiled.replace(published)
    except OSError as exc:
        if getattr(exc,'winerror',None) not in (5,32,33) or not published.exists():raise
        locked=True
        published=build/('main-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'.pdf')
        compiled.replace(published)
    (ROOT/'article-build.json').write_text(json.dumps(dict(unix=time.time(),status='compiled',article_pdf=str(published),canonical_pdf=str(build/'main.pdf'),canonical_locked=locked),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'figure':str(pdf),'known15k':[(r['T'],r['eta_mPa_s'],r['eta_std_mPa_s'],r['n_replicas']) for r in known],'new15k':[(r['T'],r['n_replicas']) for r in new]},ensure_ascii=True))
