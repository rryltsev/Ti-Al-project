"""Render the user-selected article figure with durable source provenance."""
from pathlib import Path
import sys,os,json,time,hashlib,subprocess
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[2]
BASE=WORK/'audit/oct29/pbe15k_ti_temperature'
dependency_dir='audit/nov01/r2scan_viscosity_diagnostics/deps' if sys.version_info[:2]==(3,12) else 'audit/oct27/viscosity_5k_vs_15k/python_deps'
sys.path.insert(0,str(WORK/dependency_dir))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import argparse
_parser=argparse.ArgumentParser(description=__doc__)
_parser.add_argument('--profile',type=Path,default=ROOT/'figure_profile.json')
_parser.add_argument('--output-dir',type=Path,default=ROOT)
_parser.add_argument('--publish',action='store_true')
_parser.add_argument('--compile-article',action='store_true')
_args=_parser.parse_args()
PROFILE=_args.profile.resolve();OUTPUT=_args.output_dir.resolve()
OUTPUT.mkdir(parents=True,exist_ok=True)
p=json.loads(PROFILE.read_text(encoding='utf-8'))
krp=None
if p.get('krp'):
 krp_path=WORK/p['krp']['data_path']
 assert hashlib.sha256(krp_path.read_bytes()).hexdigest()==p['krp']['data_sha256'], 'KRP data changed; recalculate before plotting'
 krp=json.loads(krp_path.read_text(encoding='utf-8'))
rendering=p.get('rendering',{})
arrhenius=rendering.get('temperature_curve')=='weighted_arrhenius'
uniform_takeda=rendering.get('takeda_symbols')=='uniform_black_squares'
fits={}
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.labelsize':10,
 'axes.titlesize':10,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'mathtext.fontset':'dejavuserif'})
fig,axs=plt.subplots(1,2,figsize=(8.8,3.65),layout='constrained')
def series(ax,rows,x,**kw):
 rows=sorted(rows,key=x)
 return ax.errorbar([x(r) for r in rows],[r['eta_mPa_s'] for r in rows],
  yerr=[r['eta_std_mPa_s'] for r in rows],capsize=2.7,lw=1.2,ms=4.5,**kw)
for model,color,label in [('pbe','#2878B5','PBE'),('scan','#D55E00',r'r$^2$SCAN')]:
 series(axs[0],p['composition'][model],lambda r:1-r['c_Ti'],fmt='o-',color=color,label=label)
 if arrhenius:
  rows=sorted(p['temperature'][model],key=lambda r:r['T'])
  t=np.array([r['T'] for r in rows],float);y=np.array([r['eta_mPa_s'] for r in rows]);se=np.array([r['eta_std_mPa_s'] for r in rows])
  x=np.column_stack([np.ones(len(t)),1000/t]);sd=se/y;design=x/sd[:,None]
  pars=np.linalg.lstsq(design,np.log(y)/sd,rcond=None)[0]
  cov=np.linalg.inv(design.T@design);residual=(np.log(y)-x@pars)/sd
  fits[model]={'a':float(pars[0]),'B_K':float(1000*pars[1]),'eta0_mPa_s':float(np.exp(pars[0])),
   'covariance_a_b':cov.tolist(),'b_definition':'coefficient of 1000/T','chi2':float(residual@residual),'dof':len(t)-2,
   'weight_definition':'(eta/SE)^2 in ln(eta) regression; covariance not rescaled','input_points':rows}
  tt=np.linspace(t.min(),t.max(),400)
  axs[1].plot(tt,np.exp(pars[0]+pars[1]*1000/tt),color=color,lw=1.35,zorder=2)
  series(axs[1],rows,lambda r:r['T'],fmt='o',color=color,label=label,zorder=4)
 else:
  series(axs[1],p['temperature'][model],lambda r:r['T'],fmt='o-',color=color,label=label)
for c,(a,b,lo,hi) in p['references']['takeda_table3'].items():
 axs[0].scatter(1-float(c),10**(a+b/1873),s=24,marker='s',facecolors='black' if uniform_takeda or hi>=1873 else 'white',edgecolors='black',linewidths=.9,zorder=5)
axs[0].plot([],[],'ks',ms=4,label='Takeda fit')
if not uniform_takeda:axs[0].plot([],[],'ks',mfc='white',ms=4,label='Extrapolated by 3 K')
krp_labels=[]
if krp:
 for key,color,label,ls in [('experiment','#20836B',r'KRP: $\Delta H_{\rm exp}$','-.'),
                           ('pbe','#2878B5',r'KRP: $\Delta H_{\rm PBE}$','--'),
                           ('scan','#D55E00',r'KRP: $\Delta H_{\rm r^2SCAN}$',':')]:
  curve=krp['curves'][key]
  axs[0].plot(curve['x_Al'],curve['eta_mPa_s'],color=color,ls=ls,lw=1.5,label=label,zorder=3)
  krp_labels.append(label)
t=np.linspace(1850,2120,200);eta=.61*np.exp(27000/(8.31*t))
axs[1].plot(t,eta,'k--',lw=1,label='Ishikawa fit*')
axs[1].fill_between(t,eta*.897,eta*1.103,color='.65',alpha=.20,zorder=0)
axs[0].set(title='(a) Ti-Al, 1873 K',xlabel=r'$x_{\mathrm{Al}}$',ylabel=r'$\eta$ (mPa s)',xlim=(-.025,1.025),ylim=(0,8.2))
axs[0].set_xticks(np.arange(0,1.01,.2))
axs[1].set(title='(b) Pure Ti',xlabel='Temperature (K)',ylabel=r'$\eta$ (mPa s)',xlim=(1840,2130),ylim=(2.2,4.35))
axs[1].set_xticks([1873,1941,2000,2100])
for ax in axs:
 ax.grid(axis='y',alpha=.15,lw=.6);ax.spines[['top','right']].set_visible(False)
composition_legend=['PBE',r'r$^2$SCAN','Takeda fit']+([] if uniform_takeda else ['Extrapolated by 3 K'])+krp_labels
for ax,order,loc in [(axs[0],composition_legend,'lower left'),(axs[1],['PBE',r'r$^2$SCAN','Ishikawa fit*'],'upper right')]:
 handles,labels=ax.get_legend_handles_labels();lookup=dict(zip(labels,handles))
 if ax is axs[1] and arrhenius:
  for color,label in [('#2878B5','PBE'),('#D55E00',r'r$^2$SCAN')]:lookup[label]=Line2D([],[],color=color,marker='o',lw=1.35,ms=4.5)
 ax.legend([lookup[k] for k in order],order,frameon=False,loc=loc,fontsize=7.5 if krp and ax is axs[0] else 8)
for ext in ['png','pdf','svg']:fig.savefig(OUTPUT/f'viscosity_combined.{ext}',dpi=300 if arrhenius else 260)
if krp:
 fig.canvas.draw()
 panel_bbox=axs[0].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()).padded(.05)
 for ext in ['png','pdf','svg']:fig.savefig(OUTPUT/f'figure7a_krp.{ext}',dpi=300,bbox_inches=panel_bbox)
plt.close(fig)
sha=lambda f:hashlib.sha256(Path(f).read_bytes()).hexdigest()
manifest=dict(updated_unix=time.time(),profile_version=p['version'],profile_sha256=sha(PROFILE),
 composition=p['composition'],temperature=p['temperature'],selection=p['selection'],
 intermediate_temperature_search=p['intermediate_temperature_search'],sources_sha256=p['sources_sha256'],
 notes=p['notes'],rendering=rendering,arrhenius_fits=fits,output_sha256={e:sha(OUTPUT/f'viscosity_combined.{e}') for e in ['png','pdf','svg']})
if krp:
 manifest['krp']=dict(data_path=p['krp']['data_path'],data_sha256=sha(krp_path),
                      pure_ti=krp['pure_ti'],pure_al=krp['pure_al'],summaries=krp['summaries'],
                      limitation_notes=krp['limitation_notes'])
(OUTPUT/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(OUTPUT/'caption.tex').write_text(p['caption_tex']+'\n',encoding='utf-8')
if _args.publish:
 for ext in ['png','pdf','svg']:(BASE/f'figure7_current.{ext}').write_bytes((OUTPUT/f'viscosity_combined.{ext}').read_bytes())
 (WORK/'manuscript/figures/viscosity.pdf').write_bytes((OUTPUT/'viscosity_combined.pdf').read_bytes())
 # Preserve the manuscript's edited caption; locate it by its stable figure label.
 tex=WORK/'manuscript/sections/results.tex';s=tex.read_text(encoding='utf-8')
 label='\\label{fig:viscosity}'
 assert s.count(label)==1,'Expected one viscosity figure label'
 end=s.index(label);figure_start=s.rfind('\\begin{figure',0,end)
 start=s.find('\\caption{',figure_start,end)
 assert figure_start>=0 and start>=figure_start,'Viscosity caption not found'
 manuscript_caption=s[start:end].strip()
 manifest['manuscript_caption_tex']=manuscript_caption
 manifest['caption_publication']='Existing manuscript caption preserved; profile caption retained separately.'
 (OUTPUT/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 for path in [BASE/'figure7_data.json',WORK/'manuscript/supporting_data/viscosity_article_profile.json',WORK/'manuscript/supporting_data/viscosity_temperature_current.json']:
  path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
if _args.compile_article:
 build=WORK/'manuscript/build/ti-temperature';staging=build/'staging';staging.mkdir(parents=True,exist_ok=True)
 result=subprocess.run([str(WORK/'audit/latex-tools/tectonic.exe'),'-X','compile','--outdir',str(staging),'--outfmt','pdf','--untrusted','--keep-logs','main.tex'],cwd=WORK/'manuscript',capture_output=True,text=True)
 (BASE/'article-compile.txt').write_text(result.stdout+result.stderr,encoding='utf-8')
 if result.returncode:raise RuntimeError('Article compilation failed; inspect article-compile.txt')
 compiled=staging/'main.pdf';published=build/'main.pdf';locked=False
 try:compiled.replace(published)
 except OSError as exc:
  if getattr(exc,'winerror',None) not in (5,32,33) or not published.exists():raise
  locked=True;published=build/('main-'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'.pdf');compiled.replace(published)
 (BASE/'article-build.json').write_text(json.dumps(dict(unix=time.time(),status='compiled',article_pdf=str(published),canonical_pdf=str(build/'main.pdf'),canonical_locked=locked),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({'article_pdf':str(published)}))
print(json.dumps({'figure':str(OUTPUT/'viscosity_combined.pdf'),'profile':p['version'],'temperatures':[r['T'] for r in p['temperature']['pbe']]}))
