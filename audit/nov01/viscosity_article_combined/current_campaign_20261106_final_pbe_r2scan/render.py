from pathlib import Path
import sys,json,hashlib,time,csv
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'audit/nov01/r2scan_viscosity_diagnostics/deps'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protected=[ROOT/p for p in ['audit/nov01/viscosity_article_combined/figure_profile.json','manuscript/figures/viscosity.pdf','manuscript/build/ti-temperature/main.pdf','manuscript/sections/results.tex','manuscript/supporting_data/viscosity_article_profile.json','manuscript/supporting_data/viscosity_temperature_current.json']]
before={str(p.relative_to(ROOT)):sha(p) for p in protected if p.exists()}
D=ROOT/'audit/nov06/r2scan_ti_temperature/single_job_transition'
PBE=ROOT/'audit/nov06/pbe_user_stop_final/summary.json'
pbe=json.loads(PBE.read_text())['points'];rows=[];sources={str(PBE.relative_to(ROOT)):sha(PBE)}
for r in pbe:
 w=r['windows']['40ps'];rows.append(dict(functional='PBE',T=r['T'],eta=w['eta_mPa_s'],SE=w['eta_std_mPa_s'],Nnew=r['N'],Nold=0,estimator='New 40 ps, formal SE',stop_rule_met=r['stop_rule_met']))
for T in [1873,1941,2000,2100]:
 d=D/f'Ti_{T}K';latest=d/f'collected/analysis/latest_Ti_{T}K.json'
 if latest.exists():
  p=json.loads(latest.read_text());n=p['n_replicas'];assert n%5==0 and n>=5;x=p['points'][0]['combined_20ps'];e=x['eta_mPa_s'];se=x['eta_std_mPa_s'];old=x['legacy_replicas'];method='Legacy 20 ps plus two windows per new 40 ps; max(formal, whole-replica JK)'
 else:
  latest=d/'legacy/manifest.json';x=json.loads(latest.read_text())['published'];e=x['eta_mPa_s'];se=x['eta_std_mPa_s'];old=30 if T==1873 else 15;n=0;method='Verified legacy 20 ps; published formal SE, no complete new batch yet'
 sources[str(latest.relative_to(ROOT))]=sha(latest)
 rows.append(dict(functional='r2SCAN',T=T,eta=e,SE=se,Nold=old,Nnew=n,estimator=method))
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.labelsize':10,'axes.titlesize':10,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'mathtext.fontset':'dejavuserif'})
fig,ax=plt.subplots(figsize=(4.7,3.65),layout='constrained');fits={}
for functional,color in [('PBE','#2878B5'),('r2SCAN','#D55E00')]:
 r=sorted([x for x in rows if x['functional']==functional],key=lambda x:x['T']);t=np.array([x['T'] for x in r]);y=np.array([x['eta'] for x in r]);se=np.array([x['SE'] for x in r]);X=np.column_stack([np.ones(len(t)),1000/t]);sd=se/y;pars=np.linalg.lstsq(X/sd[:,None],np.log(y)/sd,rcond=None)[0];tt=np.linspace(t.min(),t.max(),400)
 ax.plot(tt,np.exp(pars[0]+pars[1]*1000/tt),color=color,lw=1.35,zorder=2)
 ax.errorbar(t,y,yerr=se,fmt='o',color=color,capsize=2.7,lw=1.2,ms=4.5,zorder=4)
 fits[functional]=dict(logA=float(pars[0]),B_K=float(pars[1]*1000),chi2=float(np.sum(((np.log(y)-X@pars)/sd)**2)),dof=len(t)-2,note='Descriptive weighted Arrhenius fit, not convergence evidence')
tt=np.linspace(1850,2120,200);yy=.61*np.exp(27000/(8.31*tt));ax.fill_between(tt,yy*.897,yy*1.103,color='.65',alpha=.2,zorder=0);ax.plot(tt,yy,'k--',lw=1,zorder=1)
ax.set(title='(b) Pure Ti',xlabel='Temperature (K)',ylabel=r'$\eta$ (mPa s)',xlim=(1840,2130),ylim=(2.2,4.35));ax.set_xticks([1873,1941,2000,2100]);ax.grid(axis='y',alpha=.15,lw=.6);ax.spines[['top','right']].set_visible(False)
ax.legend(handles=[Line2D([],[],color='#2878B5',marker='o',lw=1.35,ms=4.5,label='PBE'),Line2D([],[],color='#D55E00',marker='o',lw=1.35,ms=4.5,label=r'r$^2$SCAN'),Line2D([],[],color='black',ls='--',lw=1,label='Ishikawa fit*')],loc='upper right',frameon=False)
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'figure7b_current.{ext}',dpi=300)
plt.close(fig)
after={str(p.relative_to(ROOT)):sha(p) for p in protected if p.exists()};assert before==after,'Protected files changed concurrently: inspect; never restore automatically'
manifest=dict(created_unix=time.time(),preview_only=True,article_changed=False,points=rows,source_sha256=sources,protected_sha256=after,arrhenius_fits=fits,model_identity=dict(PBE='checkpoint 15000, final user-requested pause',r2SCAN='checkpoint 23200, confirmed legacy plus complete new batches only'),caveats=['Actual temperature is 1941 K; no calculation at 1942 K.','Completion of PBE is user-requested stop, not convergence for 1873/1941/2100.','r2SCAN 1873 and 1941 include five new 40 ps replicas per temperature; 2000/2100 retain legacy until complete analysis.','20 ps windows of one 40 ps replica form one cluster for combined uncertainty.','Errors do not include model systematics; different preparations and reservoir dependence limit interpretation.','Ishikawa correlation and +/-10.3 percent envelope retained from existing figure recipe.'])
(OUT/'supporting_data.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
with (OUT/'points.csv').open('w',newline='',encoding='utf-8') as f:
 writer=csv.DictWriter(f,fieldnames=['functional','T','eta','SE','Nold','Nnew','estimator','stop_rule_met']);writer.writeheader();writer.writerows(rows)
print(OUT);print(json.dumps(rows))
