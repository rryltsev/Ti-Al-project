"""Two article-figure alternatives with the same weighted Arrhenius fits."""
from pathlib import Path
import sys, os, json, hashlib, time
ROOT=Path(__file__).resolve().parent; WORK=ROOT.parents[2]
sys.path.insert(0,str(WORK/'audit/nov01/r2scan_viscosity_diagnostics/deps'))
os.environ['MPLCONFIGDIR']=str(ROOT/'cache')
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from scipy.stats import chi2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROFILE=WORK/'audit/nov01/viscosity_article_combined/figure_profile.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=json.loads(PROFILE.read_text(encoding='utf-8'))
uniform_takeda=p.get('rendering',{}).get('takeda_symbols')=='uniform_black_squares'
protected=[PROFILE,WORK/'manuscript/figures/viscosity.pdf',WORK/'manuscript/sections/results.tex']
before={str(f):sha(f) for f in protected}
assert [r['T'] for r in p['temperature']['pbe']]==[1873,2000,2100]
assert [r['T'] for r in p['temperature']['scan']]==[1873,1941,2000,2100]

def fit(rows):
    t=np.array([r['T'] for r in rows],float)
    eta=np.array([r['eta_mPa_s'] for r in rows])
    se=np.array([r['eta_std_mPa_s'] for r in rows])
    assert np.all(eta>se) and np.all(se>0)
    # Fit ln(eta/[mPa s]) = a + b*(1000/T), weighted by (eta/SE)^2.
    x=np.column_stack([np.ones(len(t)),1000/t]); y=np.log(eta); sd=se/eta
    design=x/sd[:,None];rhs=y/sd
    pars=np.linalg.lstsq(design,rhs,rcond=None)[0]
    cov=np.linalg.inv(design.T@design)
    residual=(y-x@pars)/sd;q=float(residual@residual);df=len(t)-2
    return {'a':float(pars[0]),'B_K':float(1000*pars[1]),'eta0_mPa_s':float(np.exp(pars[0])),
            'a_SE':float(np.sqrt(cov[0,0])),'B_SE_K':float(1000*np.sqrt(cov[1,1])),
            'Ea_eV':float(1000*pars[1]*8.617333262e-5),'Ea_SE_eV':float(1000*np.sqrt(cov[1,1])*8.617333262e-5),
            'covariance_a_b':cov.tolist(),'b_definition':'coefficient of 1000/T',
            'chi2':q,'dof':df,'p':float(chi2.sf(q,df)),
            'standardized_residuals':residual.tolist(),'predicted_eta_mPa_s':np.exp(x@pars).tolist(),
            'fit_temperature_range_K':[float(t.min()),float(t.max())],
            'weight_definition':'1/(SE_eta/eta)^2; covariance not rescaled by reduced chi-square',
            'input_points':rows}

fits={key:fit(p['temperature'][key]) for key in ['pbe','scan']}
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':9,'axes.labelsize':10,
 'axes.titlesize':10,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'mathtext.fontset':'dejavuserif',
 'svg.fonttype':'none'})
styles=[('pbe','#2878B5','PBE'),('scan','#D55E00',r'r$^2$SCAN')]
files={}
for mode in ['eta_T','lneta_inverse_T']:
    fig,axs=plt.subplots(1,2,figsize=(8.8,3.65),layout='constrained')
    for key,color,label in styles:
        rows=sorted(p['composition'][key],key=lambda r:1-r['c_Ti'])
        axs[0].errorbar([1-r['c_Ti'] for r in rows],[r['eta_mPa_s'] for r in rows],
            yerr=[r['eta_std_mPa_s'] for r in rows],fmt='o-',capsize=2.7,lw=1.2,ms=4.5,color=color,label=label)
        rows=sorted(p['temperature'][key],key=lambda r:r['T'])
        t=np.array([r['T'] for r in rows]);eta=np.array([r['eta_mPa_s'] for r in rows]);se=np.array([r['eta_std_mPa_s'] for r in rows])
        f=fits[key];tt=np.linspace(t.min(),t.max(),400);yy=np.exp(f['a']+f['B_K']/tt)
        if mode=='eta_T':
            xx=t;values=eta;errors=se;curve_x=tt;curve_y=yy
        else:
            xx=1e4/t;values=np.log(eta);curve_x=1e4/tt;curve_y=np.log(yy)
            errors=np.vstack([np.log(eta)-np.log(eta-se),np.log(eta+se)-np.log(eta)])
            # Exact transformed endpoints retain the originally displayed intervals.
            assert np.allclose(np.exp(values-errors[0]),eta-se,rtol=1e-13)
            assert np.allclose(np.exp(values+errors[1]),eta+se,rtol=1e-13)
        axs[1].plot(curve_x,curve_y,color=color,lw=1.35,zorder=2)
        axs[1].errorbar(xx,values,yerr=errors,fmt='o',capsize=2.7,lw=1.05,ms=4.5,color=color,zorder=4)
    for c,(a,b,lo,hi) in p['references']['takeda_table3'].items():
        axs[0].scatter(1-float(c),10**(a+b/1873),s=24,marker='s',facecolors='black' if uniform_takeda or hi>=1873 else 'white',edgecolors='black',linewidths=.9,zorder=5)
    axs[0].plot([],[],'ks',ms=4,label='Takeda fit')
    if not uniform_takeda:axs[0].plot([],[],'ks',mfc='white',ms=4,label='Extrapolated by 3 K')
    tr=np.linspace(1850,2120,300);ref=.61*np.exp(27000/(8.31*tr))
    if mode=='eta_T':
        rx=tr;ry=ref;lower=ref*.897;upper=ref*1.103
        axs[1].set(xlabel='Temperature (K)',ylabel=r'$\eta$ (mPa s)',xlim=(1840,2130),ylim=(2.2,4.35))
        axs[1].set_xticks([1873,1941,2000,2100])
    else:
        rx=1e4/tr;ry=np.log(ref);lower=np.log(ref*.897);upper=np.log(ref*1.103)
        axs[1].set(xlabel=r'$1/T$ ($10^{-4}$ K$^{-1}$)',ylabel=r'$\ln[\eta/(\mathrm{mPa\,s})]$',
                   xlim=(1e4/2130,1e4/1840),ylim=(np.log(2.2),np.log(4.35)))
        axs[1].set_xticks([4.8,5.0,5.2,5.4]);axs[1].set_yticks([.8,1.,1.2,1.4])
    axs[1].plot(rx,ry,'k--',lw=1,zorder=1)
    axs[1].fill_between(rx,lower,upper,color='.65',alpha=.20,zorder=0)
    axs[0].set(title='(a) Ti-Al, 1873 K',xlabel=r'$x_{\mathrm{Al}}$',ylabel=r'$\eta$ (mPa s)',xlim=(-.025,1.025),ylim=(0,8.2))
    axs[0].set_xticks(np.arange(0,1.01,.2));axs[1].set_title('(b) Pure Ti')
    for ax in axs:
        ax.grid(axis='y',alpha=.15,lw=.6);ax.spines[['top','right']].set_visible(False)
    handles,labels=axs[0].get_legend_handles_labels();lookup=dict(zip(labels,handles));order=['PBE',r'r$^2$SCAN','Takeda fit']+([] if uniform_takeda else ['Extrapolated by 3 K'])
    axs[0].legend([lookup[k] for k in order],order,frameon=False,loc='lower left')
    leg=[Line2D([],[],color=c,marker='o',lw=1.35,ms=4.5,label=l) for _,c,l in styles]
    leg.append(Line2D([],[],color='black',ls='--',lw=1,label='Ishikawa fit*'))
    axs[1].legend(handles=leg,frameon=False,loc='upper right' if mode=='eta_T' else 'upper left')
    stem=f'viscosity_arrhenius_{mode}'
    for ext in ['png','pdf','svg']:
        path=ROOT/f'{stem}.{ext}';fig.savefig(path,dpi=300);files[path.name]=sha(path)
    plt.close(fig)
    caption=p['caption_tex'].replace('Colored symbols show PBE and \\rscan{} estimates; lines connect sampled states.',
      r'Colored symbols show PBE and \rscan{} estimates. In (a), lines connect sampled states; in (b), solid colored lines show weighted Arrhenius fits, $\ln[\eta/(\mathrm{mPa\,s})]=a+B/T$, fitted with weights $(\eta/\sigma_\eta)^2$.')
    if mode!='eta_T':
        caption=caption.replace('temperature for pure Ti (b).',r'inverse temperature for pure Ti on a logarithmic viscosity scale (b).')
        caption=caption.replace('Error bars are formal estimator standard errors and exclude systematic uncertainties;',r'Error bars in (a) are formal estimator standard errors; those in (b) are the transformed endpoints $\ln[(\eta\pm\sigma_\eta)/(\mathrm{mPa\,s})]$. They exclude systematic uncertainties;')
    (ROOT/f'caption_{mode}.tex').write_text(caption+'\n',encoding='utf-8')

assert before=={str(f):sha(f) for f in protected},'Primary article changed during variant generation'
manifest={'created_unix':time.time(),'source_profile':str(PROFILE),'profile_sha256':sha(PROFILE),
 'source_profile_version':p['version'],'composition':p['composition'],'temperature':p['temperature'],
 'fits':fits,'transformation':{'x':'1/T expressed in units 1e-4 K^-1','y':'ln(eta/[mPa s])','errors':'exact log of original eta +/- 1 SE endpoints'},
 'notes':['Both coordinate representations use the identical fits and source points.','The selected PBE values 3.200 and 2.830 are preserved exactly as in the article profile.',
          'Primary article files remain unchanged; these are alternatives for review.','Fit curves are drawn only over the sampled temperature range.',
          'Experimental reference correlation and its shading are preserved from the existing figure profile/build script.','No fit confidence bands added; the gray band belongs to the experimental reference.'],
 'source_notes':p['notes'],'source_caption_tex':p['caption_tex'],'output_sha256':files,'protected_sha256':before}
(ROOT/'figure_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
note=['Два варианта рисунка для статьи с одинаковыми взвешенными аппроксимациями Аррениуса.',
 'Слева концентрационная зависимость прежняя. Справа: eta(T) либо ln(eta/[mPa s]) от 1/T.',
 'PBE: три выбранные точки 1873, 2000, 2100 K. r2SCAN: четыре точки 1873, 1941, 2000, 2100 K.',
 'Все исходные точки и погрешности сохранены. В ln-представлении показаны точные логарифмы границ eta +/- SE.',
 'Линии PBE/r2SCAN: eta = exp(a + B/T), eta в mPa s; веса в логарифмической регрессии (eta/SE)^2.',
 'Серая полоса относится только к сохранённой экспериментальной корреляции. Ковариация fit не масштабирована по chi2/dof.',
 'Основной рисунок и текст статьи не заменены. Форматы каждого варианта: PDF, SVG, PNG; подписи: caption_*.tex.', '']
for k,f in fits.items():note.append(f"{k}: a={f['a']:.9f}, B={f['B_K']:.6f} K; eta0={f['eta0_mPa_s']:.9f} mPa s; chi2={f['chi2']:.6f}, dof={f['dof']}; Ea={f['Ea_eV']:.6f} eV.")
(ROOT/'README_RU.txt').write_text('\n'.join(note)+'\n',encoding='utf-8')
print(json.dumps({k:{a:b for a,b in f.items() if a not in ['input_points','covariance_a_b']} for k,f in fits.items()},indent=2))
