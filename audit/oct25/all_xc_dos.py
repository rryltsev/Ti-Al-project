"""Identical DOS processing, common/own geometries, and sensitivity to mesh and width."""
from pathlib import Path
import sys,os,json,hashlib,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'all_xc_audit';DEST=OUT/'analysis';DEST.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'audit/oct20/deps'))
os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import cumulative_trapezoid
from ase.dft.kpoints import monkhorst_pack
from ase import units
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg=read(OUT/'protocol.json');tasks=read(OUT/'tasks.json')
saved=np.load(HERE/'pbe_phonons_five_checkpoints/dos_comparison.npz');x=saved['frequency_THz'];dx=x[1]-x[0]
bins=np.linspace(-2,14,16001);factor=units._hbar*1e10/np.sqrt(units._e*units._amu)*241.799
meshes={'MP40':monkhorst_pack((40,40,30)),'MP50':monkhorst_pack((50,50,50)),
        'Gamma40':np.indices((40,40,30)).reshape(3,-1).T/np.array([40,40,30])}
curves={'DFT':saved['DFT']};sources={};frequency={}
for key,m in cfg['models'].items():
    if m['functional']=='PBE':
        for mesh in meshes:curves[key+'/fixed/'+mesh]=saved[str(m['step'])+'/'+mesh]
own=np.load(HERE/'pbe_phonons_15k_equilibrium/comparison/dos_comparison.npz')
for step in [5000,15000]:
    for mesh in meshes:curves[f'PBE-{step}/equilibrium/'+mesh]=own[f'{step}/own/'+mesh]
cache=DEST/'dos_cache';cache.mkdir(exist_ok=True)
def spectrum(q,D,R):
    result=[]
    for i in range(0,len(q),2048):
        dq=(np.exp(-2j*np.pi*(q[i:i+2048]@R))@D.reshape(len(D),-1)).reshape(-1,6,6)
        v=np.linalg.eigvalsh(dq,UPLO='U');result.append(np.sign(v)*np.sqrt(abs(v))*factor)
    return np.concatenate(result)
for t in tasks:
    if t['kind']!='properties':continue
    p=OUT/'results'/str(t['id'])
    if not (p/'summary.json').exists():continue
    assert read(p/'summary.json')['model_sha256']==cfg['models'][t['key']]['sha256']
    for geom in ['fixed','equilibrium']:
        src=p/geom/'results.npz'
        if not src.exists():continue
        tag=t['key']+'/'+geom;h=sha(src);dest=cache/(t['key']+'-'+geom+'.npz');meta=dest.with_suffix('.json')
        if dest.exists():
            info=read(meta);assert info['source_sha256']==h;a=np.load(dest)
            for mesh in meshes:curves[tag+'/'+mesh]=a[mesh];frequency[tag+'/'+mesh]=info['frequency'][mesh]
        else:
            a=np.load(src);assert np.allclose(spectrum(a['qpoints'],a['D'],a['R']),a['bands_THz'],atol=3e-6)
            ds={};fs={}
            for mesh,q in meshes.items():
                ts=time.monotonic();w=spectrum(q,a['D'],a['R'])
                assert w.min()>bins[0]+.4 and w.max()<bins[-1]-.4,(tag,w.min(),w.max())
                count,_=np.histogram(w,bins=bins);y=gaussian_filter1d(count.astype(float),.05/dx,mode='constant',truncate=8)/(len(q)*2*dx)
                assert abs(np.trapezoid(y,x)-3)<1e-6
                curves[tag+'/'+mesh]=y;ds[mesh]=y
                fs[mesh]={'min_THz':float(w.min()),'max_THz':float(w.max()),'imaginary_below_minus_0p01_THz':int((w<-.01).sum()),'seconds':time.monotonic()-ts}
                frequency[tag+'/'+mesh]=fs[mesh]
            np.savez_compressed(dest,**ds);meta.write_text(json.dumps({'source_sha256':h,'frequency':fs},indent=2))
            print('DOS',tag,fs['MP40'],flush=True)
        sources[tag]={'path':str(src),'sha256':h}
def broaden(y,width):
    w=np.sqrt(max(0,width**2-.05**2));z=gaussian_filter1d(y,w/dx,mode='constant',truncate=8) if w>1e-12 else y.copy()
    return z*3/np.trapezoid(z,x)
def distance(y,r):
    p=y/3;q=r/3;c=cumulative_trapezoid(p-q,x,initial=0)
    return {'L2_relative':float(np.sqrt(np.trapezoid((y-r)**2,x)/np.trapezoid(r*r,x))),
            'TV':float(.5*np.trapezoid(abs(p-q),x)),'W1_THz':float(np.trapezoid(abs(c),x)),
            'mean_THz':float(np.trapezoid(x*p,x)),'peak_THz':float(x[np.argmax(y)])}
result={'available_curves':list(curves),'source_hashes':sources,'frequency':frequency,'width_checks':{},'vs_article':{},'rankings_against_PBE_DFT':{},
        'limitations':'Only alpha-Ti harmonic DOS. DFT reference is PBE, not PBEsol/r2SCAN. DFT force supercell 4x4x3 versus MLIP 9x9x9; raw DFT q-grid unavailable. DOS was already used for PBE checkpoint development, not an independent blind test.'}
for width in [.05,.1,.15]:
    ref=broaden(curves['DFT'],width);entry={k:distance(broaden(y,width),ref) for k,y in curves.items() if k!='DFT'}
    result['width_checks'][str(width)]=entry
    for xc in ['PBE','PBEsol','r2SCAN']:
        kk=[k for k,m in cfg['models'].items() if m['functional']==xc]
        base=next(k for k in kk if cfg['models'][k]['article_checkpoint'])
        for geom in ['fixed','equilibrium']:
            for mesh in meshes:
                suffix='/'+geom+'/'+mesh;keys=[k for k in kk if k+suffix in entry]
                result['rankings_against_PBE_DFT'][f'{xc}/{geom}/{mesh}/{width}']={metric:sorted(keys,key=lambda k:entry[k+suffix][metric]) for metric in ['L2_relative','TV','W1_THz']}
                if base+suffix in curves:
                    for k in keys:result['vs_article'][f'{k}/{geom}/{mesh}/{width}']=distance(broaden(curves[k+suffix],width),broaden(curves[base+suffix],width))
np.savez_compressed(DEST/'dos_curves.npz',frequency_THz=x,**curves)
(DEST/'phonons.json').write_text(json.dumps(result,indent=2))
colors=['#2874a6','#20916f','#8e54ad','#d38c20','#bd4545']
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,3,figsize=(13,7.5),layout='constrained')
for j,xc in enumerate(['PBE','PBEsol','r2SCAN']):
    for i,geom in enumerate(['fixed','equilibrium']):
        ax=axs[i,j];ax.plot(x,curves['DFT'],'k',lw=1.4,label='PBE DFT reference',zorder=5)
        for color,key in zip(colors,[k for k,m in cfg['models'].items() if m['functional']==xc]):
            tag=key+'/'+geom+'/MP40'
            if tag in curves:ax.plot(x,curves[tag],lw=1.1,color=color,label=f"{cfg['models'][key]['step']/1000:g}k")
        ax.set(xlim=(0,10),xlabel='Frequency (THz)',ylabel='DOS (modes/atom/THz)',title=xc+' / '+('common DFT cell' if geom=='fixed' else 'own equilibrium cell'))
        ax.legend(fontsize=7,frameon=False);ax.grid(alpha=.15)
fig.suptitle('Harmonic α-Ti DOS; identical 9³ MLIP supercells and Gaussian width 0.05 THz\nPBE DFT is not a same-functional accuracy target for PBEsol or r²SCAN',fontsize=11)
fig.savefig(DEST/'phonon_dos_all_checkpoints.png',dpi=190);fig.savefig(DEST/'phonon_dos_all_checkpoints.pdf');plt.close(fig)
print('DOS curves',len(curves)-1,'of',15*2*3,flush=True)
