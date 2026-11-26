"""Compare six harmonic Ti spectra, numerical controls and archived references."""
from pathlib import Path
import sys,os,json,hashlib,csv,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'pbe_checkpoint_phonons'
os.environ['MPLBACKEND']='Agg';os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
sys.path.insert(0,str(ROOT/'audit/oct20/deps'))
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from ase import units
from ase.dft.kpoints import monkhorst_pack

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
keys=['5000','20000','23400'];colors={'5000':'#2866a6','20000':'#e28b23','23400':'#b83a53'}
names={'5000':'5k','20000':'20k','23400':'23.4k'}
cfg=read(OUT/'protocol.json');arr={};meta={};summary={}
size9={};size9meta={}
for k in keys:
    m=read(OUT/k/'summary.json');assert m['status']=='passed' and m['checkpoint']==k
    assert m['model_sha256']==cfg['models'][k]['sha256']
    assert m['results_sha256']==sha(OUT/k/'results.npz') and m['protocol_sha256']==sha(OUT/'protocol.json')
    assert m['script_sha256']==sha(OUT/'phonon_checkpoint_remote.py')
    arr[k]=np.load(OUT/k/'results.npz');meta[k]=m
    if (OUT/('size9-'+k)/'summary.json').exists():
        sm=read(OUT/('size9-'+k)/'summary.json')
        assert sm['status']=='passed' and sm['checkpoint']==k and sm['model_sha256']==cfg['models'][k]['sha256']
        assert sm['results_sha256']==sha(OUT/('size9-'+k)/'results.npz')
        assert sm['script_sha256']==sha(OUT/'phonon_checkpoint_size9.py')
        assert all(sm['source_geometries'][g]==sha(OUT/k/('POSCAR-'+g)) for g in ['common','relaxed'])
        size9[k]=np.load(OUT/('size9-'+k)/'results.npz');size9meta[k]=sm
assert len(size9) in [0,3]
summary['primary_supercell']=[9,9,9] if size9 else [5,5,5]
def bands(k,g):return size9[k][g+'-bands'] if size9 else arr[k][g+'-base-bands']
q=arr['5000']['qaxis'];ticks=arr['5000']['ticks'];x=q/q[-1];xticks=ticks/q[-1]
labels=[r'$\Gamma$','K','M',r'$\Gamma$','A']
exp=[]
for r in csv.DictReader((OUT/'reference-data.csv').open()):
    if r['Exp_x'] and r['Exp_y']:exp.append([float(r['Exp_x']),float(r['Exp_y'])])
exp=np.array(exp);assert len(exp)>10
refbands=bands('5000','common');old=np.loadtxt(OUT/'archived-phonon_bands.dat')
assert np.allclose(old[:,0],q,atol=1e-5)
def diff(a,b):
    d=a-b
    return dict(rms_THz=float(np.sqrt(np.mean(d*d))),max_abs_THz=float(abs(d).max()),mean_THz=float(d.mean()))
summary['archive_reproduction_5k']=diff(arr['5000']['common-base-bands'],old[:,1:])
summary['experimental_note']='Digitized room-temperature points; nearest-frequency branch distance is descriptive, without branch labels/uncertainties, not a statistically calibrated phonon RMSE. No points removed.'
summary['controls']={};summary['comparisons']={};summary['experiment']={};summary['geometry']={}
# Same reference-based branch association for all spectra, in addition to nearest-branch distances.
reference_at_exp=np.column_stack([np.interp(exp[:,0],x,refbands[:,i]) for i in range(6)])
association=np.argmin(abs(reference_at_exp-exp[:,1,None]),axis=1)
for k in keys:
    summary['geometry'][k]={g:meta[k][g] for g in ['common','relaxed']}
    for g in ['common','relaxed']:
        tag=k+'/'+g;b=bands(k,g);base=arr[k][g+'-base-bands']
        summary['controls'][tag]={v:diff(arr[k][g+'-'+v+'-bands'],base) for v in ['half-displacement','large-cell']}
        summary['controls'][tag]['raw_to_corrected']=diff(arr[k][g+'-base-bands-raw'],base)
        if size9:
            summary['controls'][tag]['9_vs_7']=diff(b,arr[k][g+'-large-cell-bands'])
            summary['controls'][tag]['9_vs_5']=diff(b,base)
        summary['comparisons'][tag+'_vs_5k_same_geometry_protocol']=diff(b,bands('5000',g))
        predicted=np.column_stack([np.interp(exp[:,0],x,b[:,i]) for i in range(6)])
        distance=np.min(abs(predicted-exp[:,1,None]),axis=1)
        fixed=predicted[np.arange(len(exp)),association]-exp[:,1]
        summary['experiment'][tag]={'n':len(exp),'nearest_branch_rms_THz':float(np.sqrt(np.mean(distance**2))),
          'nearest_branch_MAE_THz':float(np.mean(distance)),'fixed_5k_branch_association_rms_THz':float(np.sqrt(np.mean(fixed**2)))}
        segments={}
        for i,name in enumerate(['Gamma-K','K-M','M-Gamma','Gamma-A']):
            mask=(exp[:,0]>=(xticks[i] if i else xticks[i]-.001))&(exp[:,0]<(xticks[i+1] if i<3 else xticks[i+1]+.001))
            segments[name]={'n':int(mask.sum()),'nearest_branch_rms_THz':float(np.sqrt(np.mean(distance[mask]**2))),
                            'fixed_5k_branch_association_rms_THz':float(np.sqrt(np.mean(fixed[mask]**2)))}
        summary['experiment'][tag]['segments']=segments
    summary['comparisons'][k+'_geometry_effect']=diff(bands(k,'relaxed'),bands(k,'common'))
for g in ['common','relaxed']:
    summary['comparisons']['23400_vs_20000_'+g]=diff(bands('23400',g),bands('20000',g))

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for i,g in enumerate(['common','relaxed']):
    ax=axs[0,i]
    for k in keys:
        b=bands(k,g)
        for branch in range(6):ax.plot(x,b[:,branch],color=colors[k],lw=1.1,alpha=.9,label=names[k] if branch==0 else None)
    ax.scatter(exp[:,0],exp[:,1],s=12,color='black',alpha=.65,label='Experiment (digitized)',zorder=4)
    ax.set_title(['(a) Common archived geometry','(b) Own equilibrium a and c'][i])
    ax.set(ylabel='Frequency (THz)',ylim=(-.15,9.8));ax.legend(fontsize=8,loc='upper right',frameon=False)
for k in keys[1:]:
    d=bands(k,'common')-refbands
    for i in range(6):axs[1,0].plot(x,d[:,i],color=colors[k],lw=1,alpha=.8,label=names[k]+' minus 5k' if i==0 else None)
for k in keys:
    d=bands(k,'relaxed')-bands(k,'common')
    for i in range(6):axs[1,1].plot(x,d[:,i],color=colors[k],lw=1,alpha=.8,label=names[k] if i==0 else None)
axs[1,0].set_title('(c) Checkpoint effect at fixed geometry')
axs[1,1].set_title('(d) Geometry effect: own minus common')
for ax in axs[1]:
    ax.axhline(0,color='.3',lw=.7);ax.set_ylabel('Frequency difference (THz)');ax.legend(frameon=False,fontsize=8)
for ax in axs.flat:
    for t in xticks:ax.axvline(t,color='.8',lw=.5,zorder=0)
    ax.set(xticks=xticks,xticklabels=labels,xlim=(0,1));ax.grid(axis='y',alpha=.15)
fig.savefig(OUT/'checkpoint_phonons.png',dpi=180);fig.savefig(OUT/'checkpoint_phonons.pdf');plt.close(fig)
if '--bands-only' in sys.argv:
    (OUT/'preliminary-bands-analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(json.dumps(summary['experiment'],indent=2));sys.exit(0)

# CPU batched Fourier sums, verified against ASE path eigenvalues before DOS.
factor=units._hbar*1e10/np.sqrt(units._e*units._amu)*241.799
def spectrum(qpoints,D,R):
    out=[]
    for s in range(0,len(qpoints),2048):
        phase=np.exp(-2j*np.pi*(qpoints[s:s+2048]@R))
        dq=(phase@D.reshape(len(D),-1)).reshape(-1,6,6)
        w=np.linalg.eigvalsh(dq,UPLO='U')
        out.append(np.sign(w)*np.sqrt(abs(w))*factor)
    return np.concatenate(out)
mesh=monkhorst_pack((50,50,50));dos={};summary['dos']={}
bins=np.linspace(-2,14,16001);freq=(bins[:-1]+bins[1:])/2;dx=freq[1]-freq[0];sigma=.0002*241.799
for k in keys:
    for g in ['common','relaxed']:
        tag=g+'-base'
        D=size9[k][g+'-D'] if size9 else arr[k][tag+'-D'];R=size9[k][g+'-R'] if size9 else arr[k][tag+'-R']
        test=spectrum(arr[k]['qpoints'],D,R)
        assert np.allclose(test,bands(k,g),atol=3e-6)
        w=spectrum(mesh,D,R);assert w.min()>bins[0]+8*sigma and w.max()<bins[-1]-8*sigma
        counts,_=np.histogram(w,bins=bins)
        y=gaussian_filter1d(counts.astype(float),sigma/dx,mode='constant',truncate=8)/(len(mesh)*2*dx)
        area=np.trapezoid(y,freq);assert abs(area-3)<1e-6
        dos[k+'/'+g]=y
        summary['dos'][k+'/'+g]={'integral_per_atom':float(area),'min_frequency_THz':float(w.min()),
            'max_frequency_THz':float(w.max()),'mean_frequency_THz':float(w.mean()),'imaginary_below_minus_0p01_THz':int((w<-.01).sum()),'mesh':[50,50,50],'sigma_THz':sigma}
        print('DOS',k,g,summary['dos'][k+'/'+g],flush=True)
np.savez_compressed(OUT/'dos.npz',frequency_THz=freq,**dos)
fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
for i,g in enumerate(['common','relaxed']):
    for k in keys:axs[i].plot(freq,dos[k+'/'+g],color=colors[k],label=names[k],lw=1.2)
    axs[i].set(xlabel='Frequency (THz)',ylabel='DOS (modes / atom / THz)',xlim=(0,10),title=['Common geometry','Own equilibrium geometry'][i]);axs[i].legend(frameon=False)
fig.savefig(OUT/'checkpoint_phonon_dos.png',dpi=180);fig.savefig(OUT/'checkpoint_phonon_dos.pdf');plt.close(fig)
summary['provenance']={k:{'job_id':meta[k]['job_id'],'wall_seconds':meta[k]['wall_seconds'],'node':meta[k]['node']} for k in keys}
summary['size9_provenance']=size9meta
summary['reference_hashes']={p.name:sha(p) for p in OUT.glob('archived-*')}
summary['reference_hashes']['reference-data.csv']=sha(OUT/'reference-data.csv')
summary['analysis_script_sha256']=sha(Path(__file__))
(OUT/'analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
print(json.dumps({k:v for k,v in summary.items() if k in ['archive_reproduction_5k','controls','comparisons','experiment']},indent=2))
