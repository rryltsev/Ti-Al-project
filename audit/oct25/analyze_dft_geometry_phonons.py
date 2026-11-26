"""DOS at exact DFT geometry, mesh/width controls and matched-mesh geometry effect."""
from pathlib import Path
import os,sys,json,hashlib,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'pbe_phonons_dft_geometry';OLD=HERE/'pbe_checkpoint_phonons'
sys.path.insert(0,str(ROOT/'audit/oct20/deps'))
os.environ['MPLBACKEND']='Agg';os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import cumulative_trapezoid
from ase import units
from ase.dft.kpoints import monkhorst_pack
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text())
keys=['5000','20000','23400'];cfg=read(OUT/'protocol.json');arr={};metadata={}
for k in keys:
    m=read(OUT/k/'summary.json');assert m['status']=='passed' and m['geometry_unchanged'] and not m['relaxation_performed']
    assert m['checkpoint']==k and m['model_sha256']==cfg['models'][k]['sha256']
    assert m['geometry_sha256']==cfg['geometry_sha256']==sha(OUT/'POSCAR-DFT')
    assert m['results_sha256']==sha(OUT/k/'results.npz') and m['script_sha256']==sha(OUT/'dft_geometry_phonons_remote.py')
    assert m['protocol_sha256']==sha(OUT/'protocol.json')
    arr[k]=np.load(OUT/k/'results.npz');metadata[k]=m
assert all(metadata[k]['cell_A']==metadata['5000']['cell_A'] and metadata[k]['fractional_positions']==metadata['5000']['fractional_positions'] for k in keys)
bins=np.linspace(-2,14,16001);x=(bins[:-1]+bins[1:])/2;dx=x[1]-x[0];sigma=.05
factor=units._hbar*1e10/np.sqrt(units._e*units._amu)*241.799
meshes={'MP40':monkhorst_pack((40,40,30)),'MP50':monkhorst_pack((50,50,50)),
        'Gamma40':np.indices((40,40,30)).reshape(3,-1).T/np.array([40,40,30])}
def spectrum(q,D,R):
    result=[]
    for i in range(0,len(q),2048):
        dq=(np.exp(-2j*np.pi*(q[i:i+2048]@R))@D.reshape(len(D),-1)).reshape(-1,6,6)
        v=np.linalg.eigvalsh(dq,UPLO='U');result.append(np.sign(v)*np.sqrt(abs(v))*factor)
    return np.concatenate(result)
def broaden(y,source_sigma,target_sigma):
    w=np.sqrt(max(0,target_sigma**2-source_sigma**2));z=gaussian_filter1d(y,w/dx,mode='constant',truncate=8) if w>1e-12 else y.copy()
    return z*3/np.trapezoid(z,x)
def distance(y,r):
    p=y/3;q=r/3;c=cumulative_trapezoid(p-q,x,initial=0)
    return {'L2_relative':float(np.sqrt(np.trapezoid((y-r)**2,x)/np.trapezoid(r*r,x))),
      'TV':float(.5*np.trapezoid(abs(p-q),x)),'W1_THz':float(np.trapezoid(abs(c),x)),
      'mean_THz':float(np.trapezoid(p*x,x)),'main_peak_THz':float(x[np.argmax(y)]),'integral':float(np.trapezoid(y,x))}
curves={};freq_info={}
for k in keys:
    a=arr[k];test=spectrum(a['qpoints'],a['D'],a['R']);assert np.allclose(test,a['bands_THz'],atol=3e-6)
    for name,q in meshes.items():
        ts=time.monotonic();w=spectrum(q,a['D'],a['R']);assert w.min()>bins[0]+8*sigma and w.max()<bins[-1]-8*sigma
        counts,_=np.histogram(w,bins=bins);y=gaussian_filter1d(counts.astype(float),sigma/dx,mode='constant',truncate=8)/(len(q)*2*dx)
        assert abs(np.trapezoid(y,x)-3)<1e-6
        curves[k+'/'+name]=y;freq_info[k+'/'+name]={'min_THz':float(w.min()),'max_THz':float(w.max()),'imaginary_below_minus_0p01_THz':int((w<-.01).sum()),'wall_seconds':time.monotonic()-ts}
        print('DOS',k,name,freq_info[k+'/'+name],flush=True)
raw=np.loadtxt(OUT/'DFT-total_dos.dat');area=np.trapezoid(raw[:,1],raw[:,0]);assert abs(area-6)<1e-5
ref=np.interp(x,raw[:,0],raw[:,1],left=0,right=0)*3/area;ref*=3/np.trapezoid(ref,x);curves['DFT']=ref
old=np.load(OLD/'dos.npz');assert np.allclose(old['frequency_THz'],x)
for k in keys:
    for g in ['common','relaxed']:curves[k+'/old-'+g]=broaden(old[k+'/'+g],.0002*241.799,.05)
result={'geometry_verified_identical':True,'relaxation':False,'DFT_POSCAR_sha256':sha(OUT/'POSCAR-DFT'),
    'protocol_sha256':sha(OUT/'protocol.json'),'DFT_DOS_sha256':sha(OUT/'DFT-total_dos.dat'),'old_DOS_sha256':sha(OLD/'dos.npz'),
    'metadata':metadata,'mesh_frequency_info':freq_info,'width_checks':{},'matched_MP50_geometry_comparison':{},'mesh_controls':{},
    'limitations':'DFT uses 4x4x3 force supercell; MLIP uses 9x9x9. Raw DFT mesh.yaml unavailable, so both MP and Gamma-centered 40x40x30 tested. Finite-size and force-constant construction errors are not eliminated by fixing geometry.'}
for width in [.05,.10,.15]:
    r=broaden(ref,.05,width);entry={'DFT':distance(r,r)}
    for k in keys:
        for mesh in meshes:entry[k+'/'+mesh]=distance(broaden(curves[k+'/'+mesh],.05,width),r)
    result['width_checks'][str(width)]=entry
for k in keys:
    result['matched_MP50_geometry_comparison'][k]={g:distance(curves[k+'/'+g],ref) for g in ['old-common','old-relaxed','MP50']}
    result['mesh_controls'][k]={name:distance(curves[k+'/'+name],curves[k+'/MP40']) for name in ['MP50','Gamma40']}
np.savez_compressed(OUT/'dos_comparison.npz',frequency_THz=x,**curves)
colors={'5000':'#2866a6','20000':'#e28b23','23400':'#b83a53'};names={'5000':'5k','20000':'20k','23400':'23.4k'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
for i,ax in enumerate(axs):
    ax.plot(x,ref,'k',lw=1.8,label='PBE DFT',zorder=5)
    for k in keys:ax.plot(x,curves[k+'/MP40'],color=colors[k],lw=1.3,label=names[k])
    ax.set(xlim=(0,10) if i==0 else (6.2,8.8),xlabel='Frequency (THz)',ylabel='DOS (modes / atom / THz)',title=['(a) Identical DFT geometry','(b) High-frequency region'][i]);ax.grid(alpha=.15)
    ax.legend(frameon=False,fontsize=9,loc='upper left' if i==0 else 'upper right')
fig.suptitle(r'$a=2.950$ Å, $c=4.680$ Å; no relaxation; $\sigma=0.05$ THz',fontsize=12)
fig.savefig(OUT/'dos_at_dft_geometry.png',dpi=180);fig.savefig(OUT/'dos_at_dft_geometry.pdf');plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for ax,k in zip(axs,keys):
    ax.plot(x,ref,'k',lw=1.7,label='PBE DFT')
    ax.plot(x,curves[k+'/old-relaxed'],color=colors[k],ls='--',lw=1.2,label='Own equilibrium cell')
    ax.plot(x,curves[k+'/MP50'],color=colors[k],lw=1.3,label='DFT cell')
    ax.set(xlim=(0,9),xlabel='Frequency (THz)',ylabel='DOS (modes / atom / THz)',title=names[k]);ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.15)
fig.suptitle('Geometry effect: identical MLIP supercell, q mesh and broadening',fontsize=12)
fig.savefig(OUT/'dos_geometry_effect.png',dpi=180);fig.savefig(OUT/'dos_geometry_effect.pdf');plt.close(fig)
result['analysis_script_sha256']=sha(Path(__file__))
(OUT/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False))
print(json.dumps({'primary':result['width_checks']['0.05'],'geometry':result['matched_MP50_geometry_comparison'],'mesh':result['mesh_controls']},indent=2))
