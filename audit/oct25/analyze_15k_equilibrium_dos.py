"""Compare 5k and 15k at their own hcp equilibrium cells and at the DFT cell."""
from pathlib import Path
import os,sys,json,hashlib,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
NEW=HERE/'pbe_phonons_15k_equilibrium';OLD=HERE/'pbe_checkpoint_phonons'
FIXED=HERE/'pbe_phonons_five_checkpoints';DFT=HERE/'pbe_phonons_dft_geometry'
OUT=NEW/'comparison';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'audit/oct20/deps'))
os.environ['MPLBACKEND']='Agg';os.environ['OPENBLAS_NUM_THREADS']='4';os.environ['OMP_NUM_THREADS']='4'
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import cumulative_trapezoid
from ase import units
from ase.dft.kpoints import monkhorst_pack
from ase.io import read as read_atoms
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
new=read(NEW/'15000/summary.json');old=read(OLD/'5000/summary.json');old9=read(OLD/'size9-5000/summary.json')
cfg=read(NEW/'protocol.json');fixed_info=read(FIXED/'analysis.json')
assert new['status']=='passed' and new['optimization']['success'] and new['geometry_unchanged_during_phonons']
assert new['model_sha256']==cfg['models']['15000']['sha256']==fixed_info['metadata']['15000']['model_sha256']
assert new['train_infos']['step']==14999 and new['supercell']==[9,9,9] and new['displacement_A']==.01
assert new['results_sha256']==sha(NEW/'15000/results.npz')
assert new['script_sha256']==sha(NEW/'phonon_15k_equilibrium_remote.py') and new['protocol_sha256']==sha(NEW/'protocol.json')
assert new['relaxed_POSCAR_sha256']==sha(NEW/'15000/POSCAR-relaxed')
assert old9['results_sha256']==sha(OLD/'size9-5000/results.npz') and old9['status']=='passed'
assert old9['model_sha256']==old['model_sha256']==fixed_info['metadata']['5000']['model_sha256']
assert old9['source_geometries']['relaxed']==sha(OLD/'5000/POSCAR-relaxed')
assert new['versions']==old['versions']
for info,path in [(new,NEW/'15000/POSCAR-relaxed'),(old,OLD/'5000/POSCAR-relaxed')]:
    at=read_atoms(path,format='vasp');d=info['relaxed']
    assert np.isclose(at.cell.cellpar()[0],d['a_A']) and np.isclose(at.cell.cellpar()[2],d['c_A'])
    assert np.max(abs(np.array(d['stress_GPa'])))<.005 and d['max_force_eV_A']<1e-4

nf=np.load(NEW/'15000/results.npz');of=np.load(OLD/'size9-5000/results.npz');oq=np.load(OLD/'5000/results.npz')
arrays={'15000':{'D':nf['D'],'R':nf['R'],'qpoints':nf['qpoints'],'bands':nf['bands_THz']},
        '5000':{'D':of['relaxed-D'],'R':of['relaxed-R'],'qpoints':oq['qpoints'],'bands':of['relaxed-bands']}}
saved=np.load(FIXED/'dos_comparison.npz');x=saved['frequency_THz'];dx=x[1]-x[0]
bins=np.linspace(-2,14,16001);assert np.allclose(x,(bins[:-1]+bins[1:])/2)
factor=units._hbar*1e10/np.sqrt(units._e*units._amu)*241.799
meshes={'MP40':monkhorst_pack((40,40,30)),'MP50':monkhorst_pack((50,50,50)),
        'Gamma40':np.indices((40,40,30)).reshape(3,-1).T/np.array([40,40,30])}
def spectrum(q,D,R):
    results=[]
    for i in range(0,len(q),2048):
        dq=(np.exp(-2j*np.pi*(q[i:i+2048]@R))@D.reshape(len(D),-1)).reshape(-1,6,6)
        v=np.linalg.eigvalsh(dq,UPLO='U');results.append(np.sign(v)*np.sqrt(abs(v))*factor)
    return np.concatenate(results)
def broaden(y,width):
    w=np.sqrt(max(0,width**2-.05**2))
    z=gaussian_filter1d(y,w/dx,mode='constant',truncate=8) if w>1e-12 else y.copy()
    return z*3/np.trapezoid(z,x)
def distance(y,r):
    p=y/3;q=r/3;c=cumulative_trapezoid(p-q,x,initial=0)
    return {'L2_relative':float(np.sqrt(np.trapezoid((y-r)**2,x)/np.trapezoid(r*r,x))),
        'TV':float(.5*np.trapezoid(abs(p-q),x)),'W1_THz':float(np.trapezoid(abs(c),x)),
        'mean_THz':float(np.trapezoid(p*x,x)),'main_peak_THz':float(x[np.argmax(y)]),'integral':float(np.trapezoid(y,x))}
curves={'DFT':saved['DFT']};freq_info={}
for k,a in arrays.items():
    assert np.allclose(spectrum(a['qpoints'],a['D'],a['R']),a['bands'],atol=3e-6)
    for mesh,q in meshes.items():
        ts=time.monotonic();w=spectrum(q,a['D'],a['R'])
        assert w.min()>bins[0]+.4 and w.max()<bins[-1]-.4
        counts,_=np.histogram(w,bins=bins)
        y=gaussian_filter1d(counts.astype(float),.05/dx,mode='constant',truncate=8)/(len(q)*2*dx)
        assert abs(np.trapezoid(y,x)-3)<1e-6
        curves[k+'/own/'+mesh]=y;curves[k+'/DFT-cell/'+mesh]=saved[k+'/'+mesh]
        freq_info[k+'/'+mesh]={'min_THz':float(w.min()),'max_THz':float(w.max()),
             'imaginary_below_minus_0p01_THz':int((w<-.01).sum()),'wall_seconds':time.monotonic()-ts}
        print('DOS',k,mesh,freq_info[k+'/'+mesh],flush=True)
raw=np.loadtxt(DFT/'DFT-total_dos.dat');r=np.interp(x,raw[:,0],raw[:,1],left=0,right=0);r*=3/np.trapezoid(r,x)
assert np.allclose(r,curves['DFT'],atol=1e-14)
result={'metadata':{'5000':old,'15000':new},'geometry':{k:m['relaxed'] for k,m in [('5000',old),('15000',new)]},
 'DFT_geometry':{'a_A':2.95,'c_A':4.68,'V_per_atom_A3':17.6356114191036},
 'source_hashes':{'15k_results':sha(NEW/'15000/results.npz'),'5k_results':sha(OLD/'size9-5000/results.npz'),
 'fixed_cell_curves':sha(FIXED/'dos_comparison.npz'),'DFT_DOS':sha(DFT/'DFT-total_dos.dat')},
 'mesh_frequency_info':freq_info,'width_checks':{},'own_cell_rankings':{},'mesh_controls':{},
 'note':'Both own-cell DOS computed from raw 9^3 dynamical matrices with identical direct histogram/Gaussian postprocessing. Earlier 5k DOS used extra Gaussian convolution from 0.04836 to 0.05 THz; small differences can result.',
 'limitations':'DFT cell differs from model equilibrium cells; DFT force supercell 4x4x3 vs MLIP 9^3; absolute supercell convergence not established. Only harmonic hcp Ti examined; no melting or mixing enthalpy calculations.'}
for width in [.05,.10,.15]:
    ref=broaden(curves['DFT'],width);entry={'DFT':distance(ref,ref)}
    for k in ['5000','15000']:
        for geom in ['own','DFT-cell']:
            for mesh in meshes:entry[k+'/'+geom+'/'+mesh]=distance(broaden(curves[k+'/'+geom+'/'+mesh],width),ref)
    result['width_checks'][str(width)]=entry
    for mesh in meshes:
        result['own_cell_rankings'][f'{width}/{mesh}']={metric:sorted(['5000','15000'],key=lambda k:entry[k+'/own/'+mesh][metric]) for metric in ['L2_relative','TV','W1_THz']}
for k in arrays:
    result['mesh_controls'][k]={mesh:distance(curves[k+'/own/'+mesh],curves[k+'/own/MP40']) for mesh in ['MP50','Gamma40']}
np.savez_compressed(OUT/'dos_comparison.npz',frequency_THz=x,**curves)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
colors={'5000':'#2866a6','15000':'#8555ad'};names={'5000':'5k','15000':'15k'}
fig,axs=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
for i,ax in enumerate(axs):
    ax.plot(x,curves['DFT'],'k',lw=1.9,label='PBE DFT',zorder=5)
    for k in ['5000','15000']:ax.plot(x,curves[k+'/own/MP40'],color=colors[k],lw=1.5,label=names[k]+' own equilibrium')
    ax.set(xlim=(0,10) if i==0 else (6.2,8.8),xlabel='Frequency (THz)',ylabel='DOS (modes / atom / THz)',title=['(a) Own equilibrium cells','(b) High-frequency region'][i])
    ax.legend(frameon=False,fontsize=9,loc='upper left' if i==0 else 'upper right');ax.grid(alpha=.15)
fig.suptitle(r'hcp Ti: independent equilibrium $a,c$; $9^3$ supercells; $\sigma=0.05$ THz',fontsize=12)
for ext in ['png','svg']:fig.savefig(OUT/f'dos_5k_15k_equilibrium.{ext}',dpi=200)
plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
for ax,k in zip(axs,['5000','15000']):
    ax.plot(x,curves['DFT'],'k',lw=1.9,label='PBE DFT')
    ax.plot(x,curves[k+'/DFT-cell/MP40'],color=colors[k],ls='--',lw=1.3,label='Fixed DFT cell')
    ax.plot(x,curves[k+'/own/MP40'],color=colors[k],lw=1.5,label='Own equilibrium cell')
    ax.set(xlim=(0,9),xlabel='Frequency (THz)',ylabel='DOS (modes / atom / THz)',title=names[k]);ax.legend(frameon=False,fontsize=9);ax.grid(alpha=.15)
fig.suptitle('Geometry effect with identical mesh, broadening, mass and supercell',fontsize=12)
for ext in ['png','svg']:fig.savefig(OUT/f'dos_equilibrium_geometry_effect.{ext}',dpi=200)
plt.close(fig)
result['analysis_script_sha256']=sha(Path(__file__))
(OUT/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({'geometry':result['geometry'],'primary':result['width_checks']['0.05'],'rankings':result['own_cell_rankings']},indent=2))
