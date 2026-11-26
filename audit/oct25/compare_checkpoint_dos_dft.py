"""Compare checkpoint phonon DOS to the archived PBE DFT selection reference."""
from pathlib import Path
import os,sys,json,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BASE=HERE/'pbe_checkpoint_phonons';OUT=BASE/'dft_dos_comparison';OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'audit/oct20/deps'));os.environ['MPLBACKEND']='Agg'
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import cumulative_trapezoid
from scipy.signal import find_peaks
from ase.io import read

SOURCE=Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project\TiAl-papers\phonons\DFT_Ti_Ti3Al\Ti_hcp')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest={}
for n in ['total_dos.dat','POSCAR','phonopy.yaml','mesh.conf','make_total_dos_from_mesh.py','force_run_INCAR','force_run_KPOINTS']:
    p=SOURCE/n;(OUT/('DFT-'+n)).write_bytes(p.read_bytes());manifest[n]={'source':str(p),'sha256':sha(p)}
assert 'sigma=0.05' in (OUT/'DFT-total_dos.dat').read_text().splitlines()[1]
raw=np.loadtxt(OUT/'DFT-total_dos.dat');raw_area=np.trapezoid(raw[:,1],raw[:,0]);assert abs(raw_area-6)<1e-5
cell=read(OUT/'DFT-POSCAR',format='vasp');cp=cell.cell.cellpar()
data=np.load(BASE/'dos.npz');x=data['frequency_THz'];dx=x[1]-x[0]
ref=np.interp(x,raw[:,0],raw[:,1],left=0,right=0)/raw_area*3
ref*=3/np.trapezoid(ref,x)
keys=['5000','20000','23400'];names={'5000':'5k','20000':'20k','23400':'23.4k'}
colors={'5000':'#2866a6','20000':'#e28b23','23400':'#b83a53'}
def broaden(y,source_sigma,target_sigma):
    assert target_sigma>=source_sigma
    extra=np.sqrt(max(0,target_sigma**2-source_sigma**2))
    z=gaussian_filter1d(y,extra/dx,mode='constant',truncate=8) if extra>1e-12 else y.copy()
    return z*3/np.trapezoid(z,x)
def metrics(y,r):
    p=y/3;q=r/3;cdf=cumulative_trapezoid(p-q,x,initial=0)
    hi=(x>=6.5)&(x<=8.8)
    pk,prop=find_peaks(y,prominence=.04)
    return {'integral_per_atom':float(np.trapezoid(y,x)),
       'total_variation':float(.5*np.trapezoid(abs(p-q),x)),
       'wasserstein_THz':float(np.trapezoid(abs(cdf),x)),
       'L2_relative':float(np.sqrt(np.trapezoid((y-r)**2,x)/np.trapezoid(r*r,x))),
       'mean_frequency_THz':float(np.trapezoid(x*p,x)),
       'dominant_peak_THz':float(x[np.argmax(y)]),
       'high_frequency_peak_THz':float(x[hi][np.argmax(y[hi])]),
       'prominent_peaks_THz':x[pk].tolist()}
result={'criterion_clarification':'Author selected PBE 5k by agreement with the archived DFT phonon DOS, not experimental dispersion points. Previous experiment-only assessment does not test this criterion.',
        'DFT_source':manifest,'DFT_geometry':{'a_A':float(cp[0]),'c_A':float(cp[2]),'V_per_atom_A3':cell.get_volume()/len(cell)},
        'DFT_raw_integral':float(raw_area),'normalization':'3 modes per atom, no frequency rescaling or shifting',
        'MLIP_source_sha256':sha(BASE/'dos.npz'),'MLIP_supercell':[9,9,9],'MLIP_mesh':[50,50,50],
        'DFT_supercell':[4,4,3],'DFT_mesh':[40,40,30],'width_checks':{},
        'limitations':'Different DFT/MLIP reference cells, supercells and q meshes; archival reference comparison, not a controlled same-geometry DFT force-constant error. Raw DFT mesh.yaml is not present in the supplied folder.'}
curves={}
for width in [.05,.10,.15]:
    r=broaden(ref,.05,width);entry={'DFT':metrics(r,r)}
    for k in keys:
        for geom in ['common','relaxed']:
            y=broaden(data[k+'/'+geom],.0002*241.799,width)
            entry[k+'/'+geom]=metrics(y,r)
            if width==.05:curves[k+'/'+geom]=y
    result['width_checks'][str(width)]=entry
curves['DFT']=ref
np.savez_compressed(OUT/'normalized_dos_sigma_0p05.npz',frequency_THz=x,**curves)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for col,g in enumerate(['common','relaxed']):
    for row in range(2):
        ax=axs[row,col]
        ax.plot(x,ref,color='black',lw=1.8,label='PBE DFT reference',zorder=5)
        for k in keys:ax.plot(x,curves[k+'/'+g],color=colors[k],lw=1.2,label=names[k])
        ax.set(xlim=(0,10) if row==0 else (6.5,8.8),xlabel='Frequency (THz)',ylabel='DOS (modes / atom / THz)')
        ax.grid(alpha=.15)
        if row==0:ax.legend(frameon=False,fontsize=8,loc='upper left')
    axs[0,col].set_title(['(a) Common MLIP geometry','(b) Own MLIP equilibrium geometries'][col])
    axs[1,col].set_title(['(c) High-frequency region: common','(d) High-frequency region: own'][col])
fig.suptitle(r'PBE checkpoints vs archived DFT DOS; Gaussian $\sigma=0.05$ THz',fontsize=12)
fig.savefig(OUT/'checkpoint_dos_vs_dft.png',dpi=180);fig.savefig(OUT/'checkpoint_dos_vs_dft.pdf');plt.close(fig)
result['script_sha256']=sha(Path(__file__))
(OUT/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps(result['width_checks'],indent=2))
