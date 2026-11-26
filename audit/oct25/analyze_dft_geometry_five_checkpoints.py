"""Compare all five PBE checkpoints at identical DFT geometry; no fitted shifts."""
from pathlib import Path
import os, sys, json, hashlib, time
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = HERE / 'pbe_phonons_dft_geometry'
NEW = HERE / 'pbe_phonons_dft_geometry_intermediate'
OUT = HERE / 'pbe_phonons_five_checkpoints'
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / 'audit/oct20/deps'))
os.environ['MPLBACKEND'] = 'Agg'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import cumulative_trapezoid
from ase import units
from ase.dft.kpoints import monkhorst_pack

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
keys = ['5000', '10000', '15000', '20000', '23400']
metadata = {}
source_checks = {}
for k in keys:
    src = NEW if k in ['10000', '15000'] else OLD
    cfg = read(src / 'protocol.json')
    m = read(src / k / 'summary.json')
    assert m['status'] == 'passed' and m['geometry_unchanged'] and not m['relaxation_performed']
    assert m['checkpoint'] == k and m['train_infos']['step'] == int(k)-1
    assert m['model_sha256'] == cfg['models'][k]['sha256']
    assert m['geometry_sha256'] == cfg['geometry_sha256'] == sha(OLD / 'POSCAR-DFT') == sha(src / 'POSCAR-DFT')
    assert m['results_sha256'] == sha(src / k / 'results.npz')
    assert m['script_sha256'] == sha(src / 'dft_geometry_phonons_remote.py') == sha(OLD / 'dft_geometry_phonons_remote.py')
    assert m['protocol_sha256'] == sha(src / 'protocol.json')
    assert m['supercell'] == [9,9,9] and m['displacement_A'] == .01 and m['mass_amu'] == 47.867
    assert len(list((src/k/'forces').glob('cache.*.json'))) == 13
    metadata[k] = m
    source_checks[k] = {'folder': str(src/k), 'results_sha256': m['results_sha256'], 'model_sha256': m['model_sha256']}
for m in metadata.values():
    assert m['cell_A'] == metadata['5000']['cell_A'] and m['fractional_positions'] == metadata['5000']['fractional_positions']
    assert m['versions'] == metadata['5000']['versions']

previous = read(OLD / 'analysis.json')
cache = np.load(OLD / 'dos_comparison.npz')
x = cache['frequency_THz']
bins = np.linspace(-2,14,16001)
assert np.allclose(x, (bins[:-1]+bins[1:])/2)
dx = x[1]-x[0]
sigma = .05
factor = units._hbar*1e10/np.sqrt(units._e*units._amu)*241.799
meshes = {'MP40': monkhorst_pack((40,40,30)), 'MP50': monkhorst_pack((50,50,50)),
          'Gamma40': np.indices((40,40,30)).reshape(3,-1).T / np.array([40,40,30])}

def spectrum(q,D,R):
    result=[]
    for i in range(0,len(q),2048):
        dq=(np.exp(-2j*np.pi*(q[i:i+2048]@R))@D.reshape(len(D),-1)).reshape(-1,6,6)
        v=np.linalg.eigvalsh(dq,UPLO='U')
        result.append(np.sign(v)*np.sqrt(abs(v))*factor)
    return np.concatenate(result)

def broaden(y,source_sigma,target_sigma):
    w=np.sqrt(max(0,target_sigma**2-source_sigma**2))
    z=gaussian_filter1d(y,w/dx,mode='constant',truncate=8) if w>1e-12 else y.copy()
    return z*3/np.trapezoid(z,x)

def distance(y,r):
    p=y/3; q=r/3; c=cumulative_trapezoid(p-q,x,initial=0)
    return {'L2_relative':float(np.sqrt(np.trapezoid((y-r)**2,x)/np.trapezoid(r*r,x))),
            'TV':float(.5*np.trapezoid(abs(p-q),x)), 'W1_THz':float(np.trapezoid(abs(c),x)),
            'mean_THz':float(np.trapezoid(p*x,x)), 'main_peak_THz':float(x[np.argmax(y)]),
            'integral':float(np.trapezoid(y,x))}

curves = {n: cache[n].copy() for n in cache.files if n == 'DFT' or any(n == k+'/'+mesh for k in keys for mesh in meshes)}
freq_info = previous['mesh_frequency_info'].copy()
for k in ['10000','15000']:
    a=np.load(NEW/k/'results.npz')
    test=spectrum(a['qpoints'],a['D'],a['R'])
    assert np.allclose(test,a['bands_THz'],atol=3e-6)
    for name,q in meshes.items():
        ts=time.monotonic()
        w=spectrum(q,a['D'],a['R'])
        assert w.min()>bins[0]+8*sigma and w.max()<bins[-1]-8*sigma
        counts,_=np.histogram(w,bins=bins)
        y=gaussian_filter1d(counts.astype(float),sigma/dx,mode='constant',truncate=8)/(len(q)*2*dx)
        assert abs(np.trapezoid(y,x)-3)<1e-6
        curves[k+'/'+name]=y
        freq_info[k+'/'+name]={'min_THz':float(w.min()),'max_THz':float(w.max()),
          'imaginary_below_minus_0p01_THz':int((w<-.01).sum()),'wall_seconds':time.monotonic()-ts}
        print('DOS',k,name,freq_info[k+'/'+name],flush=True)

ref=curves['DFT']
raw=np.loadtxt(OLD/'DFT-total_dos.dat')
area=np.trapezoid(raw[:,1],raw[:,0]); assert abs(area-6)<1e-5
recheck=np.interp(x,raw[:,0],raw[:,1],left=0,right=0)*3/area
recheck*=3/np.trapezoid(recheck,x)
assert np.allclose(ref,recheck,atol=1e-14)
errors=read(HERE/'pbe_checkpoint_sweep/analysis.json')['groups']
for k in keys:
    assert metadata[k]['model_sha256'] == read(HERE/'pbe_checkpoint_sweep/remote-input.json')['models'][k]['sha256']
result={'geometry_verified_identical':True,'relaxation':False,'metadata':metadata,'source_checks':source_checks,
        'DFT_POSCAR_sha256':sha(OLD/'POSCAR-DFT'),'DFT_DOS_sha256':sha(OLD/'DFT-total_dos.dat'),
        'reused_three_checkpoint_DOS_sha256':sha(OLD/'dos_comparison.npz'),
        'energy_force_errors_sha256':sha(HERE/'pbe_checkpoint_sweep/analysis.json'),
        'mesh_frequency_info':freq_info,'width_checks':{},'rankings':{},'pareto':{},'mesh_controls':{},
        'errors':errors,'limitations':previous['limitations']}
def pareto(values):
    return [k for k in keys if not any(all(a<=b for a,b in zip(values[j],values[k])) and
            any(a<b for a,b in zip(values[j],values[k])) for j in keys if j!=k)]
for width in [.05,.10,.15]:
    r=broaden(ref,.05,width)
    entry={'DFT':distance(r,r)}
    for k in keys:
        for mesh in meshes:
            entry[k+'/'+mesh]=distance(broaden(curves[k+'/'+mesh],.05,width),r)
    result['width_checks'][str(width)]=entry
    for mesh in meshes:
        result['rankings'][str(width)+'/'+mesh]={metric:sorted(keys,key=lambda k:entry[k+'/'+mesh][metric])
                                               for metric in ['L2_relative','TV','W1_THz']}
        for split in ['validation','test20']:
            for metric in ['L2_relative','TV','W1_THz']:
                values={k:[entry[k+'/'+mesh][metric],errors[k][split]['energy_rmse_meV_atom']] for k in keys}
                result['pareto'][f'{width}/{mesh}/{split}/{metric}']=pareto(values)
for k in keys:
    result['mesh_controls'][k]={name:distance(curves[k+'/'+name],curves[k+'/MP40']) for name in ['MP50','Gamma40']}
primary=result['width_checks']['0.05']
for k in ['5000','20000','23400']:
    for name in meshes:
        for metric,v in primary[k+'/'+name].items():
            assert abs(v-previous['width_checks']['0.05'][k+'/'+name][metric]) < 1e-12
np.savez_compressed(OUT/'dos_comparison.npz',frequency_THz=x,**curves)

colors={'5000':'#2866a6','10000':'#15956c','15000':'#8555ad','20000':'#e28b23','23400':'#b83a53'}
names={k:f'{int(k)/1000:g}k' for k in keys}
styles={'5000':'-','10000':'-','15000':'--','20000':'-.','23400':':'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,axs=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
for i,ax in enumerate(axs):
    ax.plot(x,ref,'k',lw=1.8,label='PBE DFT',zorder=8)
    for k in keys: ax.plot(x,curves[k+'/MP40'],color=colors[k],ls=styles[k],lw=1.35,label=names[k])
    ax.set(xlim=(0,10) if i==0 else (6.2,8.6),xlabel='Frequency (THz)',ylabel='DOS (modes / atom / THz)',
           title=['(a) Identical DFT geometry','(b) High-frequency region'][i])
    ax.legend(frameon=False,fontsize=9,loc='upper left' if i==0 else 'upper right');ax.grid(alpha=.15)
fig.suptitle(r'PBE checkpoints; $a=2.950$ Å, $c=4.680$ Å; $\sigma=0.05$ THz; no relaxation',fontsize=12)
for ext in ['png','svg']:fig.savefig(OUT/f'dos_five_checkpoints.{ext}',dpi=200)
plt.close(fig)

fig,axs=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
for ax,split,title in zip(axs[:2],['validation','test20'],['(a) Original validation set (149)','(b) Held-out liquid subset (20)']):
    for k in keys:
        ax.scatter(errors[k][split]['energy_rmse_meV_atom'],primary[k+'/MP40']['L2_relative'],color=colors[k],s=60)
        ax.annotate(names[k],(errors[k][split]['energy_rmse_meV_atom'],primary[k+'/MP40']['L2_relative']),
                    xytext=(5,7 if k!='23400' else -14),textcoords='offset points',fontsize=9,color=colors[k])
    ax.set(xlabel='Energy RMSE (meV/atom)',ylabel='Relative DOS L2 discrepancy',title=title)
    ax.margins(.18);ax.grid(alpha=.18)
ax=axs[2]
for metric,lab,style in [('L2_relative','Relative L2','o-'),('TV','Total variation','s--'),('W1_THz','W1 (relative to 5k)','^:')]:
    ys=[primary[k+'/MP40'][metric]/primary['5000/MP40'][metric] for k in keys]
    ax.plot([int(k)/1000 for k in keys],ys,style,label=lab,lw=1.3,ms=4)
ax.axhline(1,color='0.5',lw=.7)
ax.set(xlabel='Optimization steps (thousands)',ylabel='DOS discrepancy / 5k discrepancy',title='(c) Three spectral metrics')
ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.18)
fig.suptitle('Checkpoint tradeoff at fixed DFT geometry: lower errors are better',fontsize=12)
for ext in ['png','svg']:fig.savefig(OUT/f'checkpoint_tradeoff.{ext}',dpi=200)
plt.close(fig)
result['analysis_script_sha256']=sha(Path(__file__))
(OUT/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({'primary':primary,'rankings':result['rankings'],'primary_pareto':{k:v for k,v in result['pareto'].items() if k.startswith('0.05/MP40/')}},indent=2))
