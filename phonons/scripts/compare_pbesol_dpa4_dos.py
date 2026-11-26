"""Compare current DFT-PBEsol DOS with the article DPA4-PBEsol checkpoint.

Reuses audited real-space force constants. No DFT or ML inference is performed.
Both spectra use a Gamma-centered50^3 mesh, mass47.867amu and sigma0.05THz.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
try:
    import build_pbesol_phonons as b
except ImportError:
    # A portable exported copy can use an ordinary environment with dependencies.
    import os
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
else:
    np,plt=b.np,b.plt
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import cumulative_trapezoid

SCRIPT=Path(__file__).resolve()
ROOT=SCRIPT.parents[1]
MODEL_SHA='5a5359a713848d98a7746c70b46be65313d7d990c6bf39d68b40908dd9c7c295'
UNIT_FACTOR=15.633302
MASS=47.867
MESH=50
SIGMA=.05

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def prepare_inputs(out):
    dft=ROOT/'results/PBEsol-phonons-preliminary-20261031-02'
    ml=ROOT.parent/'audit/oct25/all_xc_audit'
    folder=out/'inputs';folder.mkdir()
    sources={
        'DFT-mesh50.npz':dft/'mesh50.npz','DFT-summary.json':dft/'summary.json',
        'DFT-reference-DOS.csv':dft/'PBEsol-DOS.csv',
        'ML-results.npz':ml/'results/17/equilibrium/results.npz',
        'ML-geometry.json':ml/'results/17/equilibrium/summary.json',
        'ML-task-summary.json':ml/'results/17/summary.json',
        'ML-protocol.json':ml/'protocol.json','ML-collection.json':ml/'collection-17.json',
        'ML-worker.py':ROOT.parent/'audit/oct25/all_xc_worker.py',
        'article-model-identity.json':ROOT.parent/'manuscript/supporting_data/checkpoint_identity.json',
        'DFT-source-POSCAR':dft/'inputs/force000/POSCAR',
    }
    for name,path in sources.items():shutil.copyfile(path,folder/name)
    return folder

def validate(inputs):
    read=lambda name:json.loads((inputs/name).read_text())
    task,geom,cfg,col,ident=[read(n) for n in ['ML-task-summary.json','ML-geometry.json','ML-protocol.json','ML-collection.json','article-model-identity.json']]
    assert task['status']=='passed' and task['task']['key']=='PBEsol-23100'
    assert task['model_sha256']==cfg['models']['PBEsol-23100']['sha256']==ident['models']['PBEsol']['checkpoint_sha256']==MODEL_SHA
    assert task['train_infos']['step']==23099 and cfg['models']['PBEsol-23100']['article_checkpoint']
    assert task['protocol_sha256']==sha(inputs/'ML-protocol.json')
    assert task['script_sha256']==sha(inputs/'ML-worker.py')
    for name,key in [('ML-results.npz','17/equilibrium/results.npz'),('ML-geometry.json','17/equilibrium/summary.json'),('ML-task-summary.json','17/summary.json')]:
        assert sha(inputs/name)==col['files'][key]
    assert geom['geometry_unchanged'] and geom['supercell']==[9,9,9] and geom['delta_A']==.01
    dft=read('DFT-summary.json')
    assert dft['q_mesh']==[50,50,50] and dft['q_mesh_gamma_centered'] and dft['mass_amu']==MASS
    return dft,geom,task

def direct(q,D,R):
    matrix=(np.exp(-2j*np.pi*(q@R))@D.reshape(len(D),-1)).reshape(-1,6,6)
    ev=np.linalg.eigvalsh(matrix,UPLO='U')
    return np.sign(ev)*np.sqrt(abs(ev))*UNIT_FACTOR,matrix

def ml_frequencies(inputs):
    with np.load(inputs/'ML-results.npz',allow_pickle=False) as z:
        C,D,R,q,bands=[z[n] for n in ['C','D','R','qpoints','bands_THz']]
    assert C.shape==D.shape==(729,6,6) and R.shape==(3,729)
    assert np.max(abs(D-C/MASS))<1e-12
    original,_=direct(q,D,R)
    cached_error=float(np.max(abs(original-bands)))
    assert cached_error<1e-5  # Legacy ASE used slightly rounded physical constants.
    grid=np.zeros((MESH,MESH,MESH,6,6),dtype=np.float64)
    indices=R.T.astype(int)%MESH
    assert len(np.unique(indices,axis=0))==len(D)
    grid[indices[:,0],indices[:,1],indices[:,2]]=D
    matrices=np.fft.fftn(grid,axes=(0,1,2))
    del grid
    # Check the FFT sign, lattice indexing, eigenvalues and phase convention.
    rng=np.random.default_rng(20261031)
    sample=np.vstack([np.zeros((1,3),int),rng.integers(0,MESH,size=(63,3))])
    df,dm=direct(sample/MESH,D,R)
    fm=matrices[sample[:,0],sample[:,1],sample[:,2]]
    matrix_error=float(np.max(abs(dm-fm)))
    assert matrix_error<1e-11
    hermitian_error=float(np.max(abs(matrices-matrices.swapaxes(-1,-2).conj())))
    ev=np.linalg.eigvalsh(matrices,UPLO='U')
    frequencies=(np.sign(ev)*np.sqrt(abs(ev))*UNIT_FACTOR).reshape(-1,6)
    ef=np.linalg.eigvalsh(fm,UPLO='U')
    assert np.max(abs(np.sign(ef)*np.sqrt(abs(ef))*UNIT_FACTOR-df))<1e-6
    return frequencies,dict(cached_band_max_difference_THz=cached_error,
        FFT_direct_max_matrix_difference=matrix_error,FFT_max_Hermitian_residual=hermitian_error,
        unit_conversion_THz=UNIT_FACTOR,mass_check_D_equals_C_divided_by_47p867=True)

def gaussian_dos(freq,weights,dx):
    x=np.arange(-1.,12.+dx/2,dx)
    f=freq.ravel();w=np.repeat(weights,6).astype(float)
    pos=(f-x[0])/dx;lower=np.floor(pos).astype(int);frac=pos-lower
    assert lower.min()>=0 and lower.max()<len(x)-2
    assert f.min()>x[0]+8*SIGMA and f.max()<x[-1]-8*SIGMA
    count=np.bincount(lower,weights=w*(1-frac),minlength=len(x))
    count+=np.bincount(lower+1,weights=w*frac,minlength=len(x))
    y=gaussian_filter1d(count,SIGMA/dx,mode='constant',truncate=8)/(weights.sum()*2*dx)
    before=float(np.trapezoid(y,x));y*=3/before
    assert abs(before-3)<1e-7
    return x,y

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path)
    ap.add_argument('--output',type=Path,default=ROOT/'results/PBEsol-DFT-vs-DPA4-20261031-01')
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    inputs=args.inputs.resolve() if args.inputs else prepare_inputs(out)
    dmeta,geom,task=validate(inputs)
    with np.load(inputs/'DFT-mesh50.npz',allow_pickle=False) as z:
        dft,weights=z['frequencies_THz'],z['weights']
    assert int(weights.sum())==MESH**3
    ml,checks=ml_frequencies(inputs)
    assert ml.shape==(MESH**3,6) and np.isfinite(ml).all()
    np.savez_compressed(out/'ML-Gamma50-frequencies.npz',frequencies_THz=ml)
    curves={}
    for key,frequencies,w in [('DFT_PBEsol',dft,weights),('DPA4_finetuned_PBEsol',ml,np.ones(len(ml)))]:
        x,y=gaussian_dos(frequencies,w,.0005);curves[key]=y
        xc,yc=gaussian_dos(frequencies,w,.001)
        error=float(np.trapezoid(abs(y-np.interp(x,xc,yc)),x)/3)
        assert error<2e-4
        checks[key+'_DOS_grid_refinement_L1_per_mode']=error
    original=np.genfromtxt(inputs/'DFT-reference-DOS.csv',delimiter=',',names=True)
    dif=np.interp(original['frequency_THz'],x,curves['DFT_PBEsol'])-original['DOS_modes_per_atom_per_THz']
    checks['DFT_vs_original_exact_Gaussian_L1_per_mode']=float(np.trapezoid(abs(dif),original['frequency_THz'])/3)
    assert checks['DFT_vs_original_exact_Gaussian_L1_per_mode']<2e-4
    with (out/'DOS-comparison.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['frequency_THz','DFT_PBEsol_modes_per_atom_per_THz','DPA4_finetuned_PBEsol_modes_per_atom_per_THz'])
        writer.writerows(zip(x,curves['DFT_PBEsol'],curves['DPA4_finetuned_PBEsol']))
    np.savez_compressed(out/'DOS-comparison.npz',frequency_THz=x,**curves)
    p=curves['DFT_PBEsol']/3;q=curves['DPA4_finetuned_PBEsol']/3
    difference={'W1_THz':float(np.trapezoid(abs(cumulative_trapezoid(p-q,x,initial=0)),x)),
                'relative_L2':float(np.sqrt(np.trapezoid((p-q)**2,x)/np.trapezoid(p*p,x))),
                'mean_frequency_DFT_THz':float(np.trapezoid(x*p,x)),
                'mean_frequency_DPA4_THz':float(np.trapezoid(x*q,x))}
    summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),model='DPA4-finetuned-PBEsol',checkpoint=23100,
        model_sha256=MODEL_SHA,ML_source_job=task['job_id'],DFT_source_jobs=dmeta['source_jobs'],
        geometry='Each method at its own independently relaxed equilibrium cell',ML_cell_A=geom['cell_A'],
        ML_volume_A3_per_atom=geom['V_A3_atom'],DFT_source_POSCAR_sha256=sha(inputs/'DFT-source-POSCAR'),
        ML_supercell=[9,9,9],DFT_supercell=[4,4,3],qmesh=[50,50,50],gamma_centered=True,
        mass_amu=MASS,Gaussian_sigma_THz=SIGMA,DOS_integrals={k:float(np.trapezoid(y,x)) for k,y in curves.items()},
        imaginary_threshold_THz=-.001,ML_modes_below_threshold=int((ml<-.001).sum()),
        DFT_modes_below_threshold=int(np.sum(weights[:,None]*(dft<-.001))),
        ML_min_frequency_THz=float(ml.min()),ML_max_frequency_THz=float(ml.max()),
        DFT_max_frequency_THz=float(dft.max()),ML_FC_processing='Existing ASE Frederiksen force correction, 3symmetrization cycles, acoustic=True; unchanged. Re-evaluate Fourier spectrum by exact-grid FFT; match cached bands and direct Fourier samples.',
        DOS_processing='Identical linear spectral deposition onto0.0005THz frequency grid then Gaussian convolution sigma0.05THz, truncate8sigma; refinement and exactDFTcomparison checks.',
        checks=checks,descriptive_difference=difference,
        qualification='Preliminary DFT convergence unestablished; unequal force-supercell sizes and own equilibrium geometries. Curve differences are descriptive, not an isolated same-geometry force-error benchmark.',
        input_hashes={str(p.relative_to(inputs)):sha(p) for p in inputs.iterdir() if p.is_file()},script_sha256=sha(SCRIPT))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(9,5.5))
    ax.plot(x,curves['DFT_PBEsol'],color='#16658e',lw=1.8,label='DFT-PBEsol (this work)')
    ax.plot(x,curves['DPA4_finetuned_PBEsol'],color='#d46b22',lw=1.7,label='DPA4-finetuned-PBEsol')
    ax.set(xlabel='Frequency (THz)',ylabel='DOS (modes / atom / THz)',xlim=(0,max(dft.max(),ml.max())+.3),ylim=(0,None))
    ax.grid(axis='y',color='#e3e8eb',lw=.65)
    ax.legend(loc='upper left',frameon=False,fontsize=10)
    fig.suptitle(r'$\alpha$-Ti phonon DOS  |  PBEsol',x=.105,y=.97,ha='left',fontsize=16)
    fig.text(.105,.888,r'Own equilibrium cells; common $q$ mesh $50^3$, $\sigma=0.05$ THz, mass 47.867 amu.',fontsize=10,color='#55616a')
    fig.text(.105,.05,r'Force cells: DFT $4\times4\times3$; DPA4 $9\times9\times9$ (checkpoint 23,100).',fontsize=9,color='#55616a')
    fig.text(.105,.018,'Preliminary DFT: numerical and supercell convergence checks deferred.',fontsize=9,color='#8a5a16')
    fig.subplots_adjust(left=.105,right=.98,bottom=.17,top=.84)
    for ext in ['png','pdf']:fig.savefig(out/('PBEsol-DFT-vs-DPA4-DOS.'+ext),dpi=240,facecolor='white')
    plt.close(fig)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    shutil.copyfile(SCRIPT,out/'reproduce.py')
    (out/'README.txt').write_text('Preliminary same-functional DOS comparison; each model at its own equilibrium.\n'
        'Use Python3.12 with NumPy2.5.4, SciPy1.18.1, Matplotlib3.11.2.\n'
        'Reproduce: python reproduce.py --inputs inputs --output ../comparison-reproduced\n'
        'No model inference or new DFT calculation is required. See summary.json for checks, hashes, processing and limitations.\n')
    manifest={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
    (out/'manifest-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'output':str(out),'checks':checks,'descriptive_difference':difference,
        'ML_max_frequency_THz':summary['ML_max_frequency_THz'],'ML_modes_below_threshold':summary['ML_modes_below_threshold']},indent=2),flush=True)

if __name__=='__main__':main()
