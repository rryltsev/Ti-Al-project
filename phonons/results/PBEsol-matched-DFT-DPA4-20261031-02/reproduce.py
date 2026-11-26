"""Identical Phonopy processing of matched DFT and DPA4 finite forces.

Portable: python reproduce.py --inputs inputs --output ../reproduced
Requires build_pbesol_phonons.py alongside this script and its pinned environment.
"""
import build_pbesol_phonons as b
from pathlib import Path
import argparse,json,shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

HERE=Path(__file__).resolve().parent
ROOT=b.ROOT
RUN=ROOT/'runs/alpha-ti-xc-phonons-20261021-01/DPA4-PBEsol-matched-20261031-01'
DFT=ROOT/'results/PBEsol-phonons-preliminary-20261031-02'
COLORS={'DFT':'#16658e','DPA4':'#d66a19'}
LABELS={'DFT':'DFT-PBEsol','DPA4':'DPA4-finetuned-PBEsol'}

def export(out):
    inputs=out/'inputs';inputs.mkdir()
    shutil.copytree(DFT/'inputs',inputs/'DFT')
    shutil.copytree(RUN/'results',inputs/'DPA4')
    for name in ['protocol.json','submission.json','collection.json','accounting.txt','worker.py','job.sbatch']:
        shutil.copyfile(RUN/name,inputs/name)
    shutil.copyfile(DFT/'force_constants_ASR.npy',inputs/'DFT-previous-FC.npy')
    shutil.copyfile(DFT/'mesh50.npz',inputs/'DFT-previous-mesh.npz')
    return inputs

def process(inputs,raw,folder):
    folder.mkdir()
    f=np.stack([raw['force001'],raw['force002']])
    corrected=f-raw['force000'][None,:,:]
    drift=corrected.mean(axis=1);corrected-=drift[:,None,:]
    odd=.5*(corrected[0]-corrected[1]);central=np.stack([odd,-odd])
    p=b.make_phonon(inputs,central)
    before=p.force_constants.copy()
    p.symmetrize_force_constants(use_symfc_projector=True,show_drift=False)
    projected=p.force_constants.copy()
    p.symmetrize_force_constants(level=10,use_symfc_projector=False,show_drift=False)
    polish=float(np.max(np.abs(p.force_constants-projected)))
    assert polish<1e-8
    residual=b.fc_residual(p.force_constants)
    assert max(residual.values())<1e-9
    np.save(folder/'forces_central.npy',central)
    np.save(folder/'force_constants_ASR.npy',p.force_constants)
    p.save(filename=str(folder/'phonopy_params.yaml'),settings={'force_constants':True})
    gamma=p.run_qpoints([[0,0,0]]).frequencies[0].copy()
    mesh=p.run_mesh([50,50,50],is_gamma_center=True)
    assert mesh.weights.sum()==125000 and np.isfinite(mesh.frequencies).all()
    np.savez_compressed(folder/'mesh50.npz',qpoints=mesh.qpoints,weights=mesh.weights,frequencies_THz=mesh.frequencies)
    diagnostics=dict(reference_max_force_eV_A=float(np.linalg.norm(raw['force000'],axis=1).max()),
        force_drift_removed_eV_A=drift.tolist(),ASR_residual=residual,ASR_polish_change_eV_A2=polish,
        ASR_max_change_eV_A2=float(np.max(np.abs(before-p.force_constants))),gamma_THz=gamma.tolist(),
        min_frequency_THz=float(mesh.frequencies.min()),max_frequency_THz=float(mesh.frequencies.max()),
        imaginary_modes_below_minus_0p001_THz=int(np.sum(mesh.weights[:,None]*(mesh.frequencies<-.001))))
    return p,diagnostics,central

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path);ap.add_argument('--output',type=Path,default=ROOT/'results/PBEsol-matched-DFT-DPA4-20261031-01')
    args=ap.parse_args();out=args.output.resolve();out.mkdir(exist_ok=False,parents=True)
    inputs=args.inputs.resolve() if args.inputs else export(out)
    ml_summary=json.loads((inputs/'DPA4/summary.json').read_text())
    protocol=json.loads((inputs/'protocol.json').read_text())
    assert ml_summary['status']=='passed' and ml_summary['train_infos']['step']==23099
    assert ml_summary['model']['sha256']=='5a5359a713848d98a7746c70b46be65313d7d990c6bf39d68b40908dd9c7c295'
    assert ml_summary['protocol_sha256']==b.digest(inputs/'protocol.json')
    assert ml_summary['worker_sha256']==b.digest(inputs/'worker.py')
    for name,h in json.loads((inputs/'DPA4/manifest.json').read_text()).items():assert b.digest(inputs/'DPA4'/name)==h
    dft,audits=b.load_inputs(inputs/'DFT');ml={}
    for key in ['force000','force001','force002']:
        assert protocol['input_hashes'][key+'/POSCAR']==b.digest(inputs/'DFT'/key/'POSCAR')
        assert ml_summary['structures'][key]['POSCAR_sha256']==b.digest(inputs/'DFT'/key/'POSCAR')
        ml[key]=np.load(inputs/'DPA4'/key/'forces.npy',allow_pickle=False)
        assert ml[key].shape==(96,3) and np.isfinite(ml[key]).all()
    assert protocol['input_hashes']['phonopy_disp.yaml']==b.digest(inputs/'DFT/phonopy_disp.yaml')
    objects={};summary={};central={}
    for key,raw in [('DFT',dft),('DPA4',ml)]:
        objects[key],summary[key],central[key]=process(inputs/'DFT',raw,out/key)
    ref=np.load(inputs/'DFT-previous-FC.npy')
    fc_diff=float(np.max(np.abs(objects['DFT'].force_constants-ref)));assert fc_diff<1e-10
    points={'G':[0,0,0],'M':[.5,0,0],'K':[1/3,1/3,0],'A':[0,0,.5],'L':[.5,0,.5],'H':[1/3,1/3,.5]}
    edges=[('G','M'),('M','K'),('K','G'),('G','A'),('A','L'),('L','H'),('H','A'),('L','M'),('K','H')]
    paths=[np.linspace(points[a],points[c],101) for a,c in edges]
    dos={};bands={};segments={}
    fmin=min(-.3,min(s['min_frequency_THz'] for s in summary.values())-.3)
    fmax=max(s['max_frequency_THz'] for s in summary.values())+.3
    for key,p in objects.items():
        dd=p.run_total_dos(sigma=.05,freq_min=fmin,freq_max=fmax,freq_pitch=.002,use_tetrahedron_method=False,smearing_function='Normal')
        x=dd.frequency_points.copy();y=dd.dos.copy()/2
        integral=float(np.trapezoid(y,x));assert abs(integral-3)<1e-5
        y*=3/integral;dos[key]=y
        summary[key]['DOS_integral_before_rescale']=integral
        summary[key]['DOS_integral']=float(np.trapezoid(y,x))
        peaks,_=find_peaks(y,prominence=.025)
        summary[key]['DOS_peaks']=[dict(frequency_THz=float(x[i]),height=float(y[i])) for i in peaks]
        bb=p.run_band_structure(paths,is_band_connection=False)
        bands[key]=np.array(bb.frequencies);segments[key]=[v-v[0] for v in bb.distances]
    delta=central['DPA4']-central['DFT']
    diagnostics=dict(DFT_reproduced_FC_max_difference_eV_A2=fc_diff,model_sha256=ml_summary['model']['sha256'],
        DPA4_job_id=ml_summary['job_id'],DFT_job_ids={k:v['job_id'] for k,v in audits.items()},
        geometry='Identical DFT-PBEsol geometry and exact POSCAR bytes; no ML relaxation',
        supercell=[4,4,3],displacements='Same phonopy dataset, opposite 0.01 A displacements',
        processing='Identical shared make_phonon and process functions: reference subtraction, mean force removal, central pair, traditional fit, symfc projector and 10 ASR polish cycles',
        qmesh=[50,50,50],mass_amu=47.867,sigma_THz=.05,frequency_step_THz=.002,
        force_component_RMSE_eV_A=float(np.sqrt(np.mean(delta**2))),force_max_vector_error_eV_A=float(np.linalg.norm(delta,axis=2).max()),
        sorted_band_frequency_RMSE_THz=float(np.sqrt(np.mean((bands['DPA4']-bands['DFT'])**2))),
        sorted_band_max_difference_THz=float(np.max(np.abs(bands['DPA4']-bands['DFT']))),
        band_comparison_note='Frequencies sorted independently at each q, no eigenvector branch tracking.',
        qualification='Matched-protocol comparison at current DFT settings; DFT cutoff/kmesh/smearing/supercell convergence pending.',methods=summary)
    b.write_csv(out/'DOS.csv',['frequency_THz','DFT','DPA4'],zip(x,dos['DFT'],dos['DPA4']))
    rows=[]
    for i,(a,c) in enumerate(edges):
        for j,q in enumerate(paths[i]):rows.append([i,a,c,j,*q,segments['DFT'][i][j],*bands['DFT'][i,j],*bands['DPA4'][i,j]])
    b.write_csv(out/'bands.csv',['segment','start','end','point','q1','q2','q3','distance_no_2pi',*[f'DFT_{j}' for j in range(6)],*[f'DPA4_{j}' for j in range(6)]],rows)
    np.savez_compressed(out/'comparison.npz',frequency_THz=x,DOS_DFT=dos['DFT'],DOS_DPA4=dos['DPA4'],qpaths=paths,bands_DFT=bands['DFT'],bands_DPA4=bands['DPA4'])
    plt.rcParams.update({'font.size':11,'pdf.fonttype':42})
    fig,(ax,ad)=plt.subplots(1,2,figsize=(13,5.7),sharey=True,gridspec_kw={'width_ratios':[4,1.15],'wspace':.05})
    ticks=[0.];labels=[r'$\Gamma$'];cursor=0.
    for i,(a,c) in enumerate(edges):
        xx=segments['DFT'][i]+cursor
        for key in objects:
            for branch in range(6):ax.plot(xx,bands[key][i,:,branch],color=COLORS[key],lw=1.15,ls='-' if key=='DFT' else '--',label=LABELS[key] if i==0 and branch==0 else None)
        cursor=float(xx[-1]);ticks.append(cursor)
        labels.append('A|L' if i==6 else 'M|K' if i==7 else r'$\Gamma$' if c=='G' else c)
    for t in ticks:ax.axvline(t,color='#d6dce0',lw=.6,zorder=0)
    for key in objects:ad.plot(dos[key],x,color=COLORS[key],lw=1.3,ls='-' if key=='DFT' else '--')
    ax.set(xticks=ticks,xticklabels=labels,xlim=(0,cursor),ylim=(-.15,fmax),ylabel='Frequency (THz)',xlabel='Wave vector')
    ad.set(xlim=(0,max(max(v) for v in dos.values())*1.1),xlabel='DOS\n(modes / atom / THz)')
    handles,legend_labels=ax.get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='upper left',bbox_to_anchor=(.06,.865),fontsize=10,frameon=False,ncol=2)
    for aa in [ax,ad]:aa.spines[['top','right']].set_visible(False);aa.axhline(0,color='gray',ls=':',lw=.8)
    fig.suptitle(r'$\alpha$-Ti | DFT-PBEsol vs DPA4 | Matched phonon protocol',x=.065,ha='left',fontsize=17)
    fig.text(.065,.885,r'Same DFT geometry; $4\times4\times3$ force cell; $\pm0.01$ Å; identical Phonopy processing.',fontsize=10)
    fig.text(.065,.045,r'DOS: $q=50^3$, $\sigma=0.05$ THz, 47.867 amu. DPA4 checkpoint 23,100.',fontsize=9)
    fig.text(.065,.012,'Current DFT settings: 550 eV, electronic k = 6x6x6. Numerical and supercell convergence pending.',fontsize=9,color='#8a5a16')
    fig.subplots_adjust(left=.065,right=.98,top=.82,bottom=.17)
    for ext in ['png','pdf']:fig.savefig(out/('matched-bands-DOS.'+ext),dpi=220)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5.4))
    for key in objects:ax.plot(x,dos[key],color=COLORS[key],lw=1.6,label=LABELS[key])
    ax.set(xlabel='Frequency (THz)',ylabel='DOS (modes / atom / THz)',xlim=(0,fmax),ylim=(0,None))
    ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
    ax.set_title(r'$\alpha$-Ti | Matched geometry, supercell and Phonopy protocol',loc='left',pad=18)
    fig.text(.1,.045,r'$4\times4\times3$ force cell; $q=50^3$; $\sigma=0.05$ THz; integral = 3 modes/atom.',fontsize=9)
    fig.text(.1,.012,'Preliminary DFT: numerical and supercell convergence tests pending.',fontsize=9,color='#8a5a16')
    fig.subplots_adjust(left=.1,bottom=.17,right=.98,top=.88)
    for ext in ['png','pdf']:fig.savefig(out/('matched-DOS.'+ext),dpi=220)
    plt.close(fig)
    (out/'summary.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    shutil.copyfile(HERE/'build_pbesol_phonons.py',out/'build_pbesol_phonons.py')
    shutil.copyfile(__file__,out/'reproduce.py')
    if (DFT/'requirements.lock.txt').exists():shutil.copyfile(DFT/'requirements.lock.txt',out/'requirements.lock.txt')
    (out/'README.txt').write_text('Exact matched DFT-PBEsol / DPA4-PBEsol phonon protocol.\nReproduce with Python3.12 and requirements.lock.txt: python reproduce.py --inputs inputs --output ../reproduced\nThree DPA4 predictions were made under Slurm. No relaxation; exact DFT POSCARs. Checkpoint identity, inputs, forces and outputs SHA verified.\nDFT convergence in cutoff/electronic kmesh/smearing/supercell size remains pending. This comparison does not establish those convergences.\nSorted bands are compared without eigenvector branch tracking.\n')
    (out/'manifest-sha256.json').write_text(json.dumps({str(p.relative_to(out)):b.digest(p) for p in out.rglob('*') if p.is_file()},indent=2))
    print(json.dumps(diagnostics,indent=2),flush=True)

if __name__=='__main__':main()
