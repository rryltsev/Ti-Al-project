"""Matched PBE DFT/DPA4 checkpoint DOS; portable inputs, no model selection."""
import build_pbesol_phonons as b
from analyze_dpa4_pair import process
from pathlib import Path
import argparse,json,shutil
import numpy as np
import matplotlib.pyplot as plt

ROOT=b.ROOT
STEPS=[5000,15000,20000]
RUNS={5000:'DPA4-PBE-matched-20261101-01',15000:'DPA4-PBE-15k-matched-20261101-01',20000:'DPA4-PBE-20k-matched-20261102-01'}

def export(out):
    inputs=out/'inputs';inputs.mkdir()
    geom=ROOT/'results/PBE-DFT-pair-inputs-20261101'
    shutil.copytree(geom,inputs/'geometry')
    src=ROOT/'results/PBE-matched-DFT-DPA4-20261101-01'
    manifest={k.replace('\\','/'):v for k,v in json.loads((src/'manifest-sha256.json').read_text()).items()}
    assert b.digest(src/'DFT/phonopy_params.yaml')==manifest['DFT/phonopy_params.yaml']
    shutil.copyfile(src/'DFT/phonopy_params.yaml',inputs/'DFT.yaml')
    for step in STEPS:
        run=ROOT/'runs/alpha-ti-xc-phonons-20261021-01'/RUNS[step]
        dest=inputs/str(step);dest.mkdir()
        shutil.copytree(run/'results',dest/'results')
        for name in ['protocol.json','worker.py','submission.json','collection.json','accounting.txt']:
            shutil.copyfile(run/name,dest/name)
    return inputs

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path);ap.add_argument('--output',type=Path,default=ROOT/'results/PBE-checkpoint-DOS-20261102-01')
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    inputs=args.inputs.resolve() if args.inputs else export(out)
    objects={'DFT':b.phonopy.load(inputs/'DFT.yaml',produce_fc=False,symmetrize_fc=False)}
    diagnostics={};provenance={}
    for step in STEPS:
        src=inputs/str(step);cfg=json.loads((src/'protocol.json').read_text());s=json.loads((src/'results/summary.json').read_text())
        assert s['status']=='passed' and cfg['model']['step']==step and s['train_infos']['step']==step-1
        assert s['model']['sha256']==cfg['model']['sha256']
        assert s['protocol_sha256']==b.digest(src/'protocol.json') and s['worker_sha256']==b.digest(src/'worker.py')
        for name,h in json.loads((src/'results/manifest.json').read_text()).items():assert b.digest(src/'results'/name)==h
        assert b.digest(inputs/'geometry/phonopy_disp.yaml')==cfg['input_hashes']['phonopy_disp.yaml']
        raw={}
        for key in ['force000','force001','force002']:
            assert b.digest(inputs/'geometry'/key/'POSCAR')==cfg['input_hashes'][key+'/POSCAR']==s['structures'][key]['POSCAR_sha256']
            raw[key]=np.load(src/'results'/key/'forces.npy',allow_pickle=False)
            assert raw[key].shape==(96,3) and np.isfinite(raw[key]).all()
        key=str(step)
        objects[key],diagnostics[key],_=process(inputs/'geometry',raw,out/('DPA4-'+key))
        provenance[key]={'checkpoint':step,'sha256':s['model']['sha256'],'job_id':s['job_id']}
    dos={};stats={};ref_q=None
    for key,p in objects.items():
        assert len(p.supercell)==96 and np.allclose(p.masses,47.867)
        assert np.allclose(p.unitcell.cell,objects['DFT'].unitcell.cell,atol=1e-12,rtol=0)
        mesh=p.run_mesh([50,50,50],is_gamma_center=True)
        assert mesh.weights.sum()==125000 and np.isfinite(mesh.frequencies).all()
        if ref_q is None:ref_q=mesh.qpoints.copy();ref_w=mesh.weights.copy()
        else:assert np.allclose(ref_q,mesh.qpoints) and np.array_equal(ref_w,mesh.weights)
        assert mesh.frequencies.min()>-.001
        dd=p.run_total_dos(sigma=.05,freq_min=-.3,freq_max=9.5,freq_pitch=.002,use_tetrahedron_method=False,smearing_function='Normal')
        x=dd.frequency_points.copy();y=dd.dos.copy()/2
        integral=float(np.trapezoid(y,x));assert abs(integral-3)<1e-6
        dos[key]=y*3/integral
        peak=int(np.argmax(y))
        stats[key]={'integral_before_rescale':integral,'integral':float(np.trapezoid(dos[key],x)),'peak_frequency_THz':float(x[peak]),'peak_height_per_atom_THz':float(dos[key][peak]),'min_mode_THz':float(mesh.frequencies.min()),'max_mode_THz':float(mesh.frequencies.max())}
    for step in STEPS:
        key=str(step);delta=dos[key]-dos['DFT']
        stats[key]['normalized_L1_DOS_difference']=float(np.trapezoid(abs(delta),x)/3)
        stats[key]['DOS_RMSE_0_to_9_THz']=float(np.sqrt(np.mean(delta[(x>=0)&(x<=9)]**2)))
    b.write_csv(out/'DOS.csv',['frequency_THz','DFT_PBE',*[f'DPA4_PBE_{s}' for s in STEPS]],zip(x,dos['DFT'],*[dos[str(s)] for s in STEPS]))
    np.savez_compressed(out/'DOS.npz',frequency_THz=x,**dos)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.7})
    colors={'5000':'#009E73','15000':'#D55E00','20000':'#0072B2'}
    ymax=max(y.max() for y in dos.values())*1.08
    def panel(ax,steps):
        for step in steps:
            key=str(step);ax.plot(x,dos[key],color=colors[key],lw=1.6,label=f'DPA4 {step//1000}k')
        ax.plot(x,dos['DFT'],color='black',ls=(0,(4,2)),lw=1.7,label='DFT PBE',zorder=4)
        ax.set(xlim=(0,9),ylim=(0,ymax),xlabel='Frequency (THz)',ylabel=r'DOS (atom$^{-1}$ THz$^{-1}$)')
        ax.spines[['top','right']].set_visible(False)
        ax.legend(frameon=False,loc='upper left',fontsize=9)
    def save(fig,name):
        for ext in ['png','pdf','svg']:fig.savefig(out/(name+'.'+ext),dpi=350,facecolor='white')
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.0,4.7),layout='constrained');panel(ax,STEPS)
    ax.set_title(r'$\alpha$-Ti: PBE, matched phonon protocol',fontsize=12)
    save(fig,'PBE_DOS_all_checkpoints')
    fig,axs=plt.subplots(1,3,figsize=(12,3.8),sharex=True,sharey=True,layout='constrained')
    for i,(ax,step) in enumerate(zip(axs,STEPS)):
        panel(ax,[step]);ax.set_title(f'({chr(97+i)}) DPA4 {step//1000}k vs DFT',fontsize=11)
        if i:ax.set_ylabel('')
    save(fig,'PBE_DOS_checkpoint_panels')
    for step in STEPS:
        fig,ax=plt.subplots(figsize=(6.4,4.4),layout='constrained');panel(ax,[step]);ax.set_title(f'PBE: DPA4 {step//1000}k vs DFT')
        save(fig,f'PBE_DOS_{step//1000}k')
    summary={'protocol':{'geometry':'same DFT-PBE equilibrium geometry','supercell':[4,4,3],'displacements_A':.01,'q_mesh':[50,50,50],'Gamma_centered':True,'sigma_THz':.05,'mass_amu':47.867,'integral_per_atom':3,'force_processing':'mean force removal; odd central pair; symfc projector; ASR polishing'},'models':provenance,'statistics':stats,'force_diagnostics':diagnostics,'qualification':'DFT numerical and supercell convergence not yet established. DOS mismatch alone does not select a universally best checkpoint.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for name in ['build_pbesol_phonons.py','analyze_dpa4_pair.py']:shutil.copyfile(Path(__file__).parent/name,out/name)
    shutil.copyfile(__file__,out/'reproduce.py')
    (out/'requirements.lock.txt').write_text('\n'.join(n+'=='+b.version(n) for n in ['phonopy','numpy','scipy','spglib','symfc','phonors','matplotlib','PyYAML','h5py'])+'\n')
    (out/'README.txt').write_text('PBE DFT vs DPA4 5k/15k/20k, identical geometry/displacements/processing.\nReproduce: python reproduce.py --inputs inputs --output ../reproduced-checkpoints\nAll DOS normalized to 3 modes/atom. See summary.json for provenance and scientific limitations.\n')
    (out/'manifest-sha256.json').write_text(json.dumps({str(f.relative_to(out)).replace('\\','/'):b.digest(f) for f in out.rglob('*') if f.is_file()},indent=2)+'\n')
    print(json.dumps(stats,indent=2),flush=True)

if __name__=='__main__':main()
