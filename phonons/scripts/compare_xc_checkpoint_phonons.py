"""Matched DFT checkpoint sweep: DOS and Gamma-K-M-Gamma-A dispersions."""
import build_pbesol_phonons as b
from analyze_dpa4_pair import process
from pathlib import Path
import argparse,json,shutil
import numpy as np
import matplotlib.pyplot as plt

ROOT=b.ROOT
FINAL={'PBEsol':23100,'r2SCAN':23200}
PAIRS={'PBEsol':'PBEsol-matched-DFT-DPA4-20261031-02','r2SCAN':'r2SCAN-matched-DFT-DPA4-20261101-01'}
LAST_RUN={'PBEsol':'DPA4-PBEsol-matched-20261031-01','r2SCAN':'DPA4-r2SCAN-matched-20261101-01'}

def export(out,xc):
    inputs=out/'inputs';inputs.mkdir()
    geom=ROOT/'results'/('PBEsol-phonons-preliminary-20261031-02/inputs' if xc=='PBEsol' else 'r2SCAN-DFT-pair-inputs-20261101')
    shutil.copytree(geom,inputs/'geometry')
    src=ROOT/'results'/PAIRS[xc]
    manifest={k.replace('\\','/'):v for k,v in json.loads((src/'manifest-sha256.json').read_text()).items()}
    for method in ['DFT','DPA4']:
        name=method+'/phonopy_params.yaml';assert b.digest(src/name)==manifest[name]
        shutil.copyfile(src/name,inputs/(method+'-previous.yaml'))
    record=ROOT/'runs/alpha-ti-xc-phonons-20261021-01'
    for step in [5000,10000,15000,20000,FINAL[xc]]:
        run=record/LAST_RUN[xc] if step==FINAL[xc] else record/('DPA4-'+xc+'-checkpoints-20261102-01')/str(step)
        dest=inputs/str(step);dest.mkdir();shutil.copytree(run/'results',dest/'results')
        for name in ['protocol.json','worker.py']:shutil.copyfile(run/name,dest/name)
        parent=run if step==FINAL[xc] else run.parent
        for name in ['submission.json','collection.json','accounting.txt']:shutil.copyfile(parent/name,dest/name)
    metrics=json.loads((ROOT.parent/'audit/oct25/all_xc_audit/analysis/metrics.json').read_text())['groups']
    (inputs/'validation-metrics.json').write_text(json.dumps({str(s):metrics[xc+'-'+str(s)]['validation'] for s in [5000,10000,15000,20000,FINAL[xc]]},indent=2))
    return inputs

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--xc',choices=list(FINAL),required=True);ap.add_argument('--inputs',type=Path);ap.add_argument('--output',type=Path)
    args=ap.parse_args();xc=args.xc;steps=[5000,10000,15000,20000,FINAL[xc]]
    out=(args.output or ROOT/'results'/(xc+'-checkpoint-phonons-20261102-01')).resolve();out.mkdir(parents=True,exist_ok=False)
    if args.inputs:
        shutil.copytree(args.inputs.resolve(),out/'inputs');inputs=out/'inputs'
    else:inputs=export(out,xc)
    objects={'DFT':b.phonopy.load(inputs/'DFT-previous.yaml',produce_fc=False,symmetrize_fc=False)}
    diagnostics={};provenance={}
    for step in steps:
        src=inputs/str(step);cfg=json.loads((src/'protocol.json').read_text());s=json.loads((src/'results/summary.json').read_text())
        assert s['status']=='passed' and cfg['model']['step']==step and s['train_infos']['step']==step-1
        assert s['model']['sha256']==cfg['model']['sha256'] and cfg['model']['functional']==xc
        assert s['protocol_sha256']==b.digest(src/'protocol.json') and s['worker_sha256']==b.digest(src/'worker.py')
        for n,h in json.loads((src/'results/manifest.json').read_text()).items():assert b.digest(src/'results'/n)==h
        assert b.digest(inputs/'geometry/phonopy_disp.yaml')==cfg['input_hashes']['phonopy_disp.yaml']
        raw={}
        for key in ['force000','force001','force002']:
            assert b.digest(inputs/'geometry'/key/'POSCAR')==cfg['input_hashes'][key+'/POSCAR']==s['structures'][key]['POSCAR_sha256']
            raw[key]=np.load(src/'results'/key/'forces.npy',allow_pickle=False)
            assert raw[key].shape==(96,3) and np.isfinite(raw[key]).all()
        k=str(step);objects[k],diagnostics[k],_=process(inputs/'geometry',raw,out/('DPA4-'+k))
        provenance[k]={'step':step,'sha256':s['model']['sha256'],'job_id':s['job_id']}
        print(xc,step,'force constants ready',flush=True)
    old=b.phonopy.load(inputs/'DPA4-previous.yaml',produce_fc=False,symmetrize_fc=False)
    new_fc=objects[str(FINAL[xc])].force_constants
    if old.force_constants.shape[0]==len(old.primitive):
        new_fc=new_fc[old.primitive.p2s_map]
    fc_error=float(np.max(np.abs(old.force_constants-new_fc)))
    assert fc_error<1e-8
    nodes=np.array([[0,0,0],[1/3,1/3,0],[.5,0,0],[0,0,0],[0,0,.5]],float)
    reciprocal=np.linalg.inv(objects['DFT'].primitive.cell).T
    lengths=np.linalg.norm(np.diff(nodes,axis=0)@reciprocal,axis=1)
    ticks=np.r_[0,np.cumsum(lengths)];ticks/=ticks[-1]
    q=np.concatenate([np.linspace(a,c,151) for a,c in zip(nodes[:-1],nodes[1:])])
    xp=np.concatenate([np.linspace(a,c,151) for a,c in zip(ticks[:-1],ticks[1:])])
    dos={};bands={};stats={};ref_q=None
    for k,p in objects.items():
        assert len(p.supercell)==96 and np.allclose(p.masses,47.867)
        assert np.allclose(p.unitcell.cell,objects['DFT'].unitcell.cell,atol=1e-12,rtol=0)
        bands[k]=p.run_qpoints(q).frequencies.copy()
        mesh=p.run_mesh([50,50,50],is_gamma_center=True)
        assert mesh.weights.sum()==125000 and np.isfinite(mesh.frequencies).all() and np.isfinite(bands[k]).all()
        if ref_q is None:ref_q=mesh.qpoints.copy();ref_w=mesh.weights.copy()
        else:assert np.allclose(ref_q,mesh.qpoints) and np.array_equal(ref_w,mesh.weights)
        dd=p.run_total_dos(sigma=.05,freq_min=-1,freq_max=10,freq_pitch=.002,use_tetrahedron_method=False,smearing_function='Normal')
        x=dd.frequency_points.copy();y=dd.dos.copy()/2
        integral=float(np.trapezoid(y,x));assert abs(integral-3)<1e-6
        dos[k]=y*3/integral
        stats[k]={'integral':float(np.trapezoid(dos[k],x)),'integral_before_rescale':integral,'min_mesh_mode_THz':float(mesh.frequencies.min()),'imaginary_mesh_mode_weight_below_minus_0p001':int(np.sum(mesh.weights[:,None]*(mesh.frequencies<-.001))),'min_path_mode_THz':float(bands[k].min()),'max_mesh_mode_THz':float(mesh.frequencies.max()),'gamma_THz':bands[k][0].tolist()}
        print(xc,k,'spectra ready',flush=True)
    validation=json.loads((inputs/'validation-metrics.json').read_text())
    for step in steps:
        k=str(step);delta=dos[k]-dos['DFT'];bd=bands[k]-bands['DFT']
        stats[k].update(normalized_L1_DOS_difference=float(np.trapezoid(abs(delta),x)/3),relative_L2_DOS_difference=float(np.sqrt(np.trapezoid(delta**2,x)/np.trapezoid(dos['DFT']**2,x))),DOS_RMSE_0_to_9_THz=float(np.sqrt(np.mean(delta[(x>=0)&(x<=9)]**2))),sorted_path_frequency_RMSE_THz=float(np.sqrt(np.mean(bd**2))),sorted_path_max_difference_THz=float(np.max(abs(bd))),validation=validation[k])
    b.write_csv(out/'DOS.csv',['frequency_THz','DFT',*[str(s) for s in steps]],zip(x,dos['DFT'],*[dos[str(s)] for s in steps]))
    rows=[[k,i,xp[i],*q[i],*bands[k][i]] for k in objects for i in range(len(q))]
    b.write_csv(out/'bands.csv',['method','point','path_coordinate','q1','q2','q3',*[f'frequency_{i+1}_THz' for i in range(6)]],rows)
    np.savez_compressed(out/'spectra.npz',frequency_THz=x,qpoints=q,path_coordinate=xp,**{'DOS_'+k:v for k,v in dos.items()},**{'bands_'+k:v for k,v in bands.items()})
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.7})
    colors=dict(zip(steps,['#009E73','#CC79A7','#D55E00','#0072B2','#806000']))
    title='PBEsol' if xc=='PBEsol' else r'r$^2$SCAN'
    ymax=max(v.max() for v in dos.values())*1.09
    xmin=min(0,min(s['min_mesh_mode_THz'] for s in stats.values())-.1) if any(s['min_mesh_mode_THz']<-.001 for s in stats.values()) else 0
    def label(step):return f'DPA4 {step/1000:g}k'
    def style(ax):ax.spines[['top','right']].set_visible(False)
    def dos_panel(ax,ss):
        for step in ss:ax.plot(x,dos[str(step)],c=colors[step],lw=1.3,label=label(step))
        ax.plot(x,dos['DFT'],c='black',ls=(0,(4,2)),lw=1.4,label='DFT '+title,zorder=4)
        ax.set(xlim=(xmin,9.6),ylim=(0,ymax),xlabel='Frequency (THz)',ylabel=r'DOS (atom$^{-1}$ THz$^{-1}$)');style(ax)
    def save(fig,name):
        for ext in ['png','pdf','svg']:fig.savefig(out/(name+'.'+ext),dpi=320,facecolor='white')
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,4.8),layout='constrained');dos_panel(ax,steps)
    ax.legend(frameon=False,loc='upper left',ncol=2,fontsize=9);ax.set_title(r'$\alpha$-Ti: '+title+', matched phonon protocol')
    save(fig,xc+'_DOS_all_checkpoints')
    fig,axs=plt.subplots(5,2,figsize=(8,11.8),gridspec_kw={'width_ratios':[1.35,1]},layout='constrained')
    for row,step in enumerate(steps):
        left,right=axs[row]
        for j in range(6):
            left.plot(xp,bands[str(step)][:,j],c=colors[step],lw=.9,label=label(step) if j==0 else None)
            left.plot(xp,bands['DFT'][:,j],c='black',ls=(0,(4,2)),lw=.9,label='DFT '+title if j==0 else None)
        for t in ticks:left.axvline(t,c='.87',lw=.5,zorder=0)
        left.set(xlim=(0,1),ylim=(min(-.15,xmin),10),xticks=ticks,xticklabels=[r'$\Gamma$','K','M',r'$\Gamma$','A'],ylabel='Frequency (THz)')
        left.set_title(f'({chr(97+row*2)}) '+label(step),loc='left',fontsize=9);style(left)
        left.legend(frameon=False,fontsize=7,loc='upper left',ncol=2)
        dos_panel(right,[step]);right.set_title(f'({chr(98+row*2)}) '+label(step),loc='left',fontsize=9)
        if row<4:right.set_xlabel('')
        else:left.set_xlabel('Wave vector')
    save(fig,xc+'_bands_DOS_checkpoint_panels')
    summary={'functional':xc,'models':provenance,'statistics':stats,'force_diagnostics':diagnostics,'final_checkpoint_FC_reproduction_max_error':fc_error,'protocol':{'geometry':'same XC-specific DFT equilibrium for all models','supercell':[4,4,3],'displacement_A':.01,'q_mesh':[50,50,50],'gamma_centered':True,'sigma_THz':.05,'mass_amu':47.867,'path':'Gamma-K-M-Gamma-A','band_comparison':'sorted frequencies, no eigenvector branch matching'},'qualification':'DFT cutoff/kmesh/smearing/supercell convergence not established. Checkpoint comparisons are diagnostic, not an automatic model replacement.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for n in ['build_pbesol_phonons.py','analyze_dpa4_pair.py']:shutil.copyfile(Path(__file__).parent/n,out/n)
    shutil.copyfile(__file__,out/'reproduce.py')
    (out/'requirements.lock.txt').write_text('\n'.join(n+'=='+b.version(n) for n in ['phonopy','numpy','scipy','spglib','symfc','phonors','matplotlib','PyYAML','h5py'])+'\n')
    (out/'README.txt').write_text(f'Reproduce: python reproduce.py --xc {xc} --inputs inputs --output ../reproduced-{xc}\nShared DFT geometry and processing. See summary.json for protocol, provenance and limitations.\n')
    (out/'manifest-sha256.json').write_text(json.dumps({str(f.relative_to(out)).replace('\\','/'):b.digest(f) for f in out.rglob('*') if f.is_file()},indent=2)+'\n')
    print(json.dumps({k:{n:v for n,v in s.items() if n in ['normalized_L1_DOS_difference','relative_L2_DOS_difference','sorted_path_frequency_RMSE_THz','imaginary_mesh_mode_weight_below_minus_0p001']} for k,s in stats.items()},indent=2),flush=True)

if __name__=='__main__':main()
