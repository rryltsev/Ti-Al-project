"""Six-panel article Figure2 variant, matched DFT/DPA4 and unchanged experiment.

PBE15k explicitly selected by the author on 2026-11-01. Does not overwrite
the article's current figure or text. All plotting inputs are exported.
"""
import build_pbesol_phonons as b
from analyze_dpa4_pair import process
from pathlib import Path
import argparse,csv,json,shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=b.ROOT;WORK=ROOT.parent
PAIRS={
 'PBE':ROOT/'results/PBE-matched-DFT-DPA4-20261101-01',
 'PBEsol':ROOT/'results/PBEsol-matched-DFT-DPA4-20261031-02',
 'r2SCAN':ROOT/'results/r2SCAN-matched-DFT-DPA4-20261101-01'}

def export(out):
    inputs=out/'inputs';inputs.mkdir()
    sources={}
    for xc,src in PAIRS.items():
        manifest=json.loads((src/'manifest-sha256.json').read_text())
        for kind in ['DFT','DPA4']:
            if xc=='PBE' and kind=='DPA4':continue
            name=kind+'/phonopy_params.yaml'
            normalized_manifest={k.replace('\\','/'):v for k,v in manifest.items()}
            assert b.digest(src/name)==normalized_manifest[name]
            dest=inputs/(xc+'-'+kind+'.yaml');shutil.copyfile(src/name,dest)
            sources[dest.name]=dict(source=str(src/name),sha256=b.digest(dest))
    run=ROOT/'runs/alpha-ti-xc-phonons-20261021-01/DPA4-PBE-15k-matched-20261101-01'
    shutil.copytree(run/'results',inputs/'PBE15k-forces')
    for name in ['protocol.json','submission.json','collection.json','accounting.txt','worker.py']:
        shutil.copyfile(run/name,inputs/('PBE15k-'+name))
    geom=ROOT/'results/PBE-DFT-pair-inputs-20261101';shutil.copytree(geom,inputs/'PBE-DFT-inputs')
    protocol=json.loads((run/'protocol.json').read_text());summary=json.loads((run/'results/summary.json').read_text())
    assert protocol['model']['step']==15000 and summary['train_infos']['step']==14999
    assert protocol['model']['sha256']==summary['model']['sha256']=='39506ecf42a49193717af8351521678b8665f662faf8fc374fec121f366feaaf'
    assert b.digest(run/'protocol.json')==summary['protocol_sha256']
    assert b.digest(run/'worker.py')==summary['worker_sha256']
    for n,h in json.loads((run/'results/manifest.json').read_text()).items():assert b.digest(run/'results'/n)==h
    raw={}
    for key in ['force000','force001','force002']:
        assert b.digest(geom/key/'POSCAR')==protocol['input_hashes'][key+'/POSCAR']
        raw[key]=np.load(run/'results'/key/'forces.npy')
    p,diag,_=process(geom,raw,inputs/'PBE15k-processed')
    shutil.copyfile(inputs/'PBE15k-processed/phonopy_params.yaml',inputs/'PBE-DPA4.yaml')
    (out/'PBE15k-diagnostics.json').write_text(json.dumps(diag,indent=2)+'\n')
    exp=WORK/'audit/oct25/pbe_checkpoint_phonons/reference-data.csv'
    shutil.copyfile(exp,inputs/'experimental-source.csv')
    old=json.loads((WORK/'audit/results/numerical_summary.json').read_text())
    ticks=old['phonons']['alpha_Ti/pbe']['q_special_normalized']
    cfg=dict(path='Gamma-K-M-Gamma-A',ticks=ticks,
        experimental_source=str(exp),experimental_sha256=b.digest(exp),experimental_count=77,
        citation='Stassis et al., room-temperature measurements reproduced in Souvatzis et al., Phys. Rev. Lett. 99, 015901 (2007), doi:10.1103/PhysRevLett.99.015901',
        coordinate_mapping='Experimental x/y retained without edits. Computed path segments mapped linearly between the archived normalized Gamma/K/M/Gamma/A ticks; no frequency shifts or experimental fitting.',
        checkpoints={'PBE':15000,'PBEsol':23100,'r2SCAN':23200},PBE15k_job=summary['job_id'],
        sources=sources,qualification='Draft at current converged SCF settings. Reference diagnostics and electronic/supercell convergence remain pending.')
    (inputs/'figure-config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    return inputs

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path);ap.add_argument('--output',type=Path,default=ROOT/'results/figure2-six-panels-20261101-02')
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    inputs=args.inputs.resolve() if args.inputs else export(out)
    cfg=json.loads((inputs/'figure-config.json').read_text())
    assert b.digest(inputs/'experimental-source.csv')==cfg['experimental_sha256']
    with (inputs/'experimental-source.csv').open() as f:
        exp=np.array([(float(r['Exp_x']),float(r['Exp_y'])) for r in csv.DictReader(f) if r['Exp_x'] and r['Exp_y']])
    assert len(exp)==cfg['experimental_count']==77 and np.isfinite(exp).all()
    b.write_csv(out/'experiment.csv',['normalized_path_coordinate','frequency_THz'],exp)
    points=[np.array(v,float) for v in [[0,0,0],[1/3,1/3,0],[.5,0,0],[0,0,0],[0,0,.5]]]
    paths=[np.linspace(a,c,151) for a,c in zip(points[:-1],points[1:])]
    ticks=np.array(cfg['ticks']);xp=np.concatenate([np.linspace(a,c,151) for a,c in zip(ticks[:-1],ticks[1:])])
    q=np.concatenate(paths);results={};arrays={};rows=[]
    for xc in PAIRS:
        results[xc]={}
        for kind in ['DFT','DPA4']:
            p=b.phonopy.load(inputs/(xc+'-'+kind+'.yaml'),produce_fc=False,symmetrize_fc=False)
            assert np.allclose(p.masses,47.867) and len(p.supercell)==96
            freq=p.run_qpoints(q).frequencies.copy()
            mesh=p.run_mesh([50,50,50],is_gamma_center=True)
            assert mesh.weights.sum()==125000 and np.isfinite(freq).all()
            assert min(float(freq.min()),float(mesh.frequencies.min()))>-.001
            dd=p.run_total_dos(sigma=.05,freq_min=-.3,freq_max=9.5,freq_pitch=.002,use_tetrahedron_method=False,smearing_function='Normal')
            x=dd.frequency_points.copy();dos=dd.dos.copy()/2;integral=float(np.trapezoid(dos,x))
            assert abs(integral-3)<1e-6
            dos*=3/integral
            arrays[xc+'_'+kind+'_bands']=freq;arrays[xc+'_'+kind+'_DOS']=dos
            results[xc][kind]=dict(min_frequency_THz=float(mesh.frequencies.min()),max_frequency_THz=float(mesh.frequencies.max()),
                gamma_THz=freq[0].tolist(),DOS_integral=float(np.trapezoid(dos,x)),modes_below_minus_0p001=0)
            for j in range(len(q)):rows.append([xc,kind,j,xp[j],*q[j],*freq[j]])
        results[xc]['sorted_band_RMSE_THz']=float(np.sqrt(np.mean((arrays[xc+'_DFT_bands']-arrays[xc+'_DPA4_bands'])**2)))
    b.write_csv(out/'bands.csv',['functional','method','point','normalized_path_coordinate','q1','q2','q3',*[f'frequency_{i+1}_THz' for i in range(6)]],rows)
    b.write_csv(out/'DOS.csv',['frequency_THz',*[xc+'_'+k for xc in PAIRS for k in ['DFT','DPA4']]],zip(x,*[arrays[xc+'_'+k+'_DOS'] for xc in PAIRS for k in ['DFT','DPA4']]))
    np.savez_compressed(out/'figure-data.npz',qpoints=q,path_coordinate=xp,frequency_THz=x,experiment=exp,**arrays)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8.5,'axes.titlesize':9,
        'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':.65,'pdf.fonttype':42,'svg.fonttype':'none',
        'xtick.direction':'out','ytick.direction':'out','legend.fontsize':8})
    fig,axs=plt.subplots(3,2,figsize=(7.25,7.65),gridspec_kw={'width_ratios':[1.38,1],'hspace':.28,'wspace':.3})
    styles={'DFT':('#0072B2',(0,(4,2))), 'DPA4':('#D55E00','-')}
    names={'PBE':'PBE','PBEsol':'PBEsol','r2SCAN':r'r$^2$SCAN'}
    for row,xc in enumerate(PAIRS):
        left,right=axs[row]
        for k in ['DPA4','DFT']:
            color,ls=styles[k];freq=arrays[xc+'_'+k+'_bands']
            for branch in range(6):left.plot(xp,freq[:,branch],c=color,ls=ls,lw=1.0,zorder=2)
            right.plot(x,arrays[xc+'_'+k+'_DOS'],c=color,ls=ls,lw=1.2)
        left.scatter(exp[:,0],exp[:,1],s=7,c='black',linewidths=0,zorder=4)
        for t in ticks:left.axvline(t,c='.86',lw=.5,zorder=0)
        left.set(xlim=(-.001,1.001),ylim=(-.15,9.6),xticks=ticks,xticklabels=[r'$\Gamma$','K','M',r'$\Gamma$','A'],yticks=[0,2,4,6,8],ylabel='Frequency (THz)')
        right.set(xlim=(0,9.4),ylim=(0,1.12),xticks=[0,2,4,6,8],yticks=[0,.2,.4,.6,.8,1],ylabel=r'DOS (atom$^{-1}$ THz$^{-1}$)')
        for col,ax in enumerate([left,right]):
            ax.text(.025,.965,'('+chr(97+2*row+col)+') '+names[xc],transform=ax.transAxes,va='top',ha='left',fontsize=9)
            ax.spines[['top','right']].set_visible(False)
            ax.tick_params(length=3,width=.65)
        if row==2:left.set_xlabel('Wave vector');right.set_xlabel('Frequency (THz)')
    handles=[Line2D([],[],c='black',marker='o',ls='None',ms=3,label='Experiment'),
        Line2D([],[],c=styles['DFT'][0],ls=styles['DFT'][1],lw=1.3,label='DFT'),
        Line2D([],[],c=styles['DPA4'][0],lw=1.3,label='DPA4-finetuned')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.997),ncol=3,frameon=False,handlelength=3,columnspacing=2)
    fig.subplots_adjust(left=.085,right=.985,bottom=.065,top=.94)
    for ext in ['png','pdf','svg']:fig.savefig(out/('figure2_phonons_dft_dpa4.'+ext),dpi=400,facecolor='white')
    plt.close(fig)
    caption=r'''Phonon dispersions (left column) and densities of states (right column) of $\alpha$-Ti for PBE (a,b), PBEsol (c,d), and r$^2$SCAN (e,f). Black points are digitized room-temperature experimental data reproduced by Souvatzis et al. [Phys. Rev. Lett. 99, 015901 (2007)] and attributed there to Stassis et al. Blue dashed curves denote DFT and orange solid curves the corresponding fine-tuned DPA4 models. Within each pair, both methods use the same DFT-relaxed geometry, $4\times4\times3$ supercell, $\pm0.01$ angstrom displacements, and Phonopy force-constant processing. The DPA4 checkpoints are 15k (PBE), 23.1k (PBEsol), and 23.2k (r$^2$SCAN). The DOS uses a Gamma-centred $50^3$ mesh, Ti mass 47.867 amu, and Gaussian broadening $\sigma=0.05$ THz, and integrates to three modes per atom.'''
    (out/'caption.txt').write_text(caption+'\n\nDRAFT QUALIFICATION: Current 550 eV / electronic 6x6x6 settings. Reference force/stress diagnostics for PBE and r2SCAN, and electronic/supercell convergence tests, remain pending. The central-pair force derivative cancels any common reference force exactly. The experimental digitization has no uncertainty estimates or branch assignments.\n',encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(dict(configuration=cfg,results=results,experiment_points_each_panel=len(exp),panels=6,
        common_axis='Archived normalized path ticks; experimental coordinates unchanged',
        model_geometry='Same XC-specific DFT equilibrium for DFT and DPA4; no new ML relaxation',
        script_sha256=b.digest(__file__)),indent=2)+'\n')
    for name in ['build_pbesol_phonons.py','analyze_dpa4_pair.py']:
        shutil.copyfile(Path(__file__).parent/name,out/name)
    shutil.copyfile(__file__,out/'reproduce.py')
    (out/'requirements.lock.txt').write_text('\n'.join(n+'=='+b.version(n) for n in ['phonopy','numpy','scipy','spglib','symfc','phonors','matplotlib','PyYAML','h5py'])+'\n')
    (out/'README.txt').write_text('Six-panel candidate Figure2. PBE15k explicitly selected by the author.\nReproduce: python reproduce.py --inputs inputs --output ../figure2-reproduced\nNo changes to the current manuscript figure or scientific text. See caption.txt for draft limitations.\n')
    (out/'manifest-sha256.json').write_text(json.dumps({str(p.relative_to(out)):b.digest(p) for p in out.rglob('*') if p.is_file()},indent=2)+'\n')
    print(json.dumps(results,indent=2),flush=True)
    print(out,flush=True)

if __name__=='__main__':main()
