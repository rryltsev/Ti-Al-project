"""Reproduce preliminary alpha-Ti PBEsol bands/DOS from audited finite forces.

Run from the project, or use --inputs <exported inputs> --output <new directory>.
No VASP/POTCAR is needed. Phonopy4.8.2 is pinned for this calculation.
"""
from pathlib import Path
import os
import sys

SCRIPT=Path(__file__).resolve()
ROOT=SCRIPT.parents[1]
for base in [ROOT, SCRIPT.parents[2]]:
    if (base/'.analysis-deps').is_dir():
        sys.path.insert(0,str(base/'.analysis-deps'))
        ROOT=base
        break
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.matplotlib-cache'))

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import platform
import shutil
import subprocess
import numpy as np
import phonopy
from phonopy.interface.vasp import read_vasp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RUN='alpha-ti-xc-phonons-20261021-01'
SOURCES={
 'force001':('production-03','force-001','PBEsol-force001-validation-20261028.json'),
 'force002':('production-04','force-002','PBEsol-force002-validation-20261029.json'),
 'force000':('production-04','force-000','PBEsol-force000-validation-20261031.json'),
}

def digest(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def export_inputs(out):
    target=out/'inputs'
    target.mkdir()
    record=ROOT/'runs'/RUN
    shutil.copyfile(record/'snapshot/PBEsol/recovery-01/phonopy_disp.yaml',target/'phonopy_disp.yaml')
    for key,(revision,step,audit) in SOURCES.items():
        folder=record/'collected/PBEsol'/revision/step
        dest=target/key
        dest.mkdir()
        for name in ['forces.npy','POSCAR','INCAR','KPOINTS']:
            shutil.copyfile(folder/name,dest/name)
        shutil.copyfile(ROOT/'results'/audit,dest/'audit.json')
    return target

def load_inputs(inputs, keys=None, functional='PBEsol'):
    forces={}
    audits={}
    for key in (SOURCES if keys is None else keys):
        folder=inputs/key
        audit=json.loads((folder/'audit.json').read_text())
        assert audit['accepted'] and audit['EDIFF_reached'] and audit['normal_termination']
        assert audit['functional']==functional and audit['scf_iterations']<200
        assert digest(folder/'forces.npy')==audit['output_hashes']['forces.npy']
        for name in ['POSCAR','INCAR','KPOINTS']:
            assert digest(folder/name)==audit['input_hashes'][name]
        force=np.load(folder/'forces.npy',allow_pickle=False)
        assert force.shape==(96,3) and np.isfinite(force).all()
        forces[key]=force
        audits[key]=audit
    return forces,audits

def make_phonon(inputs,forces):
    p=phonopy.load(inputs/'phonopy_disp.yaml',produce_fc=False,is_compact_fc=False,
                   symmetrize_fc=False,log_level=0)
    p.masses=[47.867]*len(p.primitive)
    assert len(p.primitive)==2 and len(p.supercell)==96
    assert np.array_equal(p.supercell_matrix,np.diag([4,4,3]))
    assert abs(p.unit_conversion_factor-15.633302)<1e-6
    reference=read_vasp(inputs/'force000/POSCAR')
    assert np.max(np.abs(reference.cell-p.supercell.cell))<1e-7
    delta=reference.scaled_positions-p.supercell.scaled_positions
    assert np.max(np.abs(delta-np.rint(delta)))<1e-7
    displacements=p.dataset['first_atoms']
    assert len(displacements)==2
    for i,key in enumerate(['force001','force002']):
        atoms=read_vasp(inputs/key/'POSCAR')
        delta=atoms.scaled_positions-reference.scaled_positions
        delta-=np.rint(delta)
        cart=delta@reference.cell
        assert displacements[i]['number']==0
        assert np.max(np.abs(cart[0]-displacements[i]['displacement']))<1e-10
        assert np.max(np.abs(cart[1:]))<1e-10
    assert np.max(np.abs(np.array(displacements[0]['displacement'])+displacements[1]['displacement']))<1e-12
    p.forces=forces
    p.produce_force_constants(fc_calculator='traditional',calculate_full_force_constants=True,show_drift=False)
    assert p.force_constants.shape==(96,96,3,3)
    return p

def fc_residual(fc):
    return dict(translation_max_eV_A2=float(np.max(np.abs(fc.sum(axis=1)))),
                translation_other_index_max_eV_A2=float(np.max(np.abs(fc.sum(axis=0)))),
                permutation_max_eV_A2=float(np.max(np.abs(fc-fc.transpose(1,0,3,2)))))

def write_csv(path,header,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(header);writer.writerows(rows)

def plots(out,segments,dos_x,dos_y,summary):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':11,
                         'axes.titlesize':12,'axes.linewidth':.7,'pdf.fonttype':42,
                         'xtick.direction':'out','ytick.direction':'out'})
    color='#16658e'
    fig,(ax,ad)=plt.subplots(1,2,figsize=(12,5.4),sharey=True,
                            gridspec_kw={'width_ratios':[4.5,1],'wspace':.055})
    ticks=[0.0];labels=[r'$\Gamma$'];cursor=0.0
    for item in segments:
        x=item['distance']+cursor
        for branch in range(6):ax.plot(x,item['frequencies'][:,branch],color=color,lw=1.15)
        cursor=float(x[-1]);ticks.append(cursor);labels.append(item['end_label'])
    for x in ticks:ax.axvline(x,color='#d6dce0',lw=.65,zorder=0)
    ax.axhline(0,color='#606b73',lw=.7,ls='--')
    ad.axhline(0,color='#606b73',lw=.7,ls='--')
    ad.plot(dos_y,dos_x,color=color,lw=1.2)
    ad.fill_betweenx(dos_x,0,dos_y,color=color,alpha=.15)
    ax.set(xlim=(0,cursor),xticks=ticks,xticklabels=labels,ylabel='Frequency (THz)',xlabel='Wave vector')
    ad.set(xlim=(0,float(max(dos_y))*1.1),xlabel='DOS\n(modes / atom / THz)')
    ad.tick_params(axis='y',left=False,labelleft=False)
    ax.set_ylim(min(-.15,summary['mesh_min_frequency_THz']-.1),math.ceil((summary['mesh_max_frequency_THz']+.2)*2)/2)
    for a in [ax,ad]:
        a.spines[['top','right']].set_visible(False)
        a.grid(axis='y',color='#e9edf0',lw=.5,zorder=0)
    fig.suptitle(r'$\alpha$-Ti  |  PBEsol',x=.075,y=.97,ha='left',fontsize=17,weight='semibold')
    fig.text(.075,.892,r'550 eV  $\cdot$  $4\times4\times3$ supercell  $\cdot$  electronic $k$: $6\times6\times6$',fontsize=10,color='#485460')
    fig.text(.075,.045,r'Preliminary: cutoff, electronic $k$-mesh and supercell convergence tests pending.',fontsize=9,color='#8a5a16')
    fig.text(.075,.012,r'DOS: $q$ mesh $50^3$; Gaussian $\sigma=0.05$ THz; $M_{\rm Ti}=47.867$ amu; integral = 3 modes/atom.',fontsize=8.5,color='#58636b')
    fig.subplots_adjust(left=.075,right=.98,bottom=.16,top=.84)
    for ext in ['png','pdf']:fig.savefig(out/('PBEsol-phonons-preliminary.'+ext),dpi=240,facecolor='white')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4.6))
    ax.plot(dos_x,dos_y,color=color,lw=1.5)
    ax.fill_between(dos_x,0,dos_y,color=color,alpha=.12)
    ax.set(xlabel='Frequency (THz)',ylabel='DOS (modes / atom / THz)',xlim=(max(-.15,float(dos_x.min())),float(dos_x.max())),ylim=(0,None))
    ax.set_title(r'$\alpha$-Ti  |  PBEsol  |  Preliminary',loc='left',pad=15)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.2)
    fig.text(.13,.03,r'$50^3$ $q$ mesh; Gaussian $\sigma=0.05$ THz. Convergence tests pending.',fontsize=9,color='#8a5a16')
    fig.subplots_adjust(bottom=.17,top=.86,left=.13,right=.97)
    for ext in ['png','pdf']:fig.savefig(out/('PBEsol-DOS-preliminary.'+ext),dpi=240,facecolor='white')
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--inputs',type=Path)
    ap.add_argument('--output',type=Path,default=ROOT/'results/PBEsol-phonons-preliminary-20261031-02')
    args=ap.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    inputs=args.inputs.resolve() if args.inputs else export_inputs(out)
    raw,audits=load_inputs(inputs)
    f=np.stack([raw['force001'],raw['force002']])
    corrected=f-raw['force000'][None,:,:]
    drift=corrected.mean(axis=1)
    corrected-=drift[:,None,:]
    odd=.5*(corrected[0]-corrected[1])
    central=np.stack([odd,-odd])
    np.save(out/'forces_raw.npy',f);np.save(out/'forces_reference_drift_corrected.npy',corrected)
    np.save(out/'forces_central.npy',central)
    p_raw=make_phonon(inputs,f)
    np.save(out/'force_constants_raw.npy',p_raw.force_constants)
    gamma_raw=p_raw.run_qpoints([[0,0,0]]).frequencies[0].tolist()
    p=make_phonon(inputs,central)
    before=p.force_constants.copy()
    p.symmetrize_force_constants(use_symfc_projector=True,show_drift=False)
    projector_residual=fc_residual(p.force_constants)
    projected=p.force_constants.copy()
    # The finite-precision symfc basis left ~3e-9 eV/A^2 translation residual.
    # Polish only this numerical residual with translation/permutation cycles.
    p.symmetrize_force_constants(level=10,use_symfc_projector=False,show_drift=False)
    ASR_polish_change=float(np.max(np.abs(p.force_constants-projected)))
    assert ASR_polish_change<1e-8
    fc=p.force_constants
    assert np.isfinite(fc).all()
    after_residual=fc_residual(fc)
    assert after_residual['translation_max_eV_A2']<1e-9
    assert after_residual['permutation_max_eV_A2']<1e-9
    np.save(out/'force_constants_before_ASR.npy',before)
    np.save(out/'force_constants_ASR.npy',fc)
    p.save(filename=str(out/'phonopy_params.yaml'),settings={'force_constants':True})
    gamma=p.run_qpoints([[0,0,0]]).frequencies[0].tolist()
    mesh=p.run_mesh([50,50,50],is_gamma_center=True)
    frequencies=mesh.frequencies.copy();weights=mesh.weights.copy()
    assert int(weights.sum())==50**3 and np.isfinite(frequencies).all()
    np.savez_compressed(out/'mesh50.npz',qpoints=mesh.qpoints,weights=weights,frequencies_THz=frequencies)
    minf=float(frequencies.min());maxf=float(frequencies.max())
    dos=p.run_total_dos(sigma=.05,freq_min=min(-.3,minf-.3),freq_max=maxf+.3,
                        freq_pitch=.002,use_tetrahedron_method=False,smearing_function='Normal')
    x=dos.frequency_points.copy();y=dos.dos.copy()/len(p.primitive)
    integral=float(np.trapezoid(y,x));normalization=3/integral;y*=normalization
    assert abs(np.trapezoid(y,x)-3)<1e-10 and abs(normalization-1)<1e-5
    write_csv(out/'PBEsol-DOS.csv',['frequency_THz','DOS_modes_per_atom_per_THz'],zip(x,y))
    points={'G':np.array([0.,0,0]),'M':np.array([.5,0,0]),'K':np.array([1/3,1/3,0]),
            'A':np.array([0.,0,.5]),'L':np.array([.5,0,.5]),'H':np.array([1/3,1/3,.5])}
    edges=[('G','M'),('M','K'),('K','G'),('G','A'),('A','L'),('L','H'),('H','A'),('L','M'),('K','H')]
    paths=[np.linspace(points[a],points[b],101) for a,b in edges]
    bs=p.run_band_structure(paths,is_band_connection=False)
    segments=[];rows=[]
    for i,((a,b),q,dist,freq) in enumerate(zip(edges,bs.qpoints,bs.distances,bs.frequencies)):
        segments.append(dict(distance=dist-dist[0],frequencies=freq.copy(),
            end_label=({'G':r'$\Gamma$'}.get(b,b) if i not in [6,7] else ('A|L' if i==6 else 'M|K'))))
        for j in range(len(q)): rows.append([i,a,b,j,*q[j],dist[j]-dist[0],*freq[j]])
    write_csv(out/'PBEsol-bands.csv',['segment','start','end','point','q1','q2','q3','distance_reciprocal_A_no_2pi',*[f'branch{k}_THz' for k in range(1,7)]],rows)
    weighted_negative=int(np.sum(weights[:,None]*(frequencies<-.001)))
    summary=dict(created_utc=datetime.now(timezone.utc).isoformat(),functional='PBEsol',phase='alpha-Ti hcp',
        qualification='PRELIMINARY: ENCUT/kmesh/smearing and supercell convergence unestablished',
        source_jobs={key:a['job_id'] for key,a in audits.items()},encut_eV=550,EDIFF_eV=1e-8,electronic_kmesh=[6,6,6],
        supercell=[4,4,3],atoms_supercell=96,atoms_primitive=2,mass_amu=47.867,
        force_processing='Subtract undisplaced reference; subtract each force-set mean; retain antisymmetric central pair. Phonopy traditional fit, then simultaneous symfc space-group/permutation/translational projector; 10 translation/permutation cycles polish its finite-precision ASR residual (<1e-8 eV/A^2 change).',
        force_drift_removed_eV_A=drift.tolist(),fc_residual_before_ASR=fc_residual(before),fc_residual_after_ASR=after_residual,
        fc_residual_symfc_projector=projector_residual,ASR_numerical_polish_max_change_eV_A2=ASR_polish_change,
        ASR_max_FC_change_eV_A2=float(np.max(np.abs(fc-before))),
        gamma_raw_THz=gamma_raw,gamma_ASR_THz=gamma,
        q_mesh=[50,50,50],q_mesh_gamma_centered=True,q_mesh_irreducible_points=len(weights),
        mesh_min_frequency_THz=minf,mesh_max_frequency_THz=maxf,
        imaginary_threshold_THz=-.001,imaginary_modes_full_mesh_below_threshold=weighted_negative,
        gaussian_sigma_THz=.05,dos_integral_before_numerical_rescale=integral,
        dos_numerical_scale=normalization,dos_integral_modes_per_atom=float(np.trapezoid(y,x)),
        band_path='Gamma-M-K-Gamma-A-L-H-A | L-M | K-H',band_path_reciprocal_coordinates={k:v.tolist() for k,v in points.items()},
        reference_supercell_max_stress_kbar=audits['force000'].get('reference_supercell_max_stress_kbar'),
        source_hashes={str(p.relative_to(inputs)):digest(p) for p in inputs.rglob('*') if p.is_file()},
        software={name:version(name) for name in ['phonopy','numpy','scipy','spglib','symfc','phonors','matplotlib']},
        python=sys.version,platform=platform.platform(),analysis_script_sha256=digest(SCRIPT))
    plots(out,segments,x,y,summary)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    packages=['phonopy','numpy','scipy','spglib','symfc','phonors','matplotlib','PyYAML','h5py']
    (out/'requirements.lock.txt').write_text('\n'.join(name+'=='+version(name) for name in packages)+'\n')
    shutil.copyfile(SCRIPT,out/'reproduce.py')
    (out/'README.txt').write_text('Preliminary PBEsol phonons. No experimental data included.\n'
        'Install requirements.lock.txt into an isolated Python3.12 environment.\n'
        'Reproduce: python reproduce.py --inputs inputs --output ../PBEsol-reproduced\n'
        'DOS integrates to3 modes/atom; negative-frequency Gaussian tails are not imaginary modes.\n'
        'See summary.json for raw imaginary-mode counts, ASR, source hashes and qualification.\n'
        'Cutoff/kmesh/smearing/supercell convergence tests remain pending.\n',encoding='utf-8')
    manifest={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()}
    (out/'manifest-sha256.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['gamma_raw_THz','gamma_ASR_THz','mesh_min_frequency_THz','mesh_max_frequency_THz','imaginary_modes_full_mesh_below_threshold','dos_integral_modes_per_atom','ASR_max_FC_change_eV_A2']},indent=2),flush=True)
    print('Saved:',out,flush=True)

if __name__=='__main__':main()
