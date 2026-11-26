"""Fixed-geometry harmonic baseline, using exactly the archived DFT processing."""
from pathlib import Path
import hashlib,json,os
import numpy as np
import build_pbesol_phonons as b

ROOT=Path(__file__).resolve().parent
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def process(inputs,raw,out,qpoints):
    out.mkdir(parents=True)
    f=np.stack([raw['force001'],raw['force002']]);corrected=f-f.mean(axis=1)[:,None,:]
    odd=(corrected[0]-corrected[1])/2;central=np.stack([odd,-odd])
    rawp=b.make_phonon(inputs,f);np.save(out/'force_constants_raw.npy',rawp.force_constants)
    p=b.make_phonon(inputs,central);before=p.force_constants.copy();np.save(out/'force_constants_central.npy',before)
    p.symmetrize_force_constants(use_symfc_projector=True,show_drift=False);projected=p.force_constants.copy()
    p.symmetrize_force_constants(level=10,use_symfc_projector=False,show_drift=False)
    residual=b.fc_residual(p.force_constants);assert max(residual.values())<1e-9
    assert np.max(np.abs(projected-p.force_constants))<1e-8
    np.save(out/'force_constants_ASR.npy',p.force_constants);np.save(out/'forces_central.npy',central)
    p.save(filename=str(out/'phonopy_params.yaml'),settings={'force_constants':True})
    bands=p.run_qpoints(qpoints,with_eigenvectors=True)
    frequencies=bands.frequencies.copy();vectors=bands.eigenvectors.copy()
    np.savez_compressed(out/'bands.npz',qpoints=qpoints,frequencies=frequencies,eigenvectors=vectors)
    gamma=p.run_qpoints([[0,0,0]],with_eigenvectors=True)
    gf=gamma.frequencies[0].copy();gv=gamma.eigenvectors[0].copy()
    # Polarization along crystallographic c identifies longitudinal singlet vs basal doublet.
    pol=np.sum(np.abs(gv[2::3,:])**2,axis=0)
    mesh=p.run_mesh([50,50,50],is_gamma_center=True);assert mesh.weights.sum()==125000
    np.savez_compressed(out/'mesh50.npz',qpoints=mesh.qpoints,weights=mesh.weights,frequencies=mesh.frequencies)
    dd=p.run_total_dos(sigma=.05,freq_min=min(-.5,float(mesh.frequencies.min())-.4),freq_max=float(mesh.frequencies.max())+.4,freq_pitch=.002,use_tetrahedron_method=False,smearing_function='Normal')
    area=float(np.trapezoid(dd.dos/2,dd.frequency_points));assert abs(area-3)<1e-5
    np.savez_compressed(out/'dos.npz',frequency=dd.frequency_points,dos=dd.dos/2)
    return p,frequencies,dict(gamma_THz=gf.tolist(),gamma_c_axis_polarization=pol.tolist(),mesh_min_THz=float(mesh.frequencies.min()),imaginary_modes_weighted=int(np.sum(mesh.weights[:,None]*(mesh.frequencies<-.001))),DOS_integral_modes_per_atom=area,ASR_residual=residual,ASR_max_change_eV_A2=float(np.max(np.abs(p.force_constants-before))),force_drift_eV_A=f.mean(axis=1).tolist(),reference_max_force_eV_A=float(np.linalg.norm(raw['force000'],axis=1).max()) if 'force000' in raw else None)
def main():
    assert os.environ.get('SLURM_JOB_ID')
    out=ROOT/'harmonics';out.mkdir(exist_ok=False)
    reference=np.load(ROOT/'reference_figure.npz');summary={}
    for xc in ['PBE','PBEsol','r2SCAN']:
        inp=ROOT/'harmonic_inputs'/xc
        dft,_=b.load_inputs(inp,keys=['force001','force002'],functional=xc)
        ml={key:np.load(ROOT/'baseline/harmonic_forces'/xc/key/'forces.npy') for key in ['force000','force001','force002']}
        p,df,ds=process(inp,dft,out/xc/'DFT',reference['qpoints'])
        m,mf,ms=process(inp,ml,out/xc/'original',reference['qpoints'])
        reproduction=float(np.max(np.abs(df-reference[xc+'_DFT_bands'])))
        assert reproduction<1e-5,(xc,reproduction)
        summary[xc]=dict(DFT=ds,original=ms,DFT_archived_bands_reproduction_max_THz=reproduction,sorted_band_RMSE_THz=float(np.sqrt(np.mean((mf-df)**2))),sorted_band_max_error_THz=float(np.max(np.abs(mf-df))),relative_FC_Frobenius_error=float(np.linalg.norm(m.force_constants-p.force_constants)/np.linalg.norm(p.force_constants)))
        print(xc,summary[xc],flush=True)
    dump(out/'summary.json',dict(status='PASS',job_id=os.environ['SLURM_JOB_ID'],processing='Same XC-specific DFT cell, +/-0.01 A, central pair, mass47.867, 4x4x3, same symfc/ASR, mesh50^3. No ML relaxation.',qualification='H0 establishes a matched baseline. DFT convergence and accuracy are not certified. Harmonics are physical development data when used for method choice.',results=summary))
if __name__=='__main__':main()
