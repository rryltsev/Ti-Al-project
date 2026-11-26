"""Matched harmonic diagnostics from saved force constants; no new ML/DFT forces."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import numpy as np
import phonopy
from scipy.optimize import linear_sum_assignment


def mode_comparison(ref, pred):
    assert np.array_equal(ref['qpoints'], pred['qpoints'])
    df, mf = ref['frequencies'], pred['frequencies']
    matched, assignments, isolated, degenerate = [], [], [], []
    dmat_norm, diff_norm = 0., 0.
    for i in range(len(df)):
        dv, mv = ref['eigenvectors'][i], pred['eigenvectors'][i]
        overlap = np.abs(dv.conj().T @ mv)**2
        ii, jj = linear_sum_assignment(-overlap)
        assert np.array_equal(ii, np.arange(6))
        matched.append(mf[i, jj]); assignments.append(jj)
        groups = []
        for j in range(6):
            if groups and abs(df[i, j] - df[i, groups[-1][-1]]) < 1e-4:
                groups[-1].append(j)
            else:
                groups.append([j])
        for g in groups:
            fidelity = float(overlap[np.ix_(g, jj[g])].sum() / len(g))
            (isolated if len(g) == 1 else degenerate).append(fidelity)
        # Both models use identical masses and frequency conversion. The common
        # conversion cancels in this relative Frobenius norm, including imaginary modes.
        dd = (dv * (np.sign(df[i]) * df[i]**2)) @ dv.conj().T
        md = (mv * (np.sign(mf[i]) * mf[i]**2)) @ mv.conj().T
        dmat_norm += np.linalg.norm(dd)**2
        diff_norm += np.linalg.norm(md-dd)**2
    return dict(sorted_RMSE_THz=float(np.sqrt(np.mean((mf-df)**2))),
        overlap_matched_RMSE_THz=float(np.sqrt(np.mean((np.array(matched)-df)**2))),
        isolated_overlap_min=float(min(isolated)),
        degenerate_projector_fidelity_min=float(min(degenerate)),
        relative_dynamical_matrix_Frobenius_error=float(np.sqrt(diff_norm/dmat_norm))), np.array(assignments)


def inspect(folder):
    p = phonopy.load(folder/'phonopy_params.yaml', produce_fc=False, symmetrize_fc=False, log_level=0)
    g = p.run_qpoints([[0, 0, 0]], with_eigenvectors=True)
    freq = g.frequencies[0]; pol = np.sum(np.abs(g.eigenvectors[0][2::3])**2, axis=0)
    singlet = 3 + int(np.argmax(pol[3:])); doublet = [j for j in range(3, 6) if j != singlet]
    gamma = dict(singlet_c_THz=float(freq[singlet]), doublet_basal_THz=freq[doublet].tolist(),
        c_polarization_singlet=float(pol[singlet]), singlet_below_doublet=bool(freq[singlet]<min(freq[doublet])))
    acoustics = {}
    for name, direction in {'Gamma-M':[.5,0,0], 'Gamma-K':[1/3,1/3,0], 'Gamma-A':[0,0,.5]}.items():
        eps = np.array([.0005, .001, .002, .004]); q = eps[:,None]*np.array(direction)
        f = p.run_qpoints(q).frequencies[:,:3]
        k = np.linalg.norm(q@np.linalg.inv(p.primitive.cell).T, axis=1)
        v = f/k[:,None]*100
        acoustics[name] = dict(q_fraction_scales=eps.tolist(), phase_velocity_m_s=v.tolist(),
            relative_change_two_smallest_q=(np.abs(v[0]-v[1])/np.maximum(np.abs(v[1]),1)).tolist())
    projection = {}
    for name in ['raw','central','ASR']:
        p.force_constants = np.load(folder/f'force_constants_{name}.npy')
        mesh = p.run_mesh([50,50,50], is_gamma_center=True)
        projection[name] = dict(mesh_min_THz=float(mesh.frequencies.min()),
            imaginary_modes_weighted=int(np.sum(mesh.weights[:,None]*(mesh.frequencies<-.001))))
    return dict(Gamma=gamma, acoustics=acoustics, projection=projection)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Run the Phonopy diagnostic in Slurm.'
    a.output.mkdir(exist_ok=False)
    results, hashes = {}, {}
    for xc in ['PBE','r2SCAN']:
        folders = {'DFT':a.root/'harmonics'/xc/'DFT', 'original':a.root/'harmonics'/xc/'original'}
        folders.update({method:a.root/'training-plan01/cases'/f'{xc}-{method}'/'evaluation/harmonics'
                        for method in ['lora16','full']})
        ref=np.load(folders['DFT']/'bands.npz'); ref_fc=np.load(folders['DFT']/'force_constants_ASR.npy')
        ref_dos=np.load(folders['DFT']/'dos.npz'); results[xc]={}
        for method, folder in folders.items():
            for fname in ['bands.npz','dos.npz','phonopy_params.yaml','force_constants_raw.npy',
                          'force_constants_central.npy','force_constants_ASR.npy']:
                path=folder/fname; hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            r=inspect(folder)
            if method!='DFT':
                bands=np.load(folder/'bands.npz'); m, indices=mode_comparison(ref,bands)
                np.savez_compressed(a.output/f'{xc}-{method}-mode-matching.npz', indices=indices,
                                    qpoints=ref['qpoints'], frequencies_matched=bands['frequencies'][np.arange(len(indices))[:,None],indices])
                r.update(m)
                r['relative_FC_Frobenius_error']=float(np.linalg.norm(np.load(folder/'force_constants_ASR.npy')-ref_fc)/np.linalg.norm(ref_fc))
                dos=np.load(folder/'dos.npz')
                grid=np.arange(min(ref_dos['frequency'].min(),dos['frequency'].min()),
                               max(ref_dos['frequency'].max(),dos['frequency'].max())+.002, .002)
                delta=np.interp(grid,dos['frequency'],dos['dos'],left=0,right=0)-np.interp(grid,ref_dos['frequency'],ref_dos['dos'],left=0,right=0)
                r['DOS_L1_modes_per_atom']=float(np.trapezoid(np.abs(delta),grid))
                r['singlet_eigenvalue_ratio_to_DFT']=(r['Gamma']['singlet_c_THz']/results[xc]['DFT']['Gamma']['singlet_c_THz'])**2
            results[xc][method]=r
    output=dict(status='PASS',job_id=os.environ['SLURM_JOB_ID'],results=results,input_sha256=hashes,
        interpretation='Per-q eigenvector correspondence, with degenerate reference projectors (1e-4 THz); not continuous branch tracking. Acoustic q convergence applies only to fixed FCs, not supercell/DFT convergence. Raw projection diagnostics include uncorrected numerical drift.')
    (a.output/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',job_id=output['job_id'])))


if __name__=='__main__':
    main()
