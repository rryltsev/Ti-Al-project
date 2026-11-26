"""Eigenvector/subspace comparisons and small-q acoustic convergence on frozen FCs."""
from pathlib import Path
import argparse,json,os
import numpy as np
from scipy.optimize import linear_sum_assignment
import phonopy
def analyze(folder):
    result={}
    for xc in ['PBE','PBEsol','r2SCAN']:
        root=folder/xc;ref=np.load(root/'DFT/bands.npz');pred=np.load(root/'original/bands.npz')
        assert np.array_equal(ref['qpoints'],pred['qpoints'])
        df=ref['frequencies'];mf=pred['frequencies'];assignments=[];fidelities=[];aligned=[];isolated=[];degenerate=[]
        for i in range(len(df)):
            overlap=np.abs(ref['eigenvectors'][i].conj().T@pred['eigenvectors'][i])**2
            ii,jj=linear_sum_assignment(-overlap);assert np.array_equal(ii,np.arange(6));assignments.append(jj);aligned.append(mf[i,jj])
            groups=[]
            for j in range(6):
                if groups and abs(df[i,j]-df[i,groups[-1][-1]])<1e-4:groups[-1].append(j)
                else:groups.append([j])
            for g in groups:
                fidelity=float(overlap[np.ix_(g,jj[g])].sum()/len(g));fidelities.append(fidelity)
                (isolated if len(g)==1 else degenerate).append(fidelity)
        aligned=np.array(aligned);assignments=np.array(assignments)
        np.savez_compressed(root/'mode_matching.npz',DFT_frequencies=df,original_frequencies_matched=aligned,original_indices=assignments,qpoints=ref['qpoints'])
        acoustics={};gamma={};objects={}
        for name in ['DFT','original']:
            p=phonopy.load(root/name/'phonopy_params.yaml',produce_fc=False,symmetrize_fc=False,log_level=0);objects[name]=p
            g=p.run_qpoints([[0,0,0]],with_eigenvectors=True);freq=g.frequencies[0];pol=np.sum(np.abs(g.eigenvectors[0][2::3])**2,axis=0)
            singlet=3+int(np.argmax(pol[3:]));doublet=[j for j in range(3,6) if j!=singlet]
            gamma[name]=dict(optical_singlet_c_THz=float(freq[singlet]),optical_doublet_basal_THz=freq[doublet].tolist(),c_polarization_singlet=float(pol[singlet]),singlet_below_doublet=bool(freq[singlet]<min(freq[doublet])))
            directions={'Gamma-M':[.5,0,0],'Gamma-K':[1/3,1/3,0],'Gamma-A':[0,0,.5]};acoustics[name]={}
            for direction,d in directions.items():
                eps=np.array([.0005,.001,.002,.004]);qs=eps[:,None]*np.array(d)[None,:]
                f=p.run_qpoints(qs).frequencies[:,:3]
                # f is cycles/ps and reciprocal distances are cycles/Angstrom: f/k gives Angstrom/ps.
                kr=np.linalg.norm(qs@np.linalg.inv(p.primitive.cell).T,axis=1)
                speed=f/kr[:,None]*100
                acoustics[name][direction]=dict(q_fraction_scales=eps.tolist(),q_reciprocal_A=kr.tolist(),frequencies_THz=f.tolist(),phase_velocity_m_s=speed.tolist(),relative_change_two_smallest_q=(np.abs(speed[0]-speed[1])/np.maximum(np.abs(speed[1]),1)).tolist())
        ratio=(gamma['original']['optical_singlet_c_THz']/gamma['DFT']['optical_singlet_c_THz'])**2
        dmat={}
        for name,p in objects.items():
            matrices=[]
            for q in ref['qpoints']:
                p.dynamical_matrix.run(q);matrices.append(p.dynamical_matrix.dynamical_matrix.copy())
            dmat[name]=np.array(matrices)
        result[xc]=dict(Gamma=gamma,singlet_eigenvalue_ratio_original_over_DFT=ratio,acoustics=acoustics,matched_frequency_RMSE_THz=float(np.sqrt(np.mean((aligned-df)**2))),sorted_frequency_RMSE_THz=float(np.sqrt(np.mean((mf-df)**2))),subspace_fidelity_min=float(min(fidelities)),isolated_mode_overlap_min=float(min(isolated)),degenerate_subspace_fidelity_min=float(min(degenerate)),relative_dynamical_matrix_Frobenius_error=float(np.linalg.norm(dmat['original']-dmat['DFT'])/np.linalg.norm(dmat['DFT'])),interpretation='Per-q maximum total squared eigenvector overlap; frequency-degenerate reference modes (1e-4 THz) evaluated using subspace projectors. This is local correspondence, not branch continuation along a path. Acoustic finite-q convergence pertains only to these fixed force constants, not to supercell/DFT convergence.')
    (folder/'mode_diagnostics.json').write_text(json.dumps(dict(job_id=os.environ.get('SLURM_JOB_ID'),results=result),indent=2)+'\n')
    print(json.dumps({xc:dict(Gamma=v['Gamma'],matched_RMSE=v['matched_frequency_RMSE_THz'],projector_fidelity_min=v['degenerate_subspace_fidelity_min']) for xc,v in result.items()},indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('folder',type=Path);a=ap.parse_args();analyze(a.folder)
