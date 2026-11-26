"""Integrate the verified six-panel phonon figure without changing other results."""
from pathlib import Path
import hashlib, json, re, shutil, zipfile
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
MAN=ROOT/'manuscript'
OUT=Path(__file__).resolve().parent
SRC=ROOT/'phonons/results/figure2-six-panels-20261101-02'
BACKUP=OUT/'before_revision.zip'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def rep(s,a,b):
    assert s.count(a)==1,(a[:100],s.count(a))
    return s.replace(a,b)
def write(n,s): (MAN/n).write_text(s,encoding='utf-8')

if not BACKUP.exists():
    with zipfile.ZipFile(BACKUP,'w',zipfile.ZIP_DEFLATED) as z:
        for p in MAN.rglob('*'):
            rel=p.relative_to(MAN)
            if p.is_file() and rel.parts[0]!='build': z.write(p,str(rel))
with zipfile.ZipFile(BACKUP) as z:
    baseline={n:z.read(n) for n in z.namelist()}
def old(n): return baseline[n.replace('/','\\') if n not in baseline else n].decode('utf-8-sig').replace('\r\n','\n')

manifest=read(SRC/'manifest-sha256.json')
for name,h in manifest.items(): assert sha(SRC/name)==h,name
assert sha(SRC/'figure2_phonons_dft_dpa4.pdf')==sha(MAN/'figures/variants/figure2_phonons_dft_dpa4_20261101.pdf')
a=np.load(SRC/'figure-data.npz')
summary=read(SRC/'summary.json')
assert summary['configuration']['checkpoints']=={'PBE':15000,'PBEsol':23100,'r2SCAN':23200}
assert a['experiment'].shape==(77,2) and a['qpoints'].shape==(604,3)
checked={}
for xc in ['PBE','PBEsol','r2SCAN']:
    d=a[xc+'_DFT_bands']; m=a[xc+'_DPA4_bands']
    err=float(np.sqrt(np.mean((m-d)**2)))
    assert abs(err-summary['results'][xc]['sorted_band_RMSE_THz'])<1e-12
    checked[xc]={'sorted_band_RMSE_THz':err,'Gamma_DFT_THz':d[0].tolist(),'Gamma_DPA4_THz':m[0].tolist()}
    for kind in ['DFT','DPA4']:
        f=a['frequency_THz']; g=a[xc+'_'+kind+'_DOS']; integral=float(np.trapezoid(g,f))
        assert abs(integral-3)<1e-12
        region=np.flatnonzero((f>7)&(f<8)); peak=region[np.argmax(g[region])]
        checked[xc][kind+'_DOS']={'integral':integral,'peak_7_8_THz':float(f[peak]),'peak_height':float(g[peak])}

shutil.copyfile(SRC/'figure2_phonons_dft_dpa4.pdf',MAN/'figures/phonons_alpha_ti.pdf')
data=MAN/'supporting_data/phonons_matched_20261101'
data.mkdir(exist_ok=True)
for name in ['summary.json','verification.json','bands.csv','DOS.csv','experiment.csv','figure-data.npz','manifest-sha256.json','PBE15k-diagnostics.json','requirements.lock.txt']:
    shutil.copyfile(SRC/name,data/name)
(data/'verified_metrics.json').write_text(json.dumps(checked,indent=2)+'\n',encoding='utf-8')

s=old('sections/methods.tex')
s=rep(s,'while it gives the closest integral agreement with the PBE DFT phonon density of states among the five saved checkpoints.',
      'while it gives the closest integral agreement with the archived PBE DFT phonon density of states among the five saved checkpoints.')
s=rep(s,'The DFT phonon diagnostic forms part of selection, whereas the main-text lattice-dynamics comparison uses experimental dispersions.',
      'The archived PBE DFT phonon diagnostic forms part of selection. The main-text lattice-dynamics comparison uses new, same-functional DFT references and matched DFT/DPA4 calculation settings, together with experimental dispersions.')
start=s.index(r'\subsection{Harmonic lattice dynamics}')
end=s.index(r'\subsection{Solid--liquid coexistence and melting of Ti}',start)
s=s[:start]+r'''\subsection{Harmonic lattice dynamics}
\label{sec:phonons}

For $\alpha$-Ti, direct DFT and fine-tuned DPA4 phonons were evaluated with a common finite-displacement protocol in Phonopy.\cite{Togo2015}
The hcp unit cell was relaxed separately with PBE, PBEsol, and \rscan{}; within each functional, DFT and DPA4 forces were then evaluated on exactly the same displaced structures, without a separate DPA4 relaxation.
The force-constant supercell was $4\times4\times3$ (96 atoms), with symmetry-generated opposite displacements of $\pm0.01$~\AA{} and a Ti mass of 47.867~amu.
The DFT force calculations used VASP~6.6.1, the Ti\_pv PAW dataset, a 550~eV cutoff, a $\Gamma$-centered $6\times6\times6$ electronic grid, Gaussian smearing of width 0.1~eV, and an electronic convergence criterion of $10^{-8}$~eV.
Real-space projection was disabled and nonspherical PAW contributions were included.
For both methods, mean force drift was removed and the central difference of the opposite-displacement forces was used to construct force constants, followed by identical symmetry projection and enforcement of the acoustic sum rule.
Thus, differences within a DFT/DPA4 pair probe the model's harmonic response at fixed geometry and numerical phonon settings; differences between the three DFT spectra also include the functional dependence of the equilibrium cell.

Dispersions were evaluated along $\Gamma$--K--M--$\Gamma$--A and compared with 77 digitized room-temperature experimental points reproduced in Ref.~\onlinecite{Souvatzis2007}.
The density of states (DOS) used a $\Gamma$-centered $50^3$ phonon mesh, Gaussian broadening $\sigma=0.05$~THz, and a 0.002~THz frequency spacing; each DOS integrates to three modes per atom.
The calculations are harmonic spectra at static relaxed geometries and exclude thermal expansion and anharmonic frequency renormalization.
DFT cutoff, electronic-grid, smearing, and supercell convergence are not yet established; the current spectra and their comparisons are conditional on the stated settings.
Reference-force checks and the cancellation of common reference forces in the central difference are documented in the Supporting Materials, together with the earlier diagnostics used for model selection.

The retained $\alpha_2$-Ti$_3$Al model calculations use ASE with 1000-atom supercells.
Their BFGS relaxation preserves crystal symmetry and allows hydrostatic cell strain, with a force convergence target of $10^{-4}$~eV\,\AA$^{-1}$.
Positive and negative Cartesian displacements of 0.01~\AA{} were used to construct force constants, followed by enforcement of the acoustic sum rule.
No corresponding DFT phonon benchmark is included for Ti$_3$Al.

'''+s[end:]
write('sections/methods.tex',s)

s=old('sections/results.tex')
start=s.index('The harmonic phonon dispersions of')
end=s.index('The temperature-dependent density of elemental Ti',start)
s=s[:start]+r'''Figure~\ref{fig:phonons} compares the harmonic response of $\alpha$-Ti at two levels.
The left column shows direct DFT and corresponding fine-tuned DPA4 dispersions against experiment; the right column compares their DOS.
Because the two calculations within each functional use the same DFT-relaxed geometry and phonon protocol, DFT--experiment discrepancies can be assessed separately from the additional DPA4--DFT mismatch.
All six spectra have no frequencies below $-10^{-3}$~THz on the sampled path or phonon mesh, consistent with harmonic stability within this sampling.

The direct DFT curves show that the functional ranking depends on the branch and wave vector.
Near $\Gamma$, the experimental optical features lie at approximately 4.1 and 5.5~THz.
PBE gives an optical doublet at 3.964~THz and a nondegenerate mode at 5.408~THz, closer to these features than the corresponding PBEsol values of 3.786 and 5.758~THz or the \rscan{} values of 4.587 and 5.189~THz.
The higher \rscan{} optical branches around K also lie above the experimental points near 7~THz.
These discrepancies are already present in the direct electronic-structure calculations and therefore cannot be attributed to fine-tuning.
The direct comparison does not support a uniform improvement of the experimental dispersion on replacing a GGA with \rscan{}.
This is a comparison of static harmonic predictions with room-temperature measurements; it is not a separation of exchange-correlation error from thermal and numerical effects.

\begin{figure*}[t]
\includegraphics[width=\textwidth]{figures/phonons_alpha_ti.pdf}
\caption{Harmonic dispersions (left) and densities of states (right) of $\alpha$-Ti: PBE (a,b), PBEsol (c,d), and \rscan{} (e,f). Black points are the same 77 digitized room-temperature measurements reproduced in Ref.~\onlinecite{Souvatzis2007}, attributed there to Stassis \textit{et al.} Blue dashed curves denote direct DFT; orange solid curves denote the corresponding DPA4-finetuned model. Each pair uses the same functional-specific DFT-relaxed cell, $4\times4\times3$ supercell, $\pm0.01$~\AA{} displacements, and Phonopy processing. DOS curves use a $\Gamma$-centered $50^3$ mesh and $\sigma=0.05$~THz broadening, with three modes per atom. The calculations use the numerical settings in Sec.~\ref{sec:phonons}; DFT convergence tests remain pending.}
\label{fig:phonons}
\end{figure*}

The DPA4 curves broadly reproduce the DFT dispersion shapes, but leave systematic optical-mode errors.
The nondegenerate optical frequency at $\Gamma$ decreases from 5.408 to 4.903~THz for PBE, from 5.758 to 4.954~THz for PBEsol, and from 5.189 to 4.221~THz for \rscan{}.
The last change reverses its ordering relative to the optical doublet, which remains close to DFT (4.563 versus 4.587~THz).
Thus, the rearrangement of the \rscan{} optical modes in the fine-tuned model is mainly a representation error of that nondegenerate mode, rather than an intrinsic prediction of the reference functional.
The root-mean-square DPA4--DFT differences of frequency-ordered eigenvalues along the plotted path are 0.209, 0.278, and 0.259~THz for PBE, PBEsol, and \rscan{}, respectively.
These are descriptive spectral discrepancies at the sampled wave vectors, without polarization tracking through crossings; they are not errors relative to experiment.

The DOS panels expose additional differences in the distribution of modes even where the dispersion curves appear close.
For example, the principal PBEsol peak between 7 and 8~THz shifts from 7.194~THz in DFT to 7.328~THz in DPA4, while its height increases from 0.782 to 1.007~modes/(atom\,THz).
The corresponding \rscan{} peak shifts from 7.538 to 7.744~THz.
These changes persist with identical geometry, sampling, broadening, and normalization, and therefore reflect differences in the calculated force constants rather than plotting conventions.
Agreement with experiment alone would not expose all such errors: a model shift can either increase or partially cancel the DFT--experiment discrepancy.

The retained $\alpha_2$-Ti$_3$Al model calculations likewise have no appreciable imaginary frequencies along the sampled $\Gamma$--M--K--$\Gamma$--A--L--H--A path; the largest residual negative magnitude is below $10^{-3}$~THz.
Their maximum frequencies are 9.14, 9.48, and 9.58~THz for PBE, PBEsol, and \rscan{}, respectively.
These results characterize model differences for the compound; the matched DFT/DPA4 and experimental comparisons in Fig.~\ref{fig:phonons} concern elemental $\alpha$-Ti.
Neither the harmonic calculations nor the room-temperature comparison determine the temperature-renormalized phonon spectrum near melting.

'''+s[end:]
write('sections/results.tex',s)

s=old('sections/discussion.tex')
needle=r'\subsection{Consequences for validation and transfer learning}'
s=rep(s,needle,r'''\subsection{Separating reference and representation errors}

The paired phonon calculations make this distinction explicit for the local curvature of the $\alpha$-Ti potential-energy surface.
Direct DFT--experiment comparisons assess the three electronic-structure descriptions without an intervening learned potential.
The corresponding DPA4--DFT comparisons, at identical geometry and phonon settings within each functional, test the fidelity of the final fine-tuned models to those references.
Near $\Gamma$, the closer agreement of PBE DFT with the experimental optical frequencies is a property of the reference calculation, whereas the downward shifts of the nondegenerate optical mode in all three DPA4 models identify an additional model discrepancy.
In particular, the reversed optical-mode ordering for the \rscan{} model is absent from its DFT reference.

For an identifiable mode at a given wave vector and the same calculated geometry, the model--experiment frequency residual is the sum of the model--DFT and DFT--experiment residuals.
The two contributions can reinforce or cancel one another.
Consequently, closer agreement of a learned potential with an experimental point does not necessarily imply more faithful reproduction of its training functional.
The DOS comparison provides a complementary test of that fidelity over the Brillouin zone, showing residual changes in peak positions and spectral weight at common broadening and normalization.

This separates the two levels of validation without making either error source exact.
The DFT--experiment residual includes the harmonic approximation, the use of a static equilibrium cell, and unresolved numerical convergence, in addition to the electronic-structure approximation.
The model--DFT residual tests the final fine-tuned representation under the phonon protocol, which differs from the training-label protocol in electronic sampling and smearing.
It does not isolate the optimizer's contribution or establish the improvement over the pretrained model without a corresponding pretrained phonon calculation.
The remaining optical errors therefore motivate more targeted validation of force derivatives near the crystal, even when pooled energy and force errors are small.

'''+needle)
s=rep(s,'A quantitative attribution to exchange-correlation physics must still separate functional effects from potential-fitting and sampling errors.',
      'Beyond the matched harmonic comparison, a quantitative attribution of thermodynamic changes to exchange-correlation physics must still separate functional effects from potential-fitting and sampling errors.')
write('sections/discussion.tex',s)

s=old('sections/abstract.tex')
s=rep(s,'Common DFT labeling and fine-tuning settings support comparisons across lattice dynamics, density, melting, liquid mixing thermodynamics, and viscosity.',
      'Common DFT labeling and fine-tuning settings support comparisons across lattice dynamics, density, melting, liquid mixing thermodynamics, and viscosity. Matched direct-DFT and DPA4 phonon calculations distinguish discrepancies with experiment in the reference description from residual errors in the fine-tuned harmonic response.')
write('sections/abstract.tex',s)
s=old('sections/conclusions.tex')
needle='The thermodynamic results are consistent with improved relative energies of selected environments under \\rscan{}.'
s=rep(s,needle,r'''The matched $\alpha$-Ti phonon benchmark provides a direct distinction between the DFT reference and its learned representation.
PBE DFT is closer to the experimental optical features near $\Gamma$, while no functional improves every branch uniformly.
The fine-tuned models retain DPA4--DFT spectral RMSEs of 0.209--0.278~THz and systematically underestimate the nondegenerate optical frequency at $\Gamma$.
For \rscan{}, this reverses the optical-mode ordering relative to DFT.
These discrepancies show why experimental agreement and fidelity to the reference functional require separate validation, subject to the remaining DFT convergence and finite-temperature qualifications.

'''+needle)
write('sections/conclusions.tex',s)

s=old('supporting_materials.tex')
s=rep(s,'Draft -- 29 October 2026','Draft -- 1 November 2026')
s=rep(s,'The present revision additionally incorporates the 60~ps r2SCAN enthalpy series and three PBE 15k alloy-viscosity estimates with 15 replicas each.',
      'The supporting data also include the 60~ps r2SCAN enthalpy series and three PBE 15k alloy-viscosity estimates with 15 replicas each. The new matched DFT/DPA4 phonon comparison for all three functionals is documented in Sec.~\\ref{sec:matched_phonons}; the earlier checkpoint-selection diagnostics below retain their original protocols.')
s=rep(s,r'\label{sec:phonons}',r'''\label{sec:phonons}
This section documents the earlier ASE checkpoint-selection study. Its independently optimized cells and supercells differ from the matched Phonopy calculations used in the current main-text Fig.~2 (Sec.~\ref{sec:matched_phonons}).''')
s=rep(s,r'\section{Comparison with the PBE DFT phonon DOS}',r'''\section{Comparison with the archived PBE DFT phonon DOS}
This archived PBE-only comparison was used for checkpoint selection. It is separate from the new three-functional DFT references in main-text Fig.~2 and Sec.~\ref{sec:matched_phonons}.''')
s=rep(s,'The PBE DFT DOS is not a same-functional reference for PBEsol or \\rscan{}.',
      'This archived PBE DFT DOS is not a same-functional reference for PBEsol or \\rscan{}; the new matched comparison supplies a separate DFT reference for each functional.')
s=rep(s,'PBE checkpoint coverage in the 29 October revision.','PBE checkpoint coverage in the current revision.')
needle=r'\section{Implications and reproducibility}'
s=rep(s,needle,r'''\section{Matched DFT and DPA4 phonons for the three functionals}
\label{sec:matched_phonons}

\subsection{Common geometries and numerical processing}
Main-text Fig.~2 uses new same-functional DFT references for PBE, PBEsol, and \rscan{}, with DPA4 evaluated on exactly the same displaced structures.
The models are PBE 15k, PBEsol 23.1k, and \rscan{} 23.2k. The PBE calculation was rerun explicitly with 15k for this figure; the earlier PBE 5k matched calculation is not used.
The hcp cell is relaxed independently for each functional and then held fixed for its DFT/DPA4 pair. A $4\times4\times3$ supercell contains 96 atoms; the symmetry-generated opposite displacements have magnitude 0.01~\AA.
DFT force calculations use VASP~6.6.1, Ti\_pv, a 550~eV cutoff, a $\Gamma$-centered $6^3$ electronic grid, Gaussian smearing of width 0.1~eV, and $10^{-8}$~eV electronic convergence. Real-space projection is disabled.
These settings differ from those of the archived PBE-only reference used in the checkpoint study.

For each displaced configuration, the mean force vector is subtracted to remove net drift.
The odd component of the opposite-displacement forces, $[F(+\delta)-F(-\delta)]/2$, defines the central pair supplied to the traditional Phonopy force-constant fit.
Both methods use the same symmetry projection with the \texttt{symfc} projector, followed by ten acoustic-sum-rule polishing cycles.
The saved force constants, geometries, and plotting arrays are retained with SHA-256 manifests.
An undisplaced reference force cancels identically in this construction:
\begin{equation}
 [F(+\delta)-F_0]-[F(-\delta)-F_0]
 = F(+\delta)-F(-\delta).
\end{equation}
Consequently, the missing undisplaced-force calculations for PBE and \rscan{} do not require an assumed zero force to construct the current spectra.
They are still needed to diagnose residual force and stress in the reference cells; an adverse diagnostic could require a new geometry or displaced-force calculation.

The plotted path contains four segments, $\Gamma$--K--M--$\Gamma$--A, with 151 points per segment and repeated segment boundaries (604 entries).
Each segment is mapped linearly to the previously digitized path ticks; none of the 77 experimental coordinates or frequencies is changed and no frequency shift is fitted.
The room-temperature points are those reproduced by Souvatzis et al. and attributed there to Stassis et al., as cited in the main text. They have neither branch labels nor uncertainty estimates in the digitized data.
For this reason, no branch-resolved or uncertainty-weighted experimental RMSE is inferred from them.
The DOS uses a $\Gamma$-centered $50^3$ mesh, Ti mass 47.867~amu, Gaussian width 0.05~THz, and frequency step 0.002~THz; each integral is three modes per atom.

\subsection{Fidelity of the fine-tuned harmonic response}
Table~\ref{tab:matched_phonons} lists the optical frequencies at $\Gamma$ and the descriptive DPA4--DFT spectral RMSE along the exact plotted path.
For the RMSE, frequencies are sorted independently at each wave vector, without tracking eigenvectors across crossings, and every one of the 604 path entries and six frequencies receives equal weight.
Thus, this metric is neither a Brillouin-zone average nor a comparison to experiment, and it can pair modes of different character when their ordering changes.
The optical doublet and nondegenerate mode are reported separately to expose that ordering change for \rscan{}.

\begin{table}[htbp]
\centering
\caption{Optical frequencies at $\Gamma$ and DPA4--DFT spectral RMSE in THz for main-text Fig.~2. The doublet entries give the common frequency of two degenerate modes. DFT and DPA4 use the same geometry within each functional.}
\label{tab:matched_phonons}
\begin{tabular}{lrrrrr}
\toprule
 & \multicolumn{2}{c}{Optical doublet} & \multicolumn{2}{c}{Nondegenerate mode} & Path RMSE\\
Functional & DFT & DPA4 & DFT & DPA4 & DPA4--DFT\\
\midrule
PBE & 3.964 & 3.800 & 5.408 & 4.903 & 0.209\\
PBEsol & 3.786 & 3.765 & 5.758 & 4.954 & 0.278\\
\rscan{} & 4.587 & 4.563 & 5.189 & 4.221 & 0.259\\
\bottomrule
\end{tabular}
\end{table}

The nondegenerate optical mode is lower than DFT by 0.505, 0.803, and 0.969~THz for PBE, PBEsol, and \rscan{}, respectively.
For \rscan{}, this mode falls below the doublet in DPA4 even though it lies above it in DFT; the doublet itself changes by only $-0.023$~THz.
The doublet/nondegenerate distinction follows their degeneracies at $\Gamma$ and does not imply branch tracking away from $\Gamma$.
The same calculations leave no frequencies below $-10^{-3}$~THz on the sampled path or mesh.

The dominant DOS peak in the 7--8~THz interval moves from 7.014 to 7.050~THz for PBE, from 7.194 to 7.328~THz for PBEsol, and from 7.538 to 7.744~THz for \rscan{} on replacing DFT with DPA4.
Its height changes from 0.902 to 1.107, from 0.782 to 1.007, and from 0.900 to 1.018~modes/(atom\,THz), respectively.
Since the geometry, force-constant processing, mesh, broadening, and normalization match, these spectral changes reveal differences in the harmonic force response rather than differences in plotting conventions.

\subsection{Scope and remaining checks}
The new reference calculations reached the stated electronic stopping criterion, but convergence with respect to cutoff, electronic-grid density, smearing width, and force-constant supercell size is not yet established.
The PBE and \rscan{} undisplaced-cell force/stress diagnostics were pending when this figure was assembled.
The matched comparison controls these numerical choices between DFT and DPA4 within each pair; it does not remove their effect on the absolute DFT prediction.
Static harmonic spectra also omit thermal expansion and anharmonic renormalization present in the room-temperature experiment.
The additional DPA4 discrepancy is a measure of fidelity to the present same-functional reference, not an isolated measure of optimizer error or the gain over the pretrained model. In particular, the training labels and phonon reference calculations use different electronic sampling and smearing protocols.

The exact figure and numerical inputs are frozen in \texttt{phonons\_matched\_20261101/}: \texttt{bands.csv}, \texttt{DOS.csv}, \texttt{experiment.csv}, \texttt{figure-data.npz}, and the source summary and verification manifests.
The present revision independently checks all source-manifest hashes, reproduces the three spectral RMSEs, and verifies all six DOS integrals.
The provenance manifest \texttt{revision\_nov01\_phonons\_sources.json} links these inputs to the figure and the original reproducible calculation bundle.

'''+needle)
s=rep(s,'No DFT labeling or training run was added for this manuscript revision.',
      'No training run was added for this manuscript revision. The new same-functional DFT phonon references and matched DPA4 force evaluations are documented in Sec.~\\ref{sec:matched_phonons}; they do not change the fixed test-set or thermodynamic data reported elsewhere.')
write('supporting_materials.tex',s)
s=old('main.tex').replace('29 October 2026','1 November 2026')
write('main.tex',s)

provenance={'date':'2026-11-01','source_thread':'01a15c09-6430-7c61-8706-69838a8cbe0c',
 'source_bundle':str(SRC),'verified_manifest_files':len(manifest),'source_manifest_sha256':sha(SRC/'manifest-sha256.json'),
 'figure_source_sha256':sha(SRC/'figure2_phonons_dft_dpa4.pdf'),'model_steps':summary['configuration']['checkpoints'],
 'verified_metrics':checked,'baseline_backup':str(BACKUP),'build':'build/revision-nov01-phonons',
 'scope':'Figure 2, phonon methods/results/discussion, associated abstract/conclusion statements and SI. Other numerical results preserved from the current working manuscript.'}
(MAN/'supporting_data/revision_nov01_phonons_sources.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'verified_source_files':len(manifest),'spectral_RMSE_THz':{k:v['sorted_band_RMSE_THz'] for k,v in checked.items()},'source_figure_sha256':provenance['figure_source_sha256']},indent=2))
