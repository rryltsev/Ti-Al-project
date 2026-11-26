"""Apply the manuscript revision to the frozen pre-revision sources."""
from pathlib import Path
import json
import zipfile

ROOT=Path(__file__).resolve().parents[3]
MAN=ROOT/'manuscript'
BASE=zipfile.ZipFile(Path(__file__).with_name('before_revision.zip'))
def original(name): return BASE.read('manuscript/'+name).decode('utf-8-sig').replace('\r\n','\n')
def write(name,text): (MAN/name).write_text(text,encoding='utf-8')
def replace(text,old,new):
    assert old in text,old[:100]
    return text.replace(old,new)
def section(text,start,end,new):
    a=text.index(start); b=text.index(end,a)
    return text[:a]+new+'\n\n'+text[b:]

t=original('sections/methods.tex')
t=section(t,'The checkpoint retained for evaluation',r'\subsection{Common held-out DFT benchmark}',r'''For PBE, we selected the checkpoint at 15\,000 optimization steps (15k) as a compromise between training/validation accuracy and the harmonic dynamics of hcp Ti.
At this stage, the force errors are close to their late-training plateau and the energy and force errors on both complete training and validation sets are lower than at 5k; among the five saved checkpoints, 15k also gives the closest integral agreement with the PBE DFT phonon density of states (DOS).
This is a joint error-and-property criterion, not early stopping at a minimum of the validation loss or of the training--validation gap: later checkpoints further reduce the energy error.
The final PBEsol and \rscan{} checkpoints, at 23\,100 and 23\,200 steps, are retained.
Learning curves, complete-set errors, and the five-checkpoint phonon comparison at identical DFT and separately optimized geometries are given in the Supporting Materials.
The DFT DOS is a model-selection reference, rather than a wholly independent validation of the selected PBE potential.
No model parameters or energy offsets were adjusted using the held-out structures.

\draftnote{Checkpoint coverage in this revision: PBE 15k is used for the common DFT benchmark, $\alpha$-Ti phonons, and the 2000~K mixing-enthalpy curve. PBE density, viscosity, Ti coexistence, Ti$_3$Al phonons, and mixing enthalpies at 1873 and 2500~K retain the archived 5k results pending recalculation. The Supporting Materials record this scope explicitly.}''')
t=replace(t,'Frozen exports of the supplied fine-tuned checkpoints were evaluated on the same geometries;','The original PBE 15k checkpoint and frozen exports of the retained PBEsol and \\rscan{} checkpoints were evaluated on the same geometries;')
t=replace(t,'Density and mixing enthalpy were obtained from the same production trajectories.',r'''Density and mixing enthalpy were originally obtained from the same production trajectories.
For the updated PBE 15k mixing-enthalpy series at 2000~K, each of the 11 compositions was initialized from the final liquid configuration of the corresponding archived 5k run and propagated for 30~ps of equilibration plus 30~ps of production.
Velocities and the NPT integrator were reinitialized at the stage boundary.
Scalar observables were recorded every 10~fs and full configurations every 100~fs.
The 30~ps production interval is the primary estimate; an extended 49~ps window is assessed only as a sensitivity check in the Supporting Materials.
The main-text density series retains the archived results.''')
t=replace(t,'The preliminary thermodynamic analysis used six 5~ps blocks, with additional block lengths examined as a correlation-time diagnostic.',r'''The retained PBEsol and \rscan{} thermodynamic estimates use six 5~ps blocks.
For the updated PBE 15k series, the conservative standard error of each trajectory mean is the maximum obtained with 3, 5, and 6~ps blocks, requiring at least four blocks; these errors are propagated to the mixing enthalpy.
Additional block lengths and averaging windows are examined in the Supporting Materials.''')
t=replace(t,'The actual production calculations used $5\\times5\\times5$ supercells: 250 atoms for $\\alpha$-Ti and 1000 atoms for Ti$_3$Al.',r'''The archived calculations used $5\times5\times5$ supercells: 250 atoms for $\alpha$-Ti and 1000 atoms for Ti$_3$Al.
The updated PBE 15k $\alpha$-Ti calculation instead independently optimizes $a$ and $c$ at zero external stress and uses a $9\times9\times9$ supercell (1458 atoms), central displacements of $\pm0.01$~\AA, the Frederiksen force correction, three symmetrization iterations, and the acoustic sum rule.
Its displayed DOS uses a $50^3$ mesh, Gaussian width 0.05~THz, and normalization to three modes per atom.
The PBEsol, \rscan{}, and Ti$_3$Al production spectra retain their archived numerical protocols; matched-protocol checkpoint comparisons and convergence limitations are reported separately in the Supporting Materials.''')
write('sections/methods.tex',t)

t=original('sections/results.tex')
t=section(t,'We assess accuracy relative',r'\subsection{Lattice dynamics and thermal density of Ti}',r'''We assess accuracy relative to the DFT references using the same held-out geometries for all three functionals.
The current subset contains 23 configurations: 20 from the liquid pool and one from each of the hcp-, bcc-, and $\gamma$-TiAl-based solid-solution pools.
All three DFT labels and all model predictions are available for these geometries.
Figure~\ref{fig:test_parity} and Table~\ref{tab:fit} compare PBE 15k and the retained final PBEsol and \rscan{} models with their corresponding DFT references, alongside one unchanged pretrained DPA4-Neo baseline.

\input{sections/test23_metrics_table}

\begin{figure*}[t]
\includegraphics[width=\textwidth]{figures/test23_parity.pdf}
\caption{Preliminary parity plots for the same 23 held-out configurations. Panels (a)--(c) compare raw energies per atom and (d)--(f) Cartesian force components against PBE, PBEsol, and \rscan{}, respectively. Blue denotes the fine-tuned model (15k for PBE); orange denotes the unchanged pretrained model, with open symbols for its energies. Errors on the full common subset are reported in Table~\ref{tab:fit}. Marker shapes distinguish liquid-pool (circles), hcp (squares), bcc (triangles), and $\gamma$-TiAl (diamonds) source families. The force statistics contain 15\,228 components from 23 configurations. No energy-reference offsets are fitted or applied; large raw pretrained energy offsets for PBEsol and \rscan{} do not measure configuration-dependent energy errors alone.}
\label{fig:test_parity}
\end{figure*}

The fine-tuned energy MAEs are 2.76, 1.54, and 1.37~meV/atom for PBE, PBEsol, and \rscan{}, respectively; the corresponding RMSEs are 3.36, 1.91, and 1.64~meV/atom.
Descriptive 95\% structure-bootstrap intervals for these RMSEs are 2.37--4.38, 1.36--2.38, and 1.30--1.94~meV/atom.
PBE 15k retains a positive mean residual of 2.60~meV/atom and a largest absolute residual of 8.80~meV/atom.
Its energy MAE is smaller than the pretrained value of 2.94~meV/atom, whereas its RMSE is slightly larger than the pretrained value of 3.29~meV/atom; thus, this subset does not establish a uniform improvement in PBE energy accuracy over the pretrained baseline.
On the same 23 geometries, replacing PBE 5k by 15k lowers energy MAE/RMSE from 4.18/6.11 to 2.76/3.36~meV/atom and force MAE/RMSE from 0.0484/0.0667 to 0.0469/0.0645~eV/\AA.
All configurations are retained.

Fine-tuned force-component RMSEs are 0.0645, 0.0640, and 0.0742~eV/\AA, compared with 0.0740, 0.0860, and 0.1020~eV/\AA{} for the pretrained model against PBE, PBEsol, and \rscan{}, respectively.
The central values decrease by approximately 13\%, 26\%, and 27\% after fine-tuning.
The fine-tuned force-RMSE bootstrap intervals are 0.0559--0.0721, 0.0553--0.0716, and 0.0644--0.0828~eV/\AA; their overlap cautions against a strong ranking of the three fits.
Stress RMSEs are 0.093, 0.082, and 0.087~GPa.
These pooled Cartesian-component errors differ from the force-vector quantities printed in the training logs.

The selected 15k checkpoint improves the PBE training and validation errors relative to 5k while retaining better integral agreement with the hcp-Ti DFT DOS than the later checkpoints.
Later PBE checkpoints further reduce energy errors, so 15k is a property-dependent compromise rather than the minimum-error model for every observable.
The detailed checkpoint comparison, including the earlier 20-structure diagnostic subset, is given in the Supporting Materials.
The larger residual PBE energy error must not be interpreted as an intrinsic limitation of the PBE functional itself.

Raw pretrained energy RMSEs are approximately 411~meV/atom against PBEsol and 4141~meV/atom against \rscan{}.
These values include differences in elemental energy references and exchange-correlation descriptions.
Force and stress comparisons provide complementary evidence because constant elemental energy shifts leave these derivatives unchanged.

The current subset includes only 23 of the planned 50 configurations and only one representative of each crystalline source family.
Unknown correlations between source trajectories, incomplete coverage, and repeated inspection limit the statistical interpretation.
The reported errors therefore characterize the completed geometries; they do not bound solid--liquid free-energy errors or establish convergence of the full benchmark.

\draftnote{Current benchmark: 23 common structures, including 20 liquid-pool and three solid-solution-pool configurations. Update plots, metrics, coverage, and interpretation together as further complete label triplets become available.}''')
t=replace(t,'3.98~THz for the nearly degenerate pair and 5.48~THz','3.82~THz for the nearly degenerate pair and 5.00~THz')
t=replace(t,'8.23~THz for PBE','8.34~THz for PBE')
t=replace(t,'The MLIP curves are harmonic results at their optimized reference geometries.',r'''The PBE curve uses 15k with a $9^3$ supercell and independently optimized $a,c$; PBEsol and \rscan{} retain their $5^3$ production spectra. DOS widths are 0.05~THz for PBE and approximately 0.048~THz for the retained curves. These are harmonic results at the respective optimized geometries; the Supporting Materials provide comparisons with common numerical settings.''')
t=replace(t,'The $\\alpha_2$-Ti$_3$Al calculations likewise',r'The retained $\alpha_2$-Ti$_3$Al calculations (PBE 5k) likewise')
t=replace(t,'The temperature-dependent density of elemental Ti provides',r'The retained temperature-dependent density of elemental Ti (PBE 5k) provides')
t=replace(t,'The coexistence trajectories provide a clearer distinction between the models.',r'The retained coexistence trajectories (PBE 5k) provide a clearer distinction between the models; the PBE 15k melting calculation is not yet included in this revision.')
t=replace(t,'Temperature-dependent density of elemental Ti compared with experiment.',r'Temperature-dependent density of elemental Ti compared with experiment; PBE retains the 5k results.')
t=replace(t,'Solid-like fraction in representative 1296-atom, 50~ps Ti coexistence trajectories,',r'Solid-like fraction in representative 1296-atom, 50~ps Ti coexistence trajectories (PBE 5k),')
t=replace(t,'Liquid density (a) and excess molar volume (b) at 1873~K.',r'Liquid density (a) and excess molar volume (b) at 1873~K; PBE retains the 5k results.')
t=replace(t,'The minima on the sampled grid are $-26.3$, $-25.4$, and $-32.0$~kJ/mol',r'The minima on the sampled grid are $-25.0$, $-25.4$, and $-32.0$~kJ/mol')
t=replace(t,'The values reported here use the thermodynamic enthalpy',r'''The updated PBE 15k minimum is $-25.01\pm0.30$~kJ/mol, where the uncertainty is the propagated block standard error.
Its shift from the archived 5k result, $-26.28$~kJ/mol, makes mixing less exothermic; the complete checkpoint and averaging-window comparison is given in the Supporting Materials.
The values reported here use the thermodynamic enthalpy''')
t=replace(t,'approximately 2.6, 3.1, and 1.5~kJ/mol','approximately 3.6, 3.1, and 1.5~kJ/mol')
t=replace(t,'Liquid mixing enthalpy at 2000~K compared with calorimetric points reproduced in Ref.~\\onlinecite{Witusiewicz2008} (a), and temperature dependence of the model predictions (b). Reference points are a provisional digitization of the calorimetric symbols, not the CALPHAD line. Error bars in (a) are diagnostic standard errors from six 5~ps blocks, including both pure-liquid references; they are not final confidence intervals. Lines connect sampled compositions.',r'''Liquid mixing enthalpy at 2000~K compared with calorimetric points reproduced in Ref.~\onlinecite{Witusiewicz2008} (a), and available isotherms (b). PBE uses 15k at 2000~K and retains 5k at 1873 and 2500~K. The PBE curves in (b) therefore do not constitute a single-checkpoint temperature series. Reference points are a provisional digitization of calorimetric symbols. PBE error bars propagate the maximum block SEM over 3, 5, and 6~ps blocks; PBEsol and \rscan{} retain six-block, 5~ps estimates. Both pure-liquid references are included. These are diagnostic standard errors, not final confidence intervals. Lines connect sampled compositions.''')
t=replace(t,'Increasing temperature makes the enthalpy of mixing less negative for all three models.',r'''The retained single-checkpoint series show less negative mixing enthalpies with increasing temperature.
For PBE this temperature trend still refers to the archived 5k endpoints at 1873 and 2500~K; the new 15k point at 2000~K is not used to infer a thermal derivative.''')
t=replace(t,'STACIE viscosity versus composition at 1873~K (a) and temperature for pure Ti (b).',r'STACIE viscosity versus composition at 1873~K (a) and temperature for pure Ti (b); PBE retains the 5k results.')
write('sections/results.tex',t)

t=original('sections/discussion.tex')
t=replace(t,'The approximately 95~K increase from PBE to',r'The approximately 95~K increase from the retained PBE 5k result to')
t=replace(t,'approximately 5.7~kJ/mol, or 59~meV/atom','approximately 7.0~kJ/mol, or 72~meV/atom')
t=replace(t,'The current liquid-only subset does not constrain systematic biases between solid and liquid; completing the crystalline part of the holdout is therefore essential.',r'The current subset contains 20 liquid-pool structures and only one from each crystalline source family; it does not constrain systematic solid--liquid biases sufficiently, so completing the crystalline holdout remains essential.')
t=replace(t,'including a Ti-rich configuration beyond the same-composition training-volume range.',r'although this error is reduced by the change from 5k to 15k.')
t=replace(t,'The current evidence already establishes a property-dependent pattern for the available potentials:',r'The current evidence shows a property-dependent pattern for the available calculations, with the retained 5k PBE melting, density, and transport results still awaiting a matched 15k update:')
write('sections/discussion.tex',t)
t=original('sections/conclusions.tex')
t=replace(t,'The preliminary common test on 20 liquid-pool configurations yields force RMSEs of 0.065--0.076~eV/\\AA{} but a larger energy error for the PBE fit;',r'The preliminary common test on 23 configurations yields force RMSEs of 0.064--0.074~eV/\AA{} and energy RMSEs of 3.36, 1.91, and 1.64~meV/atom for PBE 15k, PBEsol, and \rscan{}, respectively;')
t=replace(t,'approximately 1760~K for PBE','approximately 1760~K for the retained PBE 5k result')
t=replace(t,'2.6 and 3.1~kJ/mol for PBE and PBEsol','3.6 and 3.1~kJ/mol for PBE 15k and PBEsol')
t=replace(t,'These estimates retain the finite-size and finite-time limitations of the present coexistence protocol.',r'These estimates retain the finite-size and finite-time limitations of the present coexistence protocol; the corresponding PBE 15k coexistence update remains pending.')
write('sections/conclusions.tex',t)
t=original('sections/abstract.tex')
t=replace(t,'approximately 1750--1760~K for the GGA-based models','approximately 1750--1760~K for the retained GGA-based coexistence calculations')
t=replace(t,r'\end{abstract}',r'''The present revision uses PBE 15k for the held-out benchmark, $\alpha$-Ti phonons, and 2000~K mixing enthalpy; the other PBE property results retain 5k pending recalculation.
\end{abstract}''')
write('sections/abstract.tex',t)
t=original('main.tex').replace('25 October 2026','26 October 2026')
t=replace(t,'including learning curves, matched training/validation/test errors, composition and volume diagnostics, and hcp-Ti phonon comparisons at common, separately optimized, and identical DFT reference geometries.',r'including learning curves, matched training/validation/test errors, composition and volume diagnostics, five-checkpoint hcp-Ti phonon comparisons, the 2000~K enthalpy update, and the checkpoint provenance of each property in this revision.')
write('main.tex',t)

# Preserve the original diagnostics while updating their scope and conclusions.
t=original('supporting_materials.tex').replace('25 October 2026','26 October 2026')
t=section(t,'This document examines',r'\section{Checkpoint identity',r'''This document supports selection of PBE 15k through learning curves, complete training/validation errors, and the hcp-Ti PBE DFT phonon DOS. The selection balances these diagnostics: 15k improves energy and force accuracy over 5k and gives the smallest integral DOS discrepancies among the five saved checkpoints, while later checkpoints give smaller energy errors. The current 23-configuration benchmark and the 2000~K mixing-enthalpy update are distinguished below from the earlier 20-configuration checkpoint audit. The remaining PBE properties retain 5k pending recalculation.''')
t=replace(t,'The 5k checkpoint is identical, by SHA-256, to the supplied PBE model used for the main property calculations. Thus, its earlier training stage is intentional.',r'The 5k checkpoint is identical, by SHA-256, to the originally supplied PBE model. The revised selected checkpoint is 15k; the retained property calculations that still use 5k are listed in Table~\ref{tab:coverage}.')
t=section(t,'For PBE, the growing',r'Figure~\ref{fig:learning}',r'''The training--validation comparison motivated an audit of all five saved PBE checkpoints. Selection of 15k combines improved absolute training and validation errors relative to 5k with the best integral agreement with the \emph{PBE DFT DOS of hcp Ti}, evaluated both at identical DFT geometry and at each model's equilibrium cell. Experimental dispersion points are a secondary comparison, not the selection criterion.

At 15k, complete-set energy RMSEs are 3.322/3.512~meV/atom for training/validation, compared with 5.214/6.071 at 5k. Force-component RMSEs are 0.05989/0.06487~eV/\AA, compared with 0.06298/0.06669 at 5k. Further training to 23.4k changes the validation force RMSE only to 0.06468~eV/\AA, but reduces the validation energy RMSE to 1.233~meV/atom. Thus, ``optimal'' refers to the adopted balance between error and phonon criteria, not to a unique minimum of validation loss or the train--validation gap. The increasing force-error gap does not establish general overfitting. The checkpoint audit included held-out errors and is retrospective; no parameters or offsets were fitted to the holdout, but these comparisons are not a new, untouched post-selection test.''')
t=t.replace('selected 5k checkpoint','selected 15k checkpoint')
t=replace(t,'Each checkpoint was evaluated without relaxation',r'The original checkpoint audit, retained here as a fixed diagnostic comparison, evaluated each checkpoint without relaxation')
t=replace(t,'The selected checkpoint was also checked against its frozen export,',r'The original 5k checkpoint was also checked against its frozen export,')
t=replace(t,'The main article uses 5k. Values at later checkpoints are diagnostic comparisons.',r'This table uses the earlier fixed 20-structure liquid test subset. The selected model is now 15k; current 23-structure errors are reported separately below and in the main article.')
t=replace(t,'Only the liquid part of the planned benchmark is presently labeled for all functionals, and these statistics do not constrain solid--liquid free-energy errors.',r'This archived checkpoint sweep used only the 20 liquid structures completed at that time, and these statistics do not constrain solid--liquid free-energy errors.')
t=replace(t,'The large positive energy residual of the selected model is systematic within the training data.',r'The large positive energy residual of the earlier 5k model is systematic within the training data. This section retains the original 5k diagnostic and does not assign its residuals to 15k.')
t=replace(t,'TiAl-test-009 has the largest test residual.',r'TiAl-test-009 has the largest 5k residual on the archived 20-structure test subset.')
t=replace(t,'All 20 structures are retained in the main benchmark.',r'All 20 structures remain included in the expanded 23-structure main benchmark.')
t=replace(t,'archived hcp-Ti geometry used for the selected model','archived hcp-Ti geometry used for the original 5k model')
t=replace(t,r'DFT reference & 2.950000',r'15k equilibrium & 2.938057 & 4.635747 & 1.577827 & 17.327694\\'+'\n'+r'DFT reference & 2.950000')
t=replace(t,'At both the common archived cell and each model\'s own equilibrium cell, 5k is closer to the DFT DOS than the later checkpoints',r'Within the initial comparison restricted to 5k, 20k, and 23.4k, 5k is closer to the DFT DOS at both the common archived cell and each model\'s own equilibrium cell')
ph=json.loads((MAN/'supporting_data/all_checkpoint_phonons.json').read_text())
dos_rows=[]
for step in [5000,10000,15000,20000,23400]:
    fixed=ph['width_checks']['0.05'][f'PBE-{step}/fixed/MP40']; own=ph['width_checks']['0.05'][f'PBE-{step}/equilibrium/MP40']
    dos_rows.append(f'{step/1000:g}k & '+ ' & '.join(f'{x:.4f}' for x in [fixed['L2_relative'],fixed['TV'],fixed['W1_THz'],own['L2_relative']])+r' \\')
newdos=r'''\subsection{Five-checkpoint comparison at identical and equilibrium geometries}
All five checkpoints were evaluated at the unmodified DFT cell and fractional coordinates, without relaxation, using $9^3$ supercells, central displacements of $\pm0.01$~\AA, and the same force-constant corrections. The primary DOS comparison uses a $40\times40\times30$ Monkhorst--Pack mesh and Gaussian width 0.05~THz. Separate equilibrium calculations independently optimize $a$ and $c$ for every checkpoint.

\begin{table}[htbp]
\centering
\caption{DOS discrepancies against the PBE DFT reference for all five checkpoints. The first three metrics use exactly the DFT geometry; the last column uses each model's own equilibrium cell. All curves use the same $40\times40\times30$ mesh and 0.05~THz broadening.}
\label{tab:dos}
\begin{tabular}{lrrrr}
\toprule
Checkpoint & $L_2^{\rm rel}$ & $D_{\rm TV}$ & $W_1$ (THz) & Own-cell $L_2^{\rm rel}$\\
\midrule
'''+ '\n'.join(dos_rows)+r'''
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[htbp]
\centering\includegraphics[width=\linewidth]{figures/supporting/dos_five_checkpoints.pdf}
\caption{The full five-checkpoint PBE comparison: (a) identical DFT geometry and (b) independently optimized equilibrium cells. The reference is the same PBE DFT DOS in both panels. All spectra integrate to three modes per atom; no frequency shift or rescaling is applied. The selected 15k curve has the smallest integral discrepancy in both comparisons.}
\label{fig:matched_dft}
\end{figure}

At identical DFT geometry, 15k reduces $L_2^{\rm rel}$ from 0.2491 to 0.1982 relative to 5k, a 20\% reduction. The total-variation and Wasserstein discrepancies decrease by approximately 22\% and 43\%, respectively. Its preference is preserved for all three metrics on the tested $40\times40\times30$ and $50^3$ Monkhorst--Pack meshes and the $\Gamma$-centered $40\times40\times30$ mesh, at Gaussian widths of 0.05, 0.10, and 0.15~THz. The ordering at fixed geometry is 15k, 5k, 23.4k, 20k, and 10k. The evolution with training is therefore nonmonotonic.

After independent relaxation, the 15k cell has $a=2.938057$~\AA, $c=4.635747$~\AA, and $V/N=17.327694$~\AA$^3$. Its relative DOS discrepancy is 0.2817, versus 0.3134 for 5k, retaining an approximately 10\% advantage. This integral improvement does not imply improvement of every peak: the principal own-cell peaks are about 7.08~THz for 15k and 6.92~THz for 5k, compared with 6.84~THz for DFT. The selected 15k optical frequencies at $\Gamma$ are 3.8196, 3.8204, and 4.9962~THz; its maximum on the sampled path is 8.3378~THz.

\begin{figure}[htbp]
\centering\includegraphics[width=.92\linewidth]{figures/supporting/checkpoint_tradeoff.png}
\caption{Tradeoff between the fixed-geometry DFT-DOS discrepancy and complete validation/test energy RMSEs for the five PBE checkpoints. The test panel uses the original fixed 20-structure liquid diagnostic subset. The selected 15k model minimizes the DOS discrepancy; 20k and 23.4k have smaller energy errors.}
\label{fig:tradeoff}
\end{figure}

\subsection{Numerical limitations}
Identical geometry removes an important confounder but does not make the DFT and MLIP numerical protocols identical. DFT uses a $4\times4\times3$ force-constant supercell whereas the final MLIP comparison uses $9^3$. The original 5k/20k/23.4k size checks support the stability of checkpoint differences but do not establish full convergence of every absolute frequency, nor a separate size-convergence result for 15k. The selected 15k spectrum has no imaginary frequencies below $-0.01$~THz on the sampled path; the largest acoustic residual at $\Gamma$ is about $-0.00027$~THz. The PBE DFT DOS is not a same-functional reference for PBEsol or \rscan{}.
'''
t=section(t,r'\subsection{Evaluation at exactly the DFT geometry}',r'\section{Implications',newdos)
t=section(t,r'\section{Implications',r'\bibliographystyle',r'''\section{Current benchmark and checkpoint coverage}
The current common subset contains 23 configurations: TiAl-test-001--020, -026, -034, and -043. The last three originate from hcp, bcc, and $\gamma$-TiAl solid-solution trajectories. The unchanged pretrained model and all three retained fine-tuned models are compared on precisely these geometries, without energy shifts. The PBE 15k errors use direct CPU inference of the original checkpoint, whose hash matches the phonon and liquid-MD source model. The other two fine-tuned models retain their previous frozen exports. Complete metrics, bootstrap intervals, geometry identities, and model hashes are included in \texttt{test23\_metrics.json}.

On this common subset, PBE 15k energy MAE/RMSE is 2.757/3.356~meV/atom; force MAE/RMSE is 0.04688/0.06454~eV/\AA; stress MAE/RMSE is 0.06209/0.09280~GPa. The corresponding 5k energy errors were 4.18/6.11~meV/atom. These are distinct from the 20-structure diagnostic sweep in Table~\ref{tab:errors}; changing either checkpoint or sample size must be identified explicitly. Only one configuration per crystalline family is currently represented, precluding a robust phase-resolved ranking.

\begin{table}[htbp]
\centering
\caption{PBE checkpoint coverage in the 26 October revision. PBEsol and \rscan{} retain their final checkpoints throughout.}
\label{tab:coverage}
\begin{tabular}{p{.68\linewidth}l}
\toprule
Result & PBE checkpoint\\
\midrule
Current 23-structure parity plots and MAE/RMSE & 15k\\
$\alpha$-Ti dispersion and DOS & 15k\\
Mixing enthalpy at 2000~K, all 11 compositions & 15k\\
Thermal and composition-dependent density; excess volume & 5k, retained\\
Ti solid--liquid coexistence and melting estimate & 5k, retained\\
Liquid viscosity & 5k, retained\\
Ti$_3$Al harmonic phonons & 5k, retained\\
Mixing enthalpies at 1873 and 2500~K & 5k, retained\\
\bottomrule
\end{tabular}
\end{table}

The revised main article consequently contains an explicit transition in PBE checkpoint coverage. The retained 5k results are not attributed to 15k. In particular, the 1873, 2000, and 2500~K PBE enthalpy curves cannot yet be interpreted as a single-checkpoint temperature series, and the retained melting comparison does not establish the melting temperature of 15k.

\section{Updated PBE mixing enthalpy at 2000 K}
All 11 compositions were simulated with 15k in 200-atom cells at a nominal 2000~K and external pressure of 1~bar, using a 1~fs step and the same Melchionna NPT settings as the earlier liquid workflow. Each run starts from the final archived 5k liquid configuration and includes 30~ps equilibration followed by 30~ps production. Velocities and the NPT integrator are restarted between these stages. Energies, temperature, pressure, volume, density, and enthalpy are logged every 10~fs; configurations are saved every 100~fs. The retained main-text PBE density curves are not replaced by these new trajectories.

Enthalpy is evaluated as $H_m=N_{\rm A}\langle E_{\rm pot}+E_{\rm kin}+p_{\rm ext}V\rangle/N$, with a positive external-pressure term and each checkpoint's own pure-liquid references. The main estimate uses only the complete 30~ps production stage. A trajectory mean's standard error is the maximum block SEM over 3, 5, and 6~ps blocks, with at least four blocks. Independent-trajectory propagation includes the mixture and both pure components; shared endpoint references correlate different compositions. These errors describe finite sampling within one trajectory per state and exclude model error, finite-size effects, and between-replica variability.

\begin{table}[htbp]
\centering
\caption{Updated PBE 15k mixing enthalpy at 2000~K in kJ/mol of atoms, using 30~ps production. Endpoint values vanish by definition.}
\label{tab:enthalpy}
\begin{tabular}{rrr}
\toprule
$x_{\rm Al}$ & $\Delta H_{\rm mix}$ & SEM\\
\midrule
\input{supporting_data/pbe15k_enthalpy_rows}
\bottomrule
\end{tabular}
\end{table}

The equiatomic minimum is $-25.007\pm0.302$~kJ/mol, compared with $-26.284\pm0.263$~kJ/mol in the reprocessed 5k archive. The 15k curve is less exothermic over the interior compositions. Its equal-weight mean absolute discrepancy against the same nine provisional calorimetric points used in the main article is 3.61~kJ/mol, compared with the retained values of approximately 3.1 and 1.5~kJ/mol for PBEsol and \rscan{}. The comparison therefore preserves the overall enthalpy ranking in favor of \rscan{}, while increasing the PBE discrepancy. It does not motivate selecting a checkpoint by agreement with the calorimetric curve.

\begin{figure}[htbp]
\centering\includegraphics[width=\linewidth]{figures/supporting/density_enthalpy_5k_15k_20k.pdf}
\caption{Completed 2000~K checkpoint comparison using 30~ps production: density and enthalpy of mixing for PBE 5k, 15k, and 20k. The updated main-text enthalpy uses 15k. The density comparison is a supporting diagnostic; the main-text density series retains its earlier checkpoint.}
\label{fig:enthalpy_checkpoints}
\end{figure}

\subsection{Sensitivity to equilibration and sampling}
An extended window combines 10--30~ps of equilibration with 1--30~ps of production, yielding 49~ps. The first production picosecond is excluded because velocities and the thermostat/barostat were restarted. Blocks do not cross this restart, and stage means are weighted by duration. This expansion changes the 15k mixing enthalpy by at most 0.35~kJ/mol and leaves the broad difference from 5k intact. Individual windows show fluctuations and possible drift, so the 49~ps estimate is a sensitivity check rather than the primary result. The full audit retains other cutoffs and block lengths.

\begin{figure}[htbp]
\centering\includegraphics[width=\linewidth]{figures/supporting/equilibration_window_sensitivity.pdf}
\caption{Sensitivity to including late equilibration in the sampling window. Extended windows exclude the first picosecond following the velocity/integrator restart. The curves share portions of the same trajectories and are not independent estimates.}
\label{fig:window}
\end{figure}

For 200 atoms, the integrator targets $3(N-1)k_{\rm B}T/2$ whereas the logged temperature divides kinetic energy by $3N$; nominal 2000~K therefore corresponds to about 1990~K in this output. The new dense logs give 1989.91--1990.13~K. Subsampling at the older 100~fs cadence produces larger apparent temperature deviations, so the older sparse kinetic averages should not be interpreted as full-trajectory temperatures. The potential-energy mixing comparison confirms the direction of the checkpoint shift independently of this kinetic sampling effect. Diffusive motion is present in all new production trajectories. The trajectory means, block analyses, windows, and quality checks are preserved in \texttt{pbe\_checkpoint\_enthalpy\_statistics.json}.

\section{Implications and reproducibility}
The five-checkpoint analysis supports a joint training/validation-and-phonon choice of 15k. It does not support a monotonic deterioration of phonons after 5k, or general energy/force overfitting at that point. DFT-DOS agreement forms part of selection, whereas comparison with experimental thermodynamic and transport properties probes different aspects of the potential. The current replacement is limited to the rows labeled 15k in Table~\ref{tab:coverage}; remaining property calculations will require their own updates.

The supporting data retain the earlier composition and volume diagnostics with their original 5k identities, alongside the new five-checkpoint phonon results, current 23-structure metrics, and enthalpy statistics. \texttt{revision\_oct26\_sources.json} records exact source paths, hashes, model identities, and the numerical summary. The complete checkpoint hashes identify the same 15k model across inference, phonons, and new liquid MD. No DFT labeling or training run was added for this manuscript revision.

\begin{table}[htbp]
\centering
\caption{SHA-256 prefixes for principal inputs; full hashes are retained in the machine-readable provenance.}
\label{tab:hashes}
\begin{tabular}{ll}
\toprule
Input & SHA-256 prefix\\
\midrule
PBE 5k checkpoint & \texttt{59c9a637757b6d3b}\\
PBE 15k checkpoint & \texttt{39506ecf42a49193}\\
PBE 20k checkpoint & \texttt{3b0b4b69f8fca7be}\\
PBE 23.4k checkpoint & \texttt{ca0dff4ddcadcf36}\\
DFT reference POSCAR & \texttt{bb123213ff836236}\\
DFT reference DOS & \texttt{5e4438aae477df4d}\\
PBE learning log & \texttt{da2d1608b9cdc6eb}\\
\bottomrule
\end{tabular}
\end{table}
''')
t=t.replace('conditional on the current 20 structures','conditional on those 20 structures')
t=t.replace(r"model\'s own equilibrium cell", "model's own equilibrium cell")
t=t.replace(r'\input{supporting_data/pbe15k_enthalpy_rows}',(MAN/'supporting_data/pbe15k_enthalpy_rows.tex').read_text().strip())
write('supporting_materials.tex',t)
print('Revised main manuscript sections and Supporting Materials.')
