"""Replay the 29 October edit from its own source snapshot, not from later drafts."""
from pathlib import Path
import json,re,zipfile
ROOT=Path(__file__).resolve().parents[3];MAN=ROOT/'manuscript';OUT=Path(__file__).resolve().parent
Z=zipfile.ZipFile(OUT/'before_revision.zip')
def original(name):return Z.read('manuscript/'+name).decode('utf-8-sig').replace('\r\n','\n')
def rep(s,a,b):
    assert s.count(a)==1,(s.count(a),a[:90]);return s.replace(a,b)
def span(s,start,end,new):
    a=s.index(start);b=s.index(end,a);return s[:a]+new+s[b:]
def write(name,s): (MAN/name).write_text(s,encoding='utf-8')

write('sections/abstract.tex',r'''\begin{abstract}
Machine-learning interatomic potentials inherit the exchange-correlation approximation used to generate their reference data, but its effect on finite-temperature alloy properties remains difficult to assess systematically.
We compare Ti--Al potentials obtained by fine-tuning the DPA4-Neo foundation model on PBE, PBEsol, and \rscan{} datasets derived from a common pool of 1000 selected configurations.
Common DFT labeling and fine-tuning settings support comparisons across lattice dynamics, density, melting, liquid mixing thermodynamics, and viscosity.
The clearest improvement with \rscan{} is found for the melting temperature of Ti: the coexistence estimate increases from approximately 1750--1760~K for the GGA-based potentials to 1855~K, reducing the deviation from the experimental value of 1943~K from about 9--10\% to 4.5\%.
The \rscan{} model also improves the overall description of the liquid mixing enthalpy at 2000~K, although its broad minimum near $x_{\mathrm{Al}}=0.5$--0.6 is somewhat too exothermic.
These gains do not extend uniformly to other observables: PBE gives closer agreement with the liquid-density references and the two alloy-viscosity measurements at matched compositions, while harmonic phonons and pure-Ti viscosity show no uniform functional ranking.
The results identify relative phase and mixing energetics as promising areas for \rscan{}-based Ti--Al potentials and demonstrate why exchange-correlation choices must be assessed against several complementary properties.
\end{abstract}
''')

s=original('sections/methods.tex')
s=span(s,'For PBE, we selected the checkpoint',r'\subsection{Common held-out DFT benchmark}',r'''For PBE, the checkpoint at 15\,000 optimization steps (15k) was selected for the updated physical-property calculations as a compromise between training/validation accuracy and the harmonic dynamics of hcp Ti.
Its energy and force errors on the complete training and validation sets are lower than at 5k, while it gives the closest integral agreement with the PBE DFT phonon density of states among the five saved checkpoints.
This is a joint error-and-property criterion, not early stopping at a minimum of validation loss: later checkpoints further reduce energy error.
For the DFT-reference benchmark reported here, we use the 20\,000-step checkpoint (20k), which gives smaller energy errors and nearly unchanged force errors relative to 15k.
This choice followed inspection of the test results; it is not a blind post-selection evaluation, although no parameters or energy offsets were fitted to the holdout.
The final PBEsol and \rscan{} checkpoints, at 23\,100 and 23\,200 steps, are used throughout.

The PBE physical-property results use 15k for $\alpha$-Ti phonons, mixing enthalpy, density and excess volume at 1873~K, liquid-Ti density at 2000~K, Ti coexistence, and all viscosity points shown in the main article.
The other thermal-density points and the Ti$_3$Al phonons retain the archived 5k calculations.
To keep the subsequent presentation concise, we use the label PBE without repeating checkpoint indices; this allocation and the detailed provenance are recorded here and in the Supporting Materials.
Learning curves, complete-set errors, and the phonon diagnostics supporting model selection are also provided there.
The DFT phonon diagnostic forms part of selection, whereas the main-text lattice-dynamics comparison uses experimental dispersions.

''')
s=rep(s,'The original PBE 15k checkpoint and frozen exports of the retained PBEsol and \\rscan{} checkpoints were evaluated on the same geometries; the unchanged pretrained checkpoint was evaluated once and its predictions compared separately with each functional.',
r'The three fine-tuned models were evaluated on the same geometries; the unchanged pretrained model was evaluated once and its predictions compared separately with each functional.')
s=rep(s,'For the updated PBE 15k mixing-enthalpy series at 2000~K, each of the 11 compositions was initialized from the final liquid configuration of the corresponding archived 5k run',
'For the PBE mixing-enthalpy series at 2000~K, each of the 11 compositions was initialized from the final configuration of a previously equilibrated liquid')
s=rep(s,'The updated PBE 15k density and excess-volume series', 'The PBE density and excess-volume series')
s=rep(s,'; its other PBE points retain 5k.','.')
s=rep(s,'The retained PBEsol and \\rscan{} thermodynamic estimates use six 5~ps blocks.',
r'''The PBEsol enthalpy estimates and the retained density series use six 5~ps blocks.
For \rscan{} mixing enthalpy at 2000~K, every composition and both pure-component references were extended by 30~ps from the final coordinates, cell, and momenta, giving 60~ps of accumulated production.
The duplicate boundary frame was excluded; thermostat and barostat variables were reinitialized at the continuation.
The main estimate uses twelve 5~ps blocks, with block-length and restart sensitivity assessed in the Supporting Materials.''')
s=rep(s,'For the updated PBE 15k series','For the PBE series')
s=span(s,r'\subsection{Harmonic lattice dynamics}',r'\subsection{Solid--liquid coexistence and melting of Ti}',r'''\subsection{Harmonic lattice dynamics}
\label{sec:phonons}

Harmonic phonons of $\alpha$-Ti and $\alpha_2$-Ti$_3$Al were calculated using finite displacements in ASE.
The PBE $\alpha$-Ti calculation independently optimizes $a$ and $c$ at zero external stress and uses a $9\times9\times9$ supercell (1458 atoms), central displacements of $\pm0.01$~\AA, the Frederiksen force correction, three symmetrization iterations, and the acoustic sum rule.
The PBEsol and \rscan{} $\alpha$-Ti calculations use $5\times5\times5$ supercells (250 atoms); Ti$_3$Al calculations use 1000 atoms.
For these calculations, BFGS relaxation preserves crystal symmetry and allows hydrostatic cell strain, with a force convergence target of $10^{-4}$~eV\,\AA$^{-1}$.
Positive and negative Cartesian displacements of 0.01~\AA{} were used to construct force constants, followed by enforcement of the acoustic sum rule.
Dispersions were obtained from the corresponding dynamical matrices and compared with digitized room-temperature experimental points reproduced in Ref.~\onlinecite{Souvatzis2007}.
The calculations are harmonic spectra about optimized structures; they do not include finite-temperature anharmonic renormalization.
Numerical convergence and the separate diagnostics used for model selection are documented in the Supporting Materials.

''')
s=rep(s,'The updated PBE 15k series uses','The PBE series uses')
s=rep(s,'every 0.1~ps for PBE 15k','every 0.1~ps for PBE')
s=span(s,'Shear viscosity was calculated', 'For an isotropic liquid, define five equivalent shear channels,',r'''Shear viscosity was calculated from equilibrium stress fluctuations using the Green--Kubo relation.
The alloy simulations used 512-atom periodic cells and the PBE- and \rscan{}-based potentials.
The PBE composition series at 1873~K contains pure Ti and three alloys at nominal $x_{\mathrm{Al}}=0.20$, 0.45, and 0.80; the corresponding finite-cell Al fractions are 103/512, 231/512, and 410/512.
The \rscan{} series covers $c_{\mathrm{Ti}}=0.1,0.2,0.3,0.4,0.5,0.55,0.6,0.7,0.8,0.9$, and 1, where $c_{\mathrm{Ti}}=1-x_{\mathrm{Al}}$.
Pure Al is not included.

Each PBE alloy was equilibrated for 20~ps in NPT at 1873~K and 1~bar.
Starting snapshots were drawn from a continuing NPT reservoir at 5~ps intervals, assigned velocities, and equilibrated for 1~ps in NVT before production at the snapshot's fixed volume.
Preparation thermostat and barostat damping times were 0.1 and 0.5~ps; the production thermostat damping time was 0.2~ps.
The reported alloy estimates use 15 replicas per composition, each with 40~ps of production and a 1~fs time step.
The stress tensor was recorded at every step; the first and second 20~ps windows were analyzed separately as diagnostics.
These windows and cumulative replica estimates are correlated, and the reported ensemble size does not by itself establish convergence.

The PBE pure-Ti estimates at 1873 and 2000~K use 25 and 20 replicas, respectively, with 20~ps of stress data sampled at 1~fs.
Their supplied exports contain sampling and estimator metadata; limitations in the available preparation provenance are documented in the Supporting Materials.
For \rscan{}, starting snapshots were drawn from a 5~ps NPT preparation at the target temperature and 1~bar, followed by velocity assignment and 1~ps NVT equilibration.
Production used 20~ps per replica and a 0.2~ps thermostat damping time.
At 1873~K there are 50 replicas per alloy except the equiatomic case (15); pure Ti uses 30 replicas.
The additional \rscan{} pure-Ti points at 1941, 2000, and 2100~K use 15 replicas each.

''')
s=rep(s,r'\draftnote{The viscosity protocol describes the presently supplied trajectories. The agreed checks of sampling duration and preparation, and any additional STACIE processing at 1773~K, must be reflected here when those calculations are finalized.}',
r'\draftnote{Replica and trajectory-length convergence checks continue for the PBE alloy viscosities. The Supporting Materials report the diagnostics for the 15-replica estimates used here.}')
write('sections/methods.tex',s)

s=original('sections/results.tex')
s=span(s,'We assess accuracy relative',r'\subsection{Lattice dynamics and thermal density of Ti}',r'''We assess accuracy relative to the DFT references using common held-out geometries for all three functionals.
The test design contains 50 configurations: 25 liquids and 25 structures from hcp-, bcc-, and $\gamma$-TiAl-based solid-solution pools.
Figure~\ref{fig:test_parity} and Table~\ref{tab:fit} compare the fine-tuned models with their corresponding DFT references and with one unchanged pretrained DPA4-Neo baseline.

\input{sections/testset_metrics_table}

\begin{figure*}[t]
\includegraphics[width=\textwidth]{figures/testset_parity.pdf}
\caption{Parity plots for 30 completed common geometries from the 50-configuration test design. Panels (a)--(c) compare raw energies per atom and (d)--(f) Cartesian force components against PBE, PBEsol, and \rscan{}, respectively. Blue denotes the fine-tuned model; orange denotes the unchanged pretrained model, with open energy symbols. Marker shapes identify liquid (circles), hcp (squares), bcc (triangles), and $\gamma$-TiAl (diamonds) source families. Force statistics contain 20\,532 components. No energy-reference offsets are fitted or applied.}
\label{fig:test_parity}
\end{figure*}

The fine-tuned energy MAEs are 1.14, 1.45, and 1.36~meV/atom for PBE, PBEsol, and \rscan{}, respectively; the corresponding RMSEs are 1.51, 1.77, and 1.60~meV/atom.
Descriptive 95\% structure-bootstrap intervals for these RMSEs are 1.05--1.97, 1.33--2.19, and 1.32--1.87~meV/atom.
PBE has a mean energy residual of $+0.53$~meV/atom and a largest absolute residual of 4.42~meV/atom.
Its energy MAE and RMSE are both smaller than the pretrained values of 2.68 and 3.16~meV/atom.

Fine-tuned force-component RMSEs are 0.0638, 0.0633, and 0.0731~eV/\AA, compared with 0.0737, 0.0842, and 0.1027~eV/\AA{} for the pretrained model against PBE, PBEsol, and \rscan{}, respectively.
The central values decrease by approximately 13\%, 25\%, and 29\% after fine-tuning.
The force-RMSE bootstrap intervals are 0.0572--0.0701, 0.0569--0.0697, and 0.0655--0.0804~eV/\AA; their overlap cautions against a strong ranking of the three fits.
Stress RMSEs are 0.082, 0.079, and 0.083~GPa.
These pooled Cartesian-component errors differ from the force-vector quantities printed in the training logs.

Raw pretrained energy RMSEs are approximately 421~meV/atom against PBEsol and 4216~meV/atom against \rscan{}.
They include differences in elemental energy references and exchange-correlation descriptions, and therefore do not measure configuration-dependent errors alone.
Force and stress comparisons provide complementary evidence because constant elemental energy shifts leave these derivatives unchanged.

''')
s=rep(s,'The densities of states show a corresponding redistribution of spectral weight.\nThe independent PBE DFT density of states is included as a separate theoretical reference; it is not an experimental benchmark for choosing between functionals.\n', '')
s=span(s,r'\begin{figure*}[t]'+'\n'+r'\includegraphics[width=\textwidth]{figures/phonons_alpha_ti.pdf}',
'The temperature-dependent density of elemental Ti',r'''\begin{figure}[!t]
\includegraphics[width=\columnwidth]{figures/phonons_alpha_ti.pdf}
\caption{Harmonic dispersions of $\alpha$-Ti along $\Gamma$--K--M--$\Gamma$--A: (a) PBE, (b) PBEsol, and (c) \rscan{}. Black symbols are digitized room-temperature experimental points reproduced in Ref.~\onlinecite{Souvatzis2007}, attributed there to Stassis \textit{et al.} The curves are harmonic calculations at the respective optimized geometries; their numerical settings are specified in Sec.~\ref{sec:phonons}.}
\label{fig:phonons}
\end{figure}

The $\alpha_2$-Ti$_3$Al calculations likewise have no appreciable imaginary frequencies along the sampled $\Gamma$--M--K--$\Gamma$--A--L--H--A path; the largest residual negative magnitude is below $10^{-3}$~THz.
Their maximum frequencies are 9.14, 9.48, and 9.58~THz for PBE, PBEsol, and \rscan{}, respectively.
These results characterize model differences for the compound; the experimental dispersion comparison in Fig.~\ref{fig:phonons} concerns elemental $\alpha$-Ti.
Neither the harmonic calculations nor the room-temperature comparison determine the temperature-renormalized phonon spectrum near melting.

''')
s=rep(s,'For the retained PBE 5k solid-state series, the checkpoint is distinguished explicitly from the new 15k liquid points. ','')
s=rep(s,' A full 15k thermal-density curve is not yet available.','')
s=rep(s,'PBE is closest to the central reference at the displayed temperatures, with the checkpoint distinction above; the updated 15k value at 2000~K lies within the reference uncertainty band.',
'PBE is closest to the central reference at the displayed temperatures; its value at 2000~K lies within the reference uncertainty band.')
s=rep(s,' PBE uses 15k for the liquid points at 1873 and 2000~K (filled blue squares, with block SEM bars), and retains 5k elsewhere (open blue symbols).','')
s=rep(s,'within each preparation series and checkpoint.','within each preparation series. Block SEM bars are shown for PBE at 1873 and 2000~K.')
s=rep(s,'Figure~\\ref{fig:melting} uses PBE 15k throughout and shows',r'Figure~\ref{fig:melting} shows')
s=rep(s,'This agrees with the earlier 5k bracket inferred from interface motion, although the 5k crystal survived to 50~ps at 1775~K.\nFaster disappearance of the 15k slab does not establish a checkpoint-induced shift in equilibrium melting temperature, because preparation, fluctuations, and interface mobility also affect these individual trajectories.\n','')
s=rep(s,'PBE uses 15k: the 1775~K trajectory', 'For PBE, the 1775~K trajectory')
s=rep(s,'PBE uses 15k, with 50~ps','PBE uses 50~ps')
s=rep(s,'The PBE series uses 15k with 30~ps production','The PBE series uses 30~ps production')
s=span(s,'The minima on the sampled grid are', 'The values reported here use',r'''The PBE and PBEsol minima are $-25.0$ and $-25.4$~kJ/mol at $x_{\mathrm{Al}}=0.5$.
The extended 60~ps \rscan{} series gives $-31.66\pm0.19$~kJ/mol at $x_{\mathrm{Al}}=0.5$ and $-31.74\pm0.20$~kJ/mol at 0.6.
Although the latter is the numerical minimum, their difference is only $-0.081\pm0.263$~kJ/mol, including the shared pure-component covariance, so its position is not resolved between these compositions.
The PBE minimum is $-25.01\pm0.30$~kJ/mol.
The quoted errors are propagated block standard errors; the longer \rscan{} sampling changes the mean curve by at most 0.39~kJ/mol and does not uniformly reduce these errors.
''')
s=rep(s,' PBE uses the 15k checkpoint.','')
s=rep(s,'PBEsol and \\rscan{} retain six-block, 5~ps estimates.',r'PBEsol uses six 5~ps blocks and \rscan{} uses twelve 5~ps blocks from 60~ps production.')
s=span(s,'The available STACIE results provide', 'Taken together, the present benchmarks',r'''The STACIE results compare PBE and \rscan{} transport properties (Fig.~\ref{fig:viscosity}).
At 1873~K, PBE gives $3.216\pm0.109$~mPa\,s for pure Ti and $3.985\pm0.135$, $4.725\pm0.175$, and $1.826\pm0.054$~mPa\,s at $x_{\mathrm{Al}}=0.20$, 0.45, and 0.80, respectively.
The alloy estimates use 15 replicas of 40~ps each.
They show an enhancement at intermediate composition, with the largest sampled value at $x_{\mathrm{Al}}=0.45$; the sampled grid does not locate a continuous maximum precisely.
The \rscan{} viscosities exceed the PBE values at all four matched compositions and form a broad high-viscosity region over approximately $x_{\mathrm{Al}}=0.40$--0.60.
The largest sampled \rscan{} value is $7.19\pm0.28$~mPa\,s at $x_{\mathrm{Al}}=0.45$, close to $7.05\pm0.26$~mPa\,s at 0.40.

\begin{figure*}[t]
\includegraphics[width=\textwidth]{figures/viscosity.pdf}
\caption{STACIE viscosity versus composition at 1873~K (a) and temperature for pure Ti (b). The three PBE alloy points use 15 replicas of 40~ps each; its pure-Ti point in (a) is the same 1873~K estimate as in (b). Error bars are estimator standard errors; lines connect sampled states. Black squares in (a) are values of the experimental correlations of Takeda \textit{et al.}\cite{Takeda2025}; open squares require a 3~K extrapolation. The dashed curve with an asterisk in (b) is a provisional transcription of the Ishikawa \textit{et al.}\cite{Ishikawa2012} correlation; shading denotes its reported 10.3\% uncertainty, not a specified confidence level.}
\label{fig:viscosity}
\end{figure*}

The higher \rscan{} alloy viscosity does not improve agreement at the two compositions for which both PBE and experimental values are available.
At $x_{\mathrm{Al}}=0.45$, the Takeda correlation gives 5.30~mPa\,s, compared with 4.73 for PBE and 7.19 for \rscan{}.
At $x_{\mathrm{Al}}=0.80$, the corresponding values are 1.764, 1.826, and 2.528~mPa\,s.
PBE is closer at both compositions.
The value at 0.45 involves a 3~K extrapolation beyond the stated measurement range; the value at 0.80 is interpolated.
The experimental alloy series does not extend below $x_{\mathrm{Al}}=0.45$, so it cannot confirm the predicted increase on the Ti-rich side.

For pure Ti, PBE decreases from $3.216\pm0.109$~mPa\,s at 1873~K to $2.724\pm0.092$~mPa\,s at 2000~K.
The \rscan{} series decreases from 3.70~mPa\,s at 1873~K to 2.99~mPa\,s at 2100~K, with 3.30~mPa\,s at 2000~K.
Against the provisional Ishikawa correlation, PBE and \rscan{} are almost equally far from the central value at 1873~K, on opposite sides (approximately $-7\%$ and $+7\%$).
At 2000~K, \rscan{} is closer.
These pure-Ti comparisons do not establish a uniform functional ranking across the liquid states studied.

The statistical interpretation remains sensitive to stress-correlation sampling.
For the PBE alloys, accumulated estimates and separate 20~ps windows reveal residual replica and spectral-cutoff sensitivity, detailed in the Supporting Materials.
The estimator error bars do not include all preparation, finite-size, or model uncertainties and do not establish complete convergence at 15 replicas.
An archived comparison of 25 matched trajectory ensembles gave a median absolute TDM--STACIE difference of 10.6\% and a maximum of 32.2\%; this is a methodological diagnostic, not an additional error bar for the present estimates.
STACIE is used consistently for the main comparison, and TDM-only estimates at 1773~K are kept separate.

\draftnote{Verify the provisional pure-Ti correlation and its uncertainty against the full Ishikawa paper before finalizing this comparison.}

''')
s=s.replace('PBE 15k','PBE').replace('PBE 5k','PBE')
s=s.replace('the added PBE point','the PBE point').replace('The new PBE liquid densities','The PBE liquid densities')
assert not re.search(r'checkpoint|\b(?:5k|15k|20k)\b',s,re.I)
write('sections/results.tex',s)

s=original('sections/discussion.tex')
s=rep(s,'although the equiatomic minimum becomes too deep.','although the broad minimum near equiatomic composition becomes too deep.')
s=rep(s,'the representative PBE 15k estimate','the representative PBE estimate')
s=rep(s,'whereas the retained PBE 5k and \\rscan{} densities at 300~K are nearly identical. The updated PBE 15k and',
r'whereas the PBE and \rscan{} densities at 300~K are nearly identical. The PBE and')
s=rep(s,'approximately 7.0~kJ/mol, or 72~meV/atom.','approximately 6.65~kJ/mol, or 69~meV/atom.')
s=rep(s,'At 2000~K, PBE 15k is closer than','At 2000~K, PBE is closer than')
s=rep(s,'the updated 15k series also gives','PBE also gives')
s=rep(s,'Likewise, the higher alloy viscosity predicted by \\rscan{} is not generally supported by the available measurements compared with the retained PBE 5k series. This alloy-viscosity ranking remains to be tested with PBE 15k; the two new pure-Ti estimates do not establish checkpoint equivalence for mixtures.',
r'Likewise, the higher alloy viscosity predicted by \rscan{} does not improve agreement at the two experimentally covered compositions shared with the PBE calculations.')
s=rep(s,'The preliminary common-geometry benchmark in Sec.~\\ref{sec:results_fit} shows comparable force-error scales but a larger residual PBE energy error, although this error is reduced by the change from 5k to 15k.',
r'The common-geometry benchmark in Sec.~\ref{sec:results_fit} gives comparable energy- and force-error scales for the three fits, but these pooled errors do not constrain systematic solid--liquid free-energy biases directly.')
s=rep(s,'The current subset contains 20 liquid-pool structures and only one from each crystalline source family; it does not constrain systematic solid--liquid biases sufficiently, so completing the crystalline holdout remains essential.\n','')
s=rep(s,'The updated PBE 15k coexistence and liquid-density calculations preserve the property-dependent pattern:',
'The coexistence and liquid-density calculations show a property-dependent pattern:')
s=rep(s,'whereas PBE 15k is closer to the available liquid-density references at the updated states.',
'whereas PBE is closer to the available liquid-density references.')
s=rep(s,'Remaining PBE thermal-density points, Ti$_3$Al phonons, and alloy and two pure-Ti viscosity states still require 15k calculations before the full comparison can be assigned to a single PBE checkpoint.\n','')
s=rep(s,'The pure-Ti viscosity update','The pure-Ti viscosity comparison')
assert not re.search(r'checkpoint|\b(?:5k|15k|20k)\b',s,re.I)
write('sections/discussion.tex',s)

write('sections/conclusions.tex',r'''\section{Conclusions}
\label{sec:conclusions}

We developed Ti--Al interatomic potentials by fine-tuning DPA4-Neo on PBE, PBEsol, and \rscan{} datasets derived from a common pool of selected configurations.
Their comparison across lattice dynamics, density, melting, mixing thermodynamics, and viscosity shows that exchange-correlation choices have property-dependent consequences.
The DFT-reference benchmark gives energy RMSEs of 1.51, 1.77, and 1.60~meV/atom for PBE, PBEsol, and \rscan{}, respectively, with force RMSEs of 0.063--0.073~eV/\AA.
These fitting errors provide context for the property comparison, but do not determine the errors in phase free-energy differences.

The clearest improvement with \rscan{} is the melting temperature of Ti.
PBE trajectories crystallize at 1750~K and melt at 1775~K, giving a working bracket of 1750--1775~K.
Representative melting estimates are approximately 1760~K for PBE, 1750~K for PBEsol, and 1855~K for \rscan{}, reducing the discrepancy from experiment from about 9--10\% to 4.5\%.
These brackets are finite-size and finite-time observations, not confidence intervals.
For liquid mixing at 2000~K, the extended 60~ps \rscan{} series has the smallest provisional overall discrepancy from the reproduced calorimetric data, approximately 1.5~kJ/mol compared with 3.6 and 3.1~kJ/mol for PBE and PBEsol.
Its broad minimum at $x_{\mathrm{Al}}=0.5$--0.6 remains too exothermic; the two sampled compositions are not statistically distinguished as the location of the minimum.

The gains are selective.
PBE is closer overall to the tabulated liquid densities at 1873~K, with a mean absolute deviation of 0.112~g/cm$^3$, compared with 0.246 and 0.189~g/cm$^3$ for PBEsol and \rscan{}.
PBE also gives closer alloy viscosities at the two matched experimental compositions, $x_{\mathrm{Al}}=0.45$ and 0.80.
The 15-replica alloy estimates retain sampling limitations, and neither the pure-Ti viscosity comparison nor the experimental phonon dispersions provide a uniform ranking of the functionals.

The thermodynamic results are consistent with improved relative energies of selected environments under \rscan{}.
They do not identify a unique electronic mechanism; testing changes in Ti-derived bonding requires direct electronic-structure and free-energy comparisons.
Foundation-model fine-tuning provides a practical framework for exploring these differences.
The electronic-structure reference should therefore be assessed against several observables and the intended range of phases, temperatures, and compositions.
''')

s=original('main.tex').replace('27 October 2026','29 October 2026')
s=re.sub(r'See the Supporting Materials[^\n]+',lambda m: 'See the Supporting Materials for learning curves, training/validation/test errors, numerical lattice-dynamics diagnostics, mixing-enthalpy and viscosity sampling checks, density and coexistence analyses, and data provenance.',s)
write('main.tex',s)
print('Main article text updated.')
