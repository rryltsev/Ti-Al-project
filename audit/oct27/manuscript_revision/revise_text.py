"""Replay this revision from its own 27 October snapshot; not for later drafts."""
from pathlib import Path
import json, zipfile
ROOT=Path(__file__).resolve().parents[3]; MAN=ROOT/'manuscript'; OUT=Path(__file__).resolve().parent
Z=zipfile.ZipFile(OUT/'before_revision.zip')
def original(name): return Z.read('manuscript/'+name).decode('utf-8-sig').replace('\r\n','\n')
def replace(s,a,b):
    assert s.count(a)==1,(s.count(a),a[:100]); return s.replace(a,b)
def write(name,s): (MAN/name).write_text(s,encoding='utf-8')

write('sections/abstract.tex',r'''\begin{abstract}
Machine-learning interatomic potentials inherit the exchange-correlation approximation used to generate their reference data, but its effect on finite-temperature alloy properties remains difficult to assess systematically.
We compare Ti--Al potentials obtained by fine-tuning the DPA4-Neo foundation model on PBE, PBEsol, and \rscan{} datasets derived from a common pool of 1000 selected configurations.
Common DFT labeling and fine-tuning settings support comparisons across lattice dynamics, density, melting, liquid mixing thermodynamics, and viscosity.
The PBE checkpoint at 15\,000 steps is selected using training/validation errors and agreement with the hcp-Ti DFT phonon density of states.
The clearest improvement with \rscan{} is found for the melting temperature of Ti: the coexistence estimate increases from approximately 1750--1760~K for the GGA-based potentials to 1855~K, reducing the deviation from the experimental value of 1943~K from about 9--10\% to 4.5\%.
The updated PBE 15k coexistence trajectories bound melting between 1750 and 1775~K.
The \rscan{} model also improves the overall description of the liquid mixing enthalpy at 2000~K, although its minimum is somewhat too exothermic.
These gains do not extend uniformly to other observables: PBE 15k gives closer agreement with the liquid-density references at the updated state points, while harmonic phonons show no uniform functional ranking.
The retained PBE 5k alloy-viscosity series is closer to most available measurements; the new 15k pure-Ti viscosity results do not establish a uniform transport advantage for either functional.
The results identify relative phase and mixing energetics as promising areas for \rscan{}-based Ti--Al potentials and demonstrate why exchange-correlation choices must be assessed against several complementary properties.
\end{abstract}
''')

s=original('sections/methods.tex')
s=replace(s,r'\draftnote{Checkpoint coverage in this revision: PBE 15k is used for the common DFT benchmark, $\alpha$-Ti phonons, and the 2000~K mixing-enthalpy curve. PBE density, viscosity, Ti coexistence, and Ti$_3$Al phonons retain the archived 5k results pending recalculation. The Supporting Materials record this scope explicitly.}',
r'''PBE 15k is used for the common DFT benchmark, $\alpha$-Ti phonons, the 2000~K mixing-enthalpy curve, density and excess volume at 1873~K, liquid-Ti density at 2000~K, Ti coexistence, and pure-Ti viscosity at 1873 and 2000~K.
The remaining PBE thermal-density points, alloy viscosities, pure-Ti viscosities at 1941 and 2100~K, and Ti$_3$Al phonons retain their archived 5k identities.
The Supporting Materials tabulate checkpoint coverage and the updated protocols.''')
s=replace(s,'The main-text density series retains the archived results.',
r'''The updated PBE 15k density and excess-volume series at 1873~K uses the same 30~ps equilibration plus 30~ps production protocol for all 11 compositions, initialized from the corresponding archived 1873~K liquids.
The pure-Ti endpoints at 1873 and 2000~K also enter the temperature-dependent liquid-density figure; its other PBE points retain 5k.
These two updated density estimates use the 10~fs scalar logs rather than the 100~fs configuration sampling used for the retained series.''')
s=replace(s,'these errors are propagated to the mixing enthalpy.','these errors are propagated to the mixing enthalpy and excess molar volume.')
s=replace(s,'The complete system was then propagated for 50~ps in the NPT ensemble at zero external pressure.',
r'''The complete system was then propagated in the NPT ensemble at zero external pressure.
The updated PBE 15k series uses 10+20+10~ps preparation and 50~ps production at 1700, 1750, 1800, and 1850~K; the additional 1775~K trajectory extends production to 100~ps.
The retained PBEsol and \rscan{} production trajectories are 50~ps long.''')
s=replace(s,'Configurations were saved every 0.5~ps.',r'Configurations were saved every 0.1~ps for PBE 15k and every 0.5~ps for the retained trajectories; structural analysis uses a common 0.5~ps interval.')
s=replace(s,'The viscosity simulations used 512-atom periodic cells and the PBE- and \\rscan{}-based potentials.',
r'The archived viscosity simulations used 512-atom periodic cells and the PBE- and \rscan{}-based potentials.')
s=replace(s,'Starting snapshots were drawn from a 5~ps NPT preparation trajectory',
'For the archived ensembles, starting snapshots were drawn from a 5~ps NPT preparation trajectory')
s=replace(s,r'''The equiatomic cases and PBE pure-Ti point used 15 trajectories; the \rscan{} pure-Ti point used 30.
The PBE pure-Ti point at 2000~K also used 50 trajectories.
The remaining pure-Ti state points used 15 trajectories.''',
r'''The equiatomic cases used 15 trajectories, and the \rscan{} pure-Ti point used 30.
The updated PBE 15k pure-Ti estimates at 1873 and 2000~K use 25 and 20 trajectories, respectively, each with 20~ps of stress data sampled at 1~fs.
They replace the earlier 5k estimates based on 15 and 50 trajectories, respectively.
The remaining pure-Ti state points used 15 trajectories.
Only the two pure-Ti PBE records changed in the new STACIE export; unchanged alloy records remain assigned to 5k.
The export supplies sampling and estimator metadata but no raw trajectories, preparation logs, or model hashes, so the detailed 15k preparation and independence of its ensembles cannot be independently verified.''')
write('sections/methods.tex',s)

s=original('sections/results.tex')
s=replace(s,'The retained temperature-dependent density of elemental Ti (PBE 5k) provides a complementary test of the equilibrium volume',
'The temperature-dependent density of elemental Ti provides a complementary test of the equilibrium volume')
s=replace(s,'At 300~K, PBE and',r'For the retained PBE 5k solid-state series, the checkpoint is distinguished explicitly from the new 15k liquid points. At 300~K, PBE and')
s=replace(s,'It therefore covers all six separately prepared liquid states without extrapolation.',
r'It covers the six liquid temperatures from 2000 to 2500~K without extrapolation; the added PBE 15k point at 1873~K lies below its stated range and is not compared with an extrapolated correlation.')
s=replace(s,'At 2000~K, it gives 4.199~g/cm$^3$, compared with 4.271, 4.322, and 4.457~g/cm$^3$ for PBE, \\rscan{}, and PBEsol: deviations of $+1.7\\%$, $+2.9\\%$, and $+6.1\\%$.',
r'''The new PBE 15k liquid densities are $4.2828\pm0.0008$~g/cm$^3$ at 1873~K and $4.2606\pm0.0011$~g/cm$^3$ at 2000~K, with block standard errors.
At 2000~K, the reference correlation gives 4.199~g/cm$^3$, compared with 4.261, 4.322, and 4.457~g/cm$^3$ for PBE 15k, \rscan{}, and PBEsol: deviations of $+1.5\%$, $+2.9\%$, and $+6.1\%$.''')
s=replace(s,'All three liquid series decrease less steeply with temperature than the reference, so these deviations increase to $+4.4\\%$, $+5.9\\%$, and $+9.1\\%$, respectively, at 2500~K.',
r'The retained 2100--2500~K series decrease less steeply with temperature than the reference; at 2500~K the deviations are $+4.4\%$ for PBE 5k, $+5.9\%$ for \rscan{}, and $+9.1\%$ for PBEsol. A full 15k thermal-density curve is not yet available.')
s=replace(s,'PBE is closest to the central reference throughout this interval; its 2000~K value lies within the reference uncertainty band.',
r'PBE is closest to the central reference at the displayed temperatures, with the checkpoint distinction above; the updated 15k value at 2000~K lies within the reference uncertainty band.')
s=replace(s,'Temperature-dependent density of elemental Ti compared with experiment; PBE retains the 5k results.',
r'Temperature-dependent density of elemental Ti compared with experiment. PBE uses 15k for the liquid points at 1873 and 2000~K (filled blue squares, with block SEM bars), and retains 5k elsewhere (open blue symbols).')
s=replace(s,'colored lines connect points within each preparation series.','colored lines connect points within each preparation series and checkpoint.')
s=replace(s,r'''The retained coexistence trajectories (PBE 5k) provide a clearer distinction between the models; the PBE 15k melting calculation is not yet included in this revision.
Figure~\ref{fig:melting} shows the evolution of the solid-like atom fraction near the temperature at which the direction of interface motion changes.
For PBE, the solid grows at 1750~K and disappears at 1800~K.
At 1775~K, the interfaces initially fluctuate with little net displacement, followed by substantial retreat of the solid during the second half of the trajectory.
A crystalline layer remains at 50~ps.
Thus, the interface direction suggests a working interval of 1750--1775~K, while the conservative interval bounded by opposite single-phase outcomes is 1750--1800~K.''',
r'''The coexistence trajectories provide a clearer distinction between the models.
Figure~\ref{fig:melting} uses PBE 15k throughout and shows the evolution of the solid-like atom fraction near the temperature at which the direction of interface motion changes.
For PBE 15k, the crystal grows at 1750~K and the liquid layer ceases to be resolved at approximately 11~ps.
At 1775~K, both interfaces retreat into the solid, which disappears at approximately 12~ps; the system remains liquid through 100~ps.
The 1800~K trajectory also melts, with the crystalline layer disappearing at approximately 13.5~ps.
Opposite single-phase outcomes therefore give a working bracket of 1750--1775~K.
This agrees with the earlier 5k bracket inferred from interface motion, although the 5k crystal survived to 50~ps at 1775~K.
Faster disappearance of the 15k slab does not establish a checkpoint-induced shift in equilibrium melting temperature, because preparation, fluctuations, and interface mobility also affect these individual trajectories.''')
s=replace(s,'Solid-like fraction in representative 1296-atom, 50~ps Ti coexistence trajectories (PBE 5k), using',
r'Solid-like fraction in representative 1296-atom Ti coexistence trajectories, using')
s=replace(s,r'$\bar q_6>0.25$. The two',
r'$\bar q_6>0.25$. PBE uses 15k: the 1775~K trajectory extends to 100~ps and the 1750/1800~K trajectories to 50~ps. PBEsol and \rscan{} trajectories are 50~ps long; panel time ranges differ accordingly. The two')
s=replace(s,'Ti melting estimates and conservative brackets from opposite single-phase outcomes within 50~ps.',
r'Ti melting estimates and brackets from opposite single-phase outcomes. PBE uses 15k, with 50~ps at 1750~K and 100~ps at 1775~K; the other models use 50~ps trajectories. The PBE estimate is a rounded representative value within its 25~K bracket.')
s=replace(s,r'PBE      & $\sim1760$ & 1750--1800 & $-9.4$\\',r'PBE 15k  & $\sim1760$ & 1750--1775 & $-9.4$\\')
s=replace(s,'deviations of 0.116, 0.246, and 0.189','deviations of 0.112, 0.246, and 0.189')
s=replace(s,r'are 3.4\%, 7.5\%, and 5.8\%.',r'are 3.3\%, 7.5\%, and 5.8\%.')
s=replace(s,r'Liquid density (a) and excess molar volume (b) at 1873~K; PBE retains the 5k results.',
r'Liquid density (a) and excess molar volume (b) at 1873~K. The PBE series uses 15k with 30~ps production after 30~ps equilibration; PBEsol and \rscan{} retain their original series.')
s=replace(s,r'$-0.98$~cm$^3$/mol at $x_{\mathrm{Al}}=0.7$ for PBE',r'$-0.96$~cm$^3$/mol at $x_{\mathrm{Al}}=0.6$ for PBE 15k')
s=replace(s,'PBE is closest overall to this reference set.',
r'PBE 15k is closest overall to this reference set.')
s=replace(s,'The available STACIE results provide a transport comparison between PBE and \\rscan{}.',
r'''The available STACIE results provide a transport comparison between PBE and \rscan{}.
The updated PBE 15k estimates replace the pure-Ti points at 1873 and 2000~K; all alloy points and pure Ti at 1941 and 2100~K retain 5k.
Distinct symbols identify these checkpoints in Fig.~\ref{fig:viscosity}, without joining different PBE checkpoints into a single curve.''')
s=replace(s,'The PBE curve reaches its largest sampled value',r'The retained PBE 5k alloy curve reaches its largest sampled value')
s=replace(s,'STACIE viscosity versus composition at 1873~K (a) and temperature for pure Ti (b); PBE retains the 5k results. Error bars show the supplied estimator standard deviations.',
r'STACIE viscosity versus composition at 1873~K (a) and temperature for pure Ti (b). Filled blue squares denote PBE 15k (pure Ti at 1873 and 2000~K); open blue circles denote retained PBE 5k results. Lines connect points of the same checkpoint only. Error bars show the supplied estimator standard errors.')
s=replace(s,'PBE is closer at four of these five compositions;',r'The retained PBE 5k results are closer at four of these five compositions;')
s=replace(s,r'''PBE gives 3.26 and 2.65~mPa\,s at those temperatures.
At 2000~K, PBE gives $2.84\pm0.07$~mPa\,s; the central values of both models decrease monotonically over the sampled temperature range.
A provisional comparison with the Ishikawa correlation gives a mixed ranking: PBE is closer at 1873 and 1941~K, while \rscan{} is closer at 2000 and 2100~K.''',
r'''PBE 15k gives $3.216\pm0.109$~mPa\,s at 1873~K and $2.724\pm0.092$~mPa\,s at 2000~K; the retained 5k value at 2100~K is 2.65~mPa\,s.
The two updated 15k estimates are lower than the corresponding 5k values by 1.3\% and 4.0\%, but the differences are only 0.23 and 1.00 combined standard errors under a zero-covariance assumption.
They therefore do not resolve a systematic checkpoint effect or establish equivalence for alloy transport.
A provisional comparison with the Ishikawa correlation gives a mixed ranking: at 1873~K PBE 15k and \rscan{} are almost equally far from the central reference, on opposite sides (approximately $-7.0\%$ and $+7.0\%$).
The retained PBE 5k result is closer at 1941~K, whereas \rscan{} is closer at 2000 and 2100~K.''')
write('sections/results.tex',s)

s=original('sections/discussion.tex')
s=replace(s,'The approximately 95~K increase from the retained PBE 5k result to',r'The approximately 95~K increase from the representative PBE 15k estimate to')
s=replace(s,'whereas PBE and \\rscan{} have nearly identical densities at 300~K and appreciably different melting temperatures.',
r'whereas the retained PBE 5k and \rscan{} densities at 300~K are nearly identical. The updated PBE 15k and \rscan{} models also differ appreciably in melting temperature despite much smaller relative differences in liquid density.')
s=replace(s,'At 2000~K, PBE is closer than',r'At 2000~K, PBE 15k is closer than')
s=replace(s,'PBE also gives smaller overall discrepancies for the alloy densities at 1873~K.',
r'the updated 15k series also gives smaller overall discrepancies for the alloy densities at 1873~K.')
s=replace(s,'Likewise, the higher alloy viscosity predicted by \\rscan{} is not generally supported by the available measurements.',
r'Likewise, the higher alloy viscosity predicted by \rscan{} is not generally supported by the available measurements compared with the retained PBE 5k series. This alloy-viscosity ranking remains to be tested with PBE 15k; the two new pure-Ti estimates do not establish checkpoint equivalence for mixtures.')
s=replace(s,'The current evidence shows a property-dependent pattern for the available calculations, with the retained 5k PBE melting, density, and transport results still awaiting a matched 15k update: a clear improvement in the Ti coexistence melting estimate, a favorable but provisional overall mixing-enthalpy comparison, and no general improvement in density or transport.',
r'''The updated PBE 15k coexistence and liquid-density calculations preserve the property-dependent pattern: \rscan{} improves the Ti coexistence melting estimate and the provisional overall mixing-enthalpy comparison, whereas PBE 15k is closer to the available liquid-density references at the updated states.
The pure-Ti viscosity update does not establish a uniform transport ranking.
Remaining PBE thermal-density points, Ti$_3$Al phonons, and alloy and two pure-Ti viscosity states still require 15k calculations before the full comparison can be assigned to a single PBE checkpoint.''')
write('sections/discussion.tex',s)

write('sections/conclusions.tex',r'''\section{Conclusions}
\label{sec:conclusions}

We developed three Ti--Al interatomic potentials by fine-tuning the DPA4-Neo foundation model on PBE, PBEsol, and \rscan{} reference datasets derived from a common pool of selected configurations.
Their comparison across lattice dynamics, density, melting, liquid mixing thermodynamics, and viscosity shows that the choice of exchange-correlation reference materially changes predicted properties.
The PBE 15k checkpoint balances improved training/validation errors relative to 5k with the closest integral agreement with the hcp-Ti DFT DOS among the five saved checkpoints; later checkpoints further reduce energy errors.
The preliminary common test on 23 configurations yields force RMSEs of 0.064--0.074~eV/\AA{} and energy RMSEs of 3.36, 1.91, and 1.64~meV/atom for PBE 15k, PBEsol, and \rscan{}, respectively; completion of the holdout is needed to assess residual fitting biases across phases.

The clearest improvement with \rscan{} is the melting temperature of Ti.
The updated PBE 15k trajectories crystallize at 1750~K and melt at 1775~K, giving a 1750--1775~K working bracket consistent with the earlier 5k result.
Representative melting estimates are approximately 1760~K for PBE 15k, 1750~K for PBEsol, and 1855~K for \rscan{}, reducing the discrepancy from experiment from about 9--10\% to 4.5\%.
These brackets are finite-size and finite-time observations, not confidence intervals.
For liquid mixing at 2000~K, \rscan{} gives a more exothermic composition curve and the smallest provisional overall deviation from the reproduced calorimetric data, approximately 1.5~kJ/mol compared with 3.6 and 3.1~kJ/mol for PBE 15k and PBEsol.
It nevertheless overestimates the depth of the mixing-enthalpy minimum.

The gains are selective.
The updated PBE 15k density curve at 1873~K is closer overall to the tabulated reference densities, with a mean absolute deviation of 0.112~g/cm$^3$, compared with 0.246 and 0.189~g/cm$^3$ for PBEsol and \rscan{}.
The two new PBE 15k pure-Ti viscosity estimates decrease from $3.216\pm0.109$~mPa\,s at 1873~K to $2.724\pm0.092$~mPa\,s at 2000~K and do not establish a uniform advantage for either functional against the provisional experimental correlation.
The retained PBE 5k alloy-viscosity series remains closer to most available measurements; this result cannot yet be assigned to 15k.
The remaining thermal-density points, Ti$_3$Al phonons, and viscosity states retain explicit 5k labels pending their own updates.

The thermodynamic results are consistent with improved relative energies of selected environments under \rscan{}.
They do not identify a unique electronic mechanism; testing changes in Ti-derived bonding requires direct electronic-structure and free-energy comparisons.
Foundation-model fine-tuning provides a practical framework for exploring these differences.
For Ti--Al, the present benchmarks favor \rscan{} for the Ti melting estimate and overall liquid mixing enthalpy, while retaining PBE as an important reference for density and transport.
The appropriate electronic-structure reference must therefore be assessed against several observables and the intended range of phases, temperatures, and compositions.
''')

s=original('supporting_materials.tex')
s=replace(s,'Draft -- 26 October 2026','Draft -- 27 October 2026')
s=replace(s,'The remaining PBE properties retain 5k pending recalculation.',
r'The 27 October update additionally incorporates PBE 15k density at 1873~K, liquid-Ti density at 2000~K, Ti coexistence, and two pure-Ti viscosity states. Table~\ref{tab:coverage} identifies the remaining 5k results.')
s=replace(s,'PBE checkpoint coverage in the 26 October revision.','PBE checkpoint coverage in the 27 October revision.')
s=replace(s,r'''Thermal and composition-dependent density; excess volume & 5k, retained\\
Ti solid--liquid coexistence and melting estimate & 5k, retained\\
Liquid viscosity & 5k, retained\\''',
r'''Density and excess volume at 1873~K, all 11 compositions & 15k\\
Liquid-Ti density at 1873 and 2000~K & 15k\\
Ti solid--liquid coexistence and melting estimate & 15k\\
Pure-Ti viscosity at 1873 and 2000~K & 15k\\
Ti density: solid 300--1900~K, liquid 2100--2500~K & 5k, retained\\
Alloy viscosity at 1873~K; pure Ti at 1941 and 2100~K & 5k, retained\\''')
s=replace(s,'The retained melting comparison does not establish the melting temperature of 15k.',
'All main-text PBE coexistence trajectories now use 15k. Historical TDM estimates and checkpoint diagnostics retain their original model identities.')
s=replace(s,'The retained main-text PBE density curves are not replaced by these new trajectories.',
'The pure-Ti 2000~K density from this production stage replaces the corresponding 5k point in the main-text temperature plot; the full 2000~K composition-dependent density curve remains a supporting diagnostic.')
s=replace(s,'The density comparison is a supporting diagnostic; the main-text density series retains its earlier checkpoint.',
'The full density comparison is a supporting diagnostic; its 15k pure-Ti endpoint also supplies the updated 2000~K point in the main-text temperature plot.')
newsections=r'''
\section{Updated density at 1873 K}
All 11 PBE 15k runs used 200 atoms, 1~bar external pressure, 30~ps equilibration, and 30~ps production. The initial configurations came from the corresponding archived liquids. Model identity was verified by SHA-256, and CPU/GPU consistency was checked before MD. The primary density is $\langle M/V\rangle$ over $0<t\leq30$~ps of production, sampled every 10~fs. The excess molar volume uses $N_{\rm A}\langle V\rangle/N$ and the same 15k pure-component trajectories. Standard errors use the maximum block SEM for 3, 5, and 6~ps blocks and are propagated for excess volumes.

\begin{table}[htbp]
\centering
\caption{PBE 15k liquid density and excess molar volume at 1873~K. Errors are block standard errors, excluding protocol and model uncertainties.}
\label{tab:density15k}
\begin{tabular}{rrrrr}
\toprule
$x_{\rm Al}$ & $\rho$ (g/cm$^3$) & SEM & $V_m^{\rm ex}$ (cm$^3$/mol) & SEM\\
\midrule
__DENSITY_ROWS__
\bottomrule
\end{tabular}
\end{table}

The maximum change from the reprocessed 5k density is $-0.00956$~g/cm$^3$ ($-0.285\%$) at $x_{\rm Al}=0.5$. The excess-volume minimum is $-0.9643\pm0.0129$~cm$^3$/mol at $x_{\rm Al}=0.6$. Against the same tabulated experimental convention used in the main article, the density mean absolute discrepancy changes from 0.1160 to 0.1117~g/cm$^3$ (mean absolute relative discrepancy 3.376\% to 3.262\%). These small changes preserve the model ranking.

The archived 5k sampling interval was 100~fs. Decimating the new 15k logs to that cadence can shift a density mean by up to 0.0143~g/cm$^3$, exceeding the largest observed checkpoint difference; the two curves therefore do not isolate a pure checkpoint effect. The largest difference between the two production halves is 0.00588~g/cm$^3$, and using only the last 10~ps changes the mean by at most 0.00326~g/cm$^3$. All trajectories retain diffusive motion. Mean pressure is 0.00434--0.00983~GPa rather than exactly the imposed 0.0001~GPa; these residual and sampling effects are not included in the block SEM. The logged temperatures use $3N$ degrees of freedom; the thermostat convention $3(N-1)$ yields 1872.82--1873.06~K.

The pure-Ti density at 1873~K is $4.28280\pm0.00076$~g/cm$^3$. The independent 2000~K series described above gives $4.26057\pm0.00114$~g/cm$^3$. Both are shown as 15k in the main temperature plot. Remaining crystalline and 2100--2500~K PBE points retain 5k and do not constitute a complete 15k thermal-density curve.

\section{Updated Ti coexistence}
PBE 15k simulations use 1296 atoms, zero imposed pressure, a 1~fs time step, and the same thermostat/barostat damping times as the archived coexistence protocol. Preparation comprises 10~ps equilibration, 20~ps heating one half of the cell to $1.5T$, and 10~ps cooling to the target temperature with a crystalline seed. Production lasts 50~ps at 1700, 1750, 1800, and 1850~K, and 100~ps at 1775~K. Structural analysis subsamples the 0.1~ps saved configurations to a common 0.5~ps interval. Neighbor-averaged $\bar q_6$ uses 14 neighbors; spatial profiles use 36 bins along $z/L_z$ and Gaussian smoothing over one bin to locate the interfaces. Undefined interfaces after loss of two-phase coexistence are not treated as zero-width measured layers.

At 1750~K, the liquid layer ceases to be resolved at about 11~ps. At 1775~K, both interfaces retreat and the crystalline slab disappears at about 12~ps; the liquid persists to 100~ps. At 1800~K, the slab disappears at about 13.5~ps. Thresholds 0.20, 0.25, 0.30, and 0.35 and a criterion based on at least seven correlated bonds preserve these outcomes, although a high threshold undercounts thermally disordered crystal. The working bracket is consequently 1750--1775~K; a representative value of approximately 1760~K does not imply precision finer than this bracket.

The archived 5k trajectory at 1775~K has a similar initial solid-like fraction (0.500 versus 0.488) but retains a fraction of 0.241 at 50~ps, whereas the 15k trajectory is already liquid. The 5k working interval inferred from interface recession was also 1750--1775~K; its more conservative opposite-outcome interval was 1750--1800~K. The new result does not establish a shift of equilibrium melting temperature. Different initial configurations, velocities, and interface kinetics prevent attributing a shorter melting time uniquely to the checkpoint. The bracket uses one realization per temperature and excludes finite-size, interface-orientation, and between-replica uncertainties.

\section{Updated pure-Ti viscosity}
Only two PBE records change in the new STACIE export: pure Ti at 1873 and 2000~K. Six other records in that export are identical to their earlier versions, including the PBE alloys at $x_{\rm Al}=0.50$ and 0.45; they are not additional 15k results. The comparison below uses the latest 5k ensembles already incorporated in the article, including 50 replicas at 2000~K rather than the superseded 15-replica estimate.

\begin{table}[htbp]
\centering
\caption{Pure-Ti STACIE viscosity in mPa\,s. Uncertainties are estimator standard errors; $N$ is the number of 20~ps trajectories.}
\label{tab:viscosity15k}
\begin{tabular}{rrrrrr}
\toprule
$T$ (K) & $\eta_{5k}$ & $N_{5k}$ & $\eta_{15k}$ & $N_{15k}$ & Change (\%)\\
\midrule
1873 & $3.259\pm0.153$ & 15 & $3.216\pm0.109$ & 25 & $-1.33$\\
2000 & $2.839\pm0.069$ & 50 & $2.724\pm0.092$ & 20 & $-4.05$\\
\bottomrule
\end{tabular}
\end{table}

Assuming zero covariance, the standard error of a difference is $(\sigma_{5k}^2+\sigma_{15k}^2)^{1/2}$. The differences are $-0.0434\pm0.1881$ and $-0.1149\pm0.1151$~mPa\,s, respectively, or 0.23 and 1.00 combined standard errors in magnitude. Approximate normal 95\% intervals for the relative differences are $[-12.5,+9.9]\%$ and $[-11.9,+3.8]\%$. The data do not resolve a systematic checkpoint effect, but also do not establish equivalence within 5\% or justify transferring the conclusion to alloy compositions.

Both new series have 20\,001 samples per trajectory at 0.001~ps intervals. Effective spectral sample counts are approximately 34 and 40, and cutoff diagnostics, especially at 2000~K, indicate a need for longer-trajectory checks. More replicas alone do not improve the frequency resolution of a 20~ps record. Estimator uncertainty does not include model error or preparation and finite-size effects. Checkpoint assignment follows the supplied calculation provenance: the export contains no model hash, raw stress trajectories, or preparation logs, so ensemble independence and the detailed preparation cannot be independently verified. The source files and hashes are retained in the supporting provenance. The new estimates replace the two old points; they are not pooled with the 5k ensembles.

The 1873~K alloy viscosities and pure Ti at 1941 and 2100~K remain 5k. Additional 15k alloy calculations are not included until complete. At 1873~K, the new pure-Ti estimate and the \rscan{} result are nearly equidistant from the provisional Ishikawa correlation, at approximately $-7\%$ and $+7\%$. This replaces the earlier statement that PBE was closer at that temperature.

'''
d=json.loads((ROOT/'audit/oct26/pbe15k_density_1873/analysis/statistics.json').read_text())
rows=[]
for r in sorted(d['results']['15000'].values(),key=lambda r:r['x_Al']):
    rows.append(f"{r['x_Al']:.1f} & {r['rho']['mean']:.5f} & {r['rho']['sem']:.5f} & {r['Vex_cm3mol']['mean']:.5f} & {r['Vex_cm3mol']['sem']:.5f} " + r'\\')
newsections=newsections.replace('__DENSITY_ROWS__','\n'.join(rows))
s=replace(s,r'\section{Implications and reproducibility}',newsections+r'\section{Implications and reproducibility}')
s=replace(s,'The complete checkpoint hashes identify the same 15k model across inference, phonons, and new liquid MD.',
r'The complete checkpoint hashes identify the same 15k model across inference, phonons, the new density/enthalpy MD, and coexistence. The separate viscosity export has the provenance limitation specified above. \texttt{revision\_oct27\_sources.json} records the additional source hashes and numerical scope; the historical 26 October manifest remains available.')
write('supporting_materials.tex',s)

p=MAN/'supporting_data/source_manifest.json';m=json.loads(p.read_text())
m['revision_oct27']={'source_manifest':'supporting_data/revision_oct27_sources.json','notes':'revision_oct27_pbe15k_notes_ru.txt',
    'current_build':'build/revision-oct27-pbe15k','selected_PBE_step':15000,'temperature_enthalpy_panel_removed':True}
p.write_text(json.dumps(m,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
print('Updated abstract, methods, results, discussion, conclusions, and SI from the Oct27 snapshot.')
