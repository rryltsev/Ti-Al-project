"""Update the supporting audit to the inputs used in the 29 October article."""
from pathlib import Path
import json,zipfile
ROOT=Path(__file__).resolve().parents[3];MAN=ROOT/'manuscript';OUT=Path(__file__).resolve().parent
with zipfile.ZipFile(OUT/'before_revision.zip') as z:s=z.read('manuscript/supporting_materials.tex').decode('utf-8-sig').replace('\r\n','\n')
def rep(a,b):
    global s
    assert s.count(a)==1,(s.count(a),a[:80]);s=s.replace(a,b)
rep('Draft -- 27 October 2026','Draft -- 29 October 2026')
rep('The current 23-configuration benchmark and the 2000~K mixing-enthalpy update are distinguished below from the earlier 20-configuration checkpoint audit. The 27 October update additionally incorporates PBE 15k density at 1873~K, liquid-Ti density at 2000~K, Ti coexistence, and two pure-Ti viscosity states.',
    'The current benchmark uses PBE 20k on 30 completed common structures; the physical-property selection and the earlier fixed 20-structure audit remain separate. The present revision additionally incorporates the 60~ps r2SCAN enthalpy series and three PBE 15k alloy-viscosity estimates with 15 replicas each.')
rep('The revised selected checkpoint is 15k; the retained property calculations that still use 5k are listed in Table~\\ref{tab:coverage}.',
    r'The selected checkpoint for the updated physical properties is 15k; the DFT-reference benchmark uses 20k. The allocation, including retained 5k properties, is listed in Table~\ref{tab:coverage}.')
rep('All 20 structures remain included in the expanded 23-structure main benchmark.','All 20 structures remain included in the expanded 30-structure main benchmark.')
start=s.index(r'\section{Current benchmark and checkpoint coverage}');end=s.index(r'\begin{table}',start)
s=s[:start]+r'''\section{Current benchmark and checkpoint coverage}
The test design comprises 50 geometries selected before evaluating model errors. The present numerical snapshot contains 30 completed common DFT triplets: TiAl-test-001--020, -026--029, -034--036, and -043--045. These comprise 20 liquid-pool, four hcp, three bcc, and three $\gamma$-TiAl source structures, with 20\,532 force components. The main-text captions and metric table state the actual sample size; no predictions or labels for the remaining structures are imputed.

The PBE benchmark now uses 20k, while the main physical-property results retain the allocation in Table~\ref{tab:coverage}. PBEsol and \rscan{} retain their frozen exports, and the pretrained predictions are unchanged. Energies are raw, without fitted offsets. PBE 20k energy MAE/RMSE is 1.138/1.514~meV/atom, force MAE/RMSE is 0.04694/0.06376~eV/\AA, and stress MAE/RMSE is 0.05465/0.08205~GPa. Full metrics and bootstrap intervals are in \texttt{test30\_metrics.json}. All 36 energy/force/stress MAE and RMSE entries for fine-tuned and pretrained models were independently reproduced from the cached predictions and checked model/geometry hashes.

The change to 20k followed inspection of Test50 results and does not constitute blind selection. No model weights or energy offsets were fitted to the holdout, but the reported comparisons are conditional on retrospective model selection, the completed geometries, and unknown trajectory correlations. The 20-structure checkpoint sweep elsewhere in this document is preserved as its original diagnostic and must not be confused with the 30-structure benchmark.

'''+s[end:]
rep('PBE checkpoint coverage in the 27 October revision.','PBE checkpoint coverage in the 29 October revision.')
rep(r'Current 23-structure parity plots and MAE/RMSE & 15k\\',r'Current 30-structure parity plots and MAE/RMSE & 20k\\')
rep(r'Pure-Ti viscosity at 1873 and 2000~K & 15k\\',r'Pure-Ti viscosity at 1873 and 2000~K & 15k\\'+'\n'+r'Alloy viscosity at 1873~K, $x_{\rm Al}=0.20,0.45,0.80$ & 15k\\')
rep(r'Alloy viscosity at 1873~K; pure Ti at 1941 and 2100~K & 5k, retained\\'+'\n','')
rep('The revised main article consequently contains an explicit transition in PBE checkpoint coverage. The retained 5k results are not attributed to 15k. All main-text PBE coexistence trajectories now use 15k. Historical TDM estimates and checkpoint diagnostics retain their original model identities.',
    'Checkpoint allocation is stated in the main Fine-tuning subsection; subsequent main-text sections and figures use the functional label PBE. This convention does not reassign historical data to another model. All PBE viscosities shown in the main article now use 15k; the older 5k alloy and pure-Ti transport points are excluded. Historical TDM and checkpoint diagnostics retain their original identities.')
rep('compared with the retained values of approximately 3.1 and 1.5~kJ/mol for PBEsol and \\rscan{}.',
    r'compared with approximately 3.1~kJ/mol for PBEsol and 1.47~kJ/mol for the current 60~ps \rscan{} series.')
scan=json.loads((ROOT/'audit/oct26/r2scan_enthalpy_extend30k/analysis/comparison.json').read_text())
rows=[]
for r in sorted(scan['comparison'],key=lambda r:r['x_Al']):
    rows.append(f"{r['x_Al']:.1f} & ${r['deltaH_30_kJ_mol']:.3f}\\pm{r['SEM_30_kJ_mol']:.3f}$ & ${r['deltaH_60_kJ_mol']:.3f}\\pm{r['SEM_60_kJ_mol']:.3f}$ "+r'\\')
section=r'''\section{Extended r2SCAN mixing-enthalpy statistics}
All 11 liquid compositions at 2000~K were continued from the last saved coordinates, cell, and momenta for another 30\,000 steps of 1~fs, giving 60~ps of accumulated production per composition. The cells contain 200 atoms and the imposed pressure is 1~bar. The duplicate boundary frame was excluded, leaving 601 saved configurations in the combined mean. Thermostat and barostat variables were restarted; sensitivity to discarding the first 1, 2, or 5~ps after continuation was assessed separately.

Enthalpy uses $E_{\rm pot}+E_{\rm kin}+p_{\rm ext}V$, and each mixture uses the pure Al and Ti references sampled for the same total duration. Block standard errors use six or twelve 5~ps blocks for 30 or 60~ps, respectively, omitting the initial frame only in the block diagnostic. Pure-component variances are included in propagation and shared references correlate errors between compositions.

\begin{table}[htbp]
\centering
\caption{r2SCAN mixing enthalpy at 2000~K in kJ/mol of atoms. Uncertainties are propagated block SEM; endpoints vanish by definition.}
\label{tab:scan60}
\begin{tabular}{rrr}
\toprule
$x_{\rm Al}$ & 30~ps & 60~ps\\
\midrule
__ROWS__
\bottomrule
\end{tabular}
\end{table}

The maximum change in the mean curve is 0.390~kJ/mol. SEM decreases at five interior compositions and increases at four; doubled duration does not imply uniform error reduction. The numerical minimum is at $x_{\rm Al}=0.6$, but its difference from $x_{\rm Al}=0.5$ is $-0.081\pm0.263$~kJ/mol with shared-reference covariance included. The two sampled minima are not resolved statistically. The mean absolute discrepancy from the nine provisional calorimetric points is 1.468~kJ/mol.

The 30 and 60~ps estimates are nested and cannot be treated as independent means. Comparisons of disjoint 30~ps segments show no composition with a Benjamini--Hochberg-adjusted $q<0.05$ across nine compositions; this does not establish stationarity or independent blocks. At $x_{\rm Al}=0.6$, increasing the block length from 5 to 10~ps changes SEM from 0.200 to 0.267~kJ/mol. Omitting the first 1, 2, or 5~ps of continuation changes the mixing enthalpy by at most 0.041, 0.043, or 0.085~kJ/mol, respectively. Full diagnostics are retained in \texttt{r2scan\_enthalpy\_60ps.json}.

'''.replace('__ROWS__','\n'.join(rows))
rep(r'\section{Updated density at 1873 K}',section+r'\section{Updated density at 1873 K}')
rep(r'\section{Updated pure-Ti viscosity}',r'\section{PBE viscosity and sampling diagnostics}'+'\n'+r'''The main article includes only the PBE 15k pure-Ti estimates below and the three new alloy estimates with $N=15$. The earlier 5k transport values are retained here only for the historical diagnostic; they are absent from the main viscosity figure.

\subsection{Pure-Ti export and historical comparison}''')
rep('Only two PBE records change in the new STACIE export:', 'In the 27 October STACIE export, only two PBE records changed:')
rep('The comparison below uses the latest 5k ensembles already incorporated in the article,', 'The historical comparison below uses the latest archived 5k ensembles,')
rep('The 1873~K alloy viscosities and pure Ti at 1941 and 2100~K remain 5k. Additional 15k alloy calculations are not included until complete. At 1873~K, the new pure-Ti estimate and the \\rscan{} result are nearly equidistant from the provisional Ishikawa correlation, at approximately $-7\\%$ and $+7\\%$. This replaces the earlier statement that PBE was closer at that temperature.',
r'At 1873~K, the PBE pure-Ti estimate and the \rscan{} result are nearly equidistant from the provisional Ishikawa correlation, at approximately $-7\%$ and $+7\%$. PBE pure-Ti points at 1941 and 2100~K are excluded from the main figure because no 15k estimates are available for these states.')
batch=json.loads((ROOT/'audit/oct27/pbe15k_viscosity_iterative/collected/analysis/batch_0015/summary.json').read_text())
rows=[]
for r in batch['points']:
    cells=[f"${r['windows'][w]['eta_mPa_s']:.3f}\\pm{r['windows'][w]['eta_std_mPa_s']:.3f}$" for w in ['40ps','first20ps','second20ps']]
    rows.append(f"{r['x_Al']:.2f} & "+' & '.join(cells)+r'\\')
section=r'''\subsection{Alloy estimates with 15 replicas}
At 1873~K, the new PBE 15k ensembles contain 15 replicas for each of $x_{\rm Al}=0.20$, 0.45, and 0.80. The 512-atom cells contain 103, 231, and 410 Al atoms, respectively. After 20~ps NPT equilibration at 1~bar, independent velocity draws accompany snapshots from a continuing NPT reservoir separated by 5~ps. Each replica has 1~ps NVT equilibration and 40~ps production at its fixed starting volume. The step and stress-recording interval are 1~fs. Thermostat/barostat damping times during preparation are 0.1/0.5~ps, and the production thermostat damping time is 0.2~ps. Model identities are verified by the source and frozen-export hashes in \texttt{pbe15k\_viscosity\_protocol.json}.

\begin{table}[htbp]
\centering
\caption{PBE 15k alloy viscosity for $N=15$ replicas at 1873~K, in mPa\,s. The main figure uses the full 40~ps windows. Errors are conditional estimator standard errors.}
\label{tab:alloyN15}
\begin{tabular}{rrrr}
\toprule
$x_{\rm Al}$ & Full 40~ps & First 20~ps & Second 20~ps\\
\midrule
__ROWS__
\bottomrule
\end{tabular}
\end{table}

The frozen snapshot is \texttt{batch\_0015}; later partial replicas are not included. At $x_{\rm Al}=0.20$, the accumulated 40~ps estimate increased by 1.52\% from $N=10$ to 15, but the range across $N=5,10,15$ remains 12.28\%. At 0.45, the first-20~ps estimate is 4.30\% below the full estimate, and the effective spectral sample count is about 52. At 0.80, the full-window cutoff diagnostic passes the working threshold, but the second-window criterion $z=2.036$ remains slightly above 2; a separate fit to replicas 11--15 has a substantially poorer cutoff diagnostic. This is not sufficient evidence of physical drift.

No composition satisfies the full prespecified stopping rule: it requires at least 30 replicas and stable estimates and spectral diagnostics on two consecutive checks. The $N=15$ results are therefore finite-ensemble estimates with ongoing convergence checks, not fully converged transport coefficients. Windows and accumulated batches overlap and are statistically correlated; coincidence with an earlier checkpoint is not a stopping criterion. Complete estimates and diagnostic fields are in \texttt{pbe15k\_viscosity\_N15.json}.

'''.replace('__ROWS__','\n'.join(rows))
rep(r'\section{Implications and reproducibility}',section+r'\section{Implications and reproducibility}')
rep('The current replacement is limited to the rows labeled 15k in Table~\\ref{tab:coverage}; remaining property calculations will require their own updates.',
r'The main physical-property and DFT-benchmark allocations are distinct, as recorded in Table~\ref{tab:coverage}; the benchmark uses 20k and must not be interpreted as a direct error estimate for every archived property trajectory.')
rep('current 23-structure metrics','current 30-structure metrics')
rep('The complete checkpoint hashes identify the same 15k model across inference, phonons, the new density/enthalpy MD, and coexistence.',
'The complete checkpoint hashes identify 20k for current benchmark inference and 15k for the updated phonons, density/enthalpy MD, coexistence, and viscosity calculations.')
rep('No DFT labeling or training run was added for this manuscript revision.',
r'\texttt{revision\_oct29\_sources.json} records the current benchmark, extended enthalpy, and $N=15$ viscosity inputs. No DFT labeling or training run was added for this manuscript revision.')
(MAN/'supporting_materials.tex').write_text(s,encoding='utf-8')
print('SI updated to benchmark n=30, r2SCAN 60 ps, and PBE alloy N=15.')
