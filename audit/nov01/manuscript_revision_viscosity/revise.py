"""Integrate the selected viscosity figure and revise only the main manuscript."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[3]
MAN = ROOT / 'manuscript'
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'audit/nov01/viscosity_arrhenius_article/viscosity_arrhenius_eta_T.pdf'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def read(path):
    return path.read_text(encoding='utf-8-sig')

def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)

backup = OUT / 'before_revision.zip'
assert not backup.exists(), 'Do not overwrite the pre-revision snapshot.'
with zipfile.ZipFile(backup, 'w', zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(MAN.rglob('*')):
        relative = path.relative_to(MAN)
        if path.is_file() and relative.parts[0] != 'build':
            archive.write(path, relative.as_posix())

si_before = sha(MAN / 'supporting_materials.tex')
shutil.copyfile(SOURCE, MAN / 'figures/viscosity.pdf')

methods_path = MAN / 'sections/methods.tex'
methods = read(methods_path)
methods = replace_once(methods,
    'the complete Ti temperature-dependent density series, Ti coexistence, and all viscosity points shown in the main article.',
    'the complete Ti temperature-dependent density series, and Ti coexistence.')
start = methods.index('\\subsection{Liquid viscosity}\n')
methods = methods[:start] + r'''\subsection{Liquid viscosity}
\label{sec:viscosity}

Shear viscosity was calculated from equilibrium stress fluctuations using the Green--Kubo relation.
Simulations with the PBE- and \rscan{}-based potentials used 512-atom periodic cells, NPT preparation at the target temperature and 1~bar, and fixed-volume NVT production with a 1~fs time step.
The composition dependence was evaluated at 1873~K for eleven compositions spanning pure Ti to $x_{\mathrm{Al}}=0.90$; pure Al was not included.
The temperature dependence of pure liquid Ti was examined between 1873 and 2100~K.

For an isotropic liquid, define five equivalent shear channels,
\begin{align}
 Q_1&=P_{xy},\quad Q_2=P_{xz},\quad Q_3=P_{yz},\nonumber\\
 Q_4&=\tfrac12(P_{yy}-P_{zz}),\nonumber\\
 Q_5&=\frac{P_{xx}-\tfrac12(P_{yy}+P_{zz})}{\sqrt{3}}.
\end{align}
The viscosity of a fixed-volume trajectory is
\begin{equation}
 \eta=\frac{V}{5k_{\mathrm B}T}\sum_{a=1}^{5}
 \int_0^\infty
 \left\langle\delta Q_a(0)\delta Q_a(t)\right\rangle\,dt,
 \label{eq:viscosity}
\end{equation}
where $\delta Q_a=Q_a-\langle Q_a\rangle$ and the pressure tensor includes both kinetic and configurational contributions.
The integral was estimated with STACIE (STable AutoCorrelation Integral Estimator),\cite{Toraman2025} which obtains viscosity from the low-frequency stress spectrum.
The five channels were combined with the appropriate volume and temperature prefactors for each trajectory.
Reported errors are estimator standard errors; they do not include all effects of finite sampling, preparation, finite cell size, or the potential approximation.

The pure-Ti temperature dependence was summarized by the Arrhenius form $\eta(T)=\eta_0\exp(B/T)$, fitted to the logarithms of the calculated viscosities with weights $(\eta/\sigma_\eta)^2$, where $\sigma_\eta$ is the reported standard error.
These fits describe the sampled temperature range and leave the individual viscosity estimates and their errors unchanged.
'''
methods_path.write_text(methods, encoding='utf-8')

results_path = MAN / 'sections/results.tex'
results = read(results_path)
start = results.index('\\subsection{Liquid viscosity}\n')
end = results.index('Taken together, the present benchmarks', start)
results = results[:start] + r'''\subsection{Liquid viscosity}
\label{sec:results_viscosity}

Both potentials predict a nonmonotonic composition dependence of liquid viscosity at 1873~K (Fig.~\ref{fig:viscosity}a).
Adding Al initially increases the viscosity, producing a broad maximum at intermediate composition, followed by a pronounced decrease toward the Al-rich end.
The largest sampled values are $5.02\pm0.19$~mPa\,s at $x_{\mathrm{Al}}=0.40$ for PBE and $7.19\pm0.28$~mPa\,s at $x_{\mathrm{Al}}=0.45$ for \rscan{}.
The neighboring compositions have similar viscosities, so neither curve locates a sharp maximum.
The \rscan{} values exceed those of PBE at all eleven sampled compositions.

\begin{figure*}[t]
\includegraphics[width=\textwidth]{figures/viscosity.pdf}
\caption{Liquid viscosity versus Al mole fraction at 1873~K (a) and temperature for pure Ti (b). Colored symbols show the PBE and \rscan{} estimates with estimator standard errors. Lines in (a) connect sampled compositions; solid colored curves in (b) are weighted Arrhenius fits. Black squares in (a) are values from the experimental correlations of Takeda \textit{et al.}\cite{Takeda2025} The dashed curve marked with an asterisk in (b) is a provisional transcription of the Ishikawa \textit{et al.}\cite{Ishikawa2012} correlation; the shaded band denotes its reported 10.3\% uncertainty.}
\label{fig:viscosity}
\end{figure*}

PBE is closer to the experimental alloy-viscosity correlations at four of the five compared compositions: $x_{\mathrm{Al}}=0.45$, 0.60, 0.70, and 0.80.
At $x_{\mathrm{Al}}=0.45$, the experimental reference is 5.30~mPa\,s, compared with 4.88 for PBE and 7.19 for \rscan{}.
At $x_{\mathrm{Al}}=0.80$, the corresponding values are 1.764, 1.761, and 2.528~mPa\,s.
The Al-rich state at $x_{\mathrm{Al}}=0.90$ is an exception: \rscan{} gives 1.218~mPa\,s, closer to the reference value of 1.437~mPa\,s than the PBE result of 0.951~mPa\,s.
The two lowest-composition reference values involve a small, 3~K extrapolation of the experimental correlations to 1873~K.
No alloy-viscosity reference is available below $x_{\mathrm{Al}}=0.45$, so the predicted increase on the Ti-rich side cannot be tested by this dataset.

For pure Ti, both potentials predict decreasing viscosity with increasing temperature (Fig.~\ref{fig:viscosity}b).
The PBE estimates decrease from $3.200\pm0.084$~mPa\,s at 1873~K to $2.528\pm0.068$~mPa\,s at 2100~K, while the \rscan{} values decrease from $3.70\pm0.12$ to $2.99\pm0.12$~mPa\,s.
Both series are consistent with smooth Arrhenius temperature dependences within the reported errors.
In particular, the deviations of individual \rscan{} points from the fitted curve do not resolve a change in slope or an anomalous temperature dependence.
Relative to the experimental correlation, PBE and \rscan{} lie on opposite sides at 1873~K, with deviations of approximately $-7.4\%$ and $+7.0\%$, respectively.
At 2000 and 2100~K, \rscan{} is closer to the central experimental correlation.
Thus, the advantage of PBE for most of the compared alloy compositions does not extend uniformly to pure-Ti viscosity over the studied temperature range.

''' + results[end:]
results_path.write_text(results, encoding='utf-8')

abstract_path = MAN / 'sections/abstract.tex'
abstract = replace_once(read(abstract_path),
    'the two alloy-viscosity measurements at matched compositions',
    'most of the available alloy-viscosity reference values')
abstract_path.write_text(abstract, encoding='utf-8')

conclusions_path = MAN / 'sections/conclusions.tex'
conclusions = replace_once(read(conclusions_path),
    r'''PBE also gives closer alloy viscosities at the two matched experimental compositions, $x_{\mathrm{Al}}=0.45$ and 0.80.
The 15-replica alloy estimates retain sampling limitations, and neither the pure-Ti viscosity comparison nor the experimental phonon dispersions provide a uniform ranking of the functionals.''',
    r'''PBE also gives closer alloy viscosities at four of the five compared compositions; the Al-rich state at $x_{\mathrm{Al}}=0.90$ is the exception.
For pure Ti, both models give temperature dependences consistent with Arrhenius behavior over 1873--2100~K, and \rscan{} is closer to the central experimental viscosity correlation at 2000 and 2100~K.
Neither the viscosity results across these liquid states nor the experimental phonon dispersions therefore provide a uniform ranking of the functionals.''')
conclusions_path.write_text(conclusions, encoding='utf-8')

discussion_path = MAN / 'sections/discussion.tex'
discussion = replace_once(read(discussion_path),
    r'''Viscosity depends on the amplitude and relaxation time of stress fluctuations, with additional sensitivity to density and liquid structure.''',
    r'''For pure Ti, however, \rscan{} is closer to the central experimental viscosity correlation at 2000 and 2100~K, while both models give temperature dependences consistent with smooth Arrhenius behavior over the sampled interval.
The transport comparison therefore depends on composition and temperature as well as on the reference functional.
Viscosity depends on the amplitude and relaxation time of stress fluctuations, with additional sensitivity to density and liquid structure.''')
discussion = replace_once(discussion,
    'the indirect calorimetric reference, and the pending viscosity checks',
    'the indirect calorimetric reference, and the sampling uncertainty in viscosity')
discussion_path.write_text(discussion, encoding='utf-8')

assert sha(MAN / 'supporting_materials.tex') == si_before
manifest = {
    'source_chat_title': 'Проверить гладкость вязкости Ti',
    'source_chat_id': '01a1982e-412a-7c21-a953-2a1c86ddea5b',
    'source_figure': str(SOURCE.relative_to(ROOT)),
    'figure_sha256': sha(SOURCE),
    'installed_figure_sha256': sha(MAN / 'figures/viscosity.pdf'),
    'scope': 'Main manuscript only; Supporting Materials unchanged.',
    'supporting_materials_sha256': si_before,
    'note': 'Figure selected by the user: eta(T), not log eta vs inverse T. Original estimates and errors unchanged. Smooth lines are weighted Arrhenius fits. Composition uses archived PBE 5k; the pure-Ti temperature panel uses PBE 15k. Removed the false main-text assertion that all viscosity data use 15k. Detailed provenance is deferred to a future SI revision.',
    'updated_sections': ['methods (viscosity; one allocation correction in fine-tuning)', 'results (viscosity)', 'abstract', 'discussion', 'conclusions'],
}
(OUT / 'provenance.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps(manifest, ensure_ascii=False, indent=2))
