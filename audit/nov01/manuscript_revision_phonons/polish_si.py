"""Keep the historical provenance table with its discussion, before references."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
p=ROOT/'manuscript/supporting_materials.tex'
s=p.read_text(encoding='utf-8')
start=s.index(r'\begin{table}',s.index(r'\section{Implications and reproducibility}'))
end=s.index(r'\end{table}',start)+len(r'\end{table}')
table=s[start:end]
table=table.replace('[htbp]','[!htbp]').replace('SHA-256 prefixes for principal inputs; full hashes are retained in the machine-readable provenance.',
    'SHA-256 prefixes for historical checkpoint-selection inputs; full hashes and the new matched-phonon provenance are retained separately in the machine-readable manifests.')
table=table.replace('DFT reference POSCAR &','Archived PBE DFT POSCAR &').replace('DFT reference DOS &','Archived PBE DFT DOS &')
s=s[:start]+s[end:]
heading=r'\section{Implications and reproducibility}'
s=s.replace(heading,heading+'\n\n'+table+'\n')
start=s.index(r'\end{table}',s.index(heading))+len(r'\end{table}')
end=s.index(r'\bibliographystyle',start)
s=s[:start]+r'''

The historical analysis supports selecting 15k through a joint training/validation-and-phonon criterion; it does not establish general overfitting or monotonic phonon deterioration after 5k. Model allocations remain explicit in Table~\ref{tab:coverage}. In particular, the 20k DFT benchmark is not a direct error estimate for every property trajectory.

The manifests \texttt{revision\_oct26\_sources.json}, \texttt{revision\_oct27\_sources.json}, and \texttt{revision\_oct29\_sources.json} preserve the preceding calculation provenance and full model hashes. The new matched phonon comparison is recorded separately in \texttt{revision\_nov01\_phonons\_sources.json}. Earlier checkpoint, composition, and volume diagnostics retain their historical identities. The separate pure-Ti viscosity export retains the provenance limitation described in Sec.~S11.1. No model was retrained for this update, and the fixed test-set and thermodynamic results were not changed by integrating the new phonon references.

'''+s[end:]
for name in ['revision_oct26_sources.json','revision_oct27_sources.json','revision_oct29_sources.json','revision_nov01_phonons_sources.json']:
    s=s.replace(r'\texttt{'+name.replace('_',r'\_')+'}',r'\path{'+name+'}')
s=s.replace(r'\bibliographystyle{unsrtnat}',r'\renewcommand{\bibsection}{\par\medskip\noindent{\large\bfseries References}\par\smallskip}'+'\n'+r'\bibliographystyle{unsrtnat}')
p.write_text(s,encoding='utf-8')
