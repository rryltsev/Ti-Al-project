from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
MAN=ROOT/'manuscript'
def change(name,old,new):
    p=MAN/name;t=p.read_text(encoding='utf-8');assert old in t,old[:100]
    p.write_text(t.replace(old,new),encoding='utf-8')

p=MAN/'sections/results.tex';t=p.read_text(encoding='utf-8')
start=t.index(r'\begin{figure*}[t]',t.index('The mixing enthalpy gives'))
end=t.index(r'\end{figure*}',start)+len(r'\end{figure*}')
t=t[:start]+r'''\begin{figure}[tb]
\includegraphics[width=\columnwidth]{figures/mixing_enthalpy.pdf}
\caption{Liquid mixing enthalpy at 2000~K compared with calorimetric points reproduced in Ref.~\onlinecite{Witusiewicz2008}. PBE uses the 15k checkpoint. Reference points are a provisional digitization of calorimetric symbols. PBE error bars propagate the maximum block SEM over 3, 5, and 6~ps blocks; PBEsol and \rscan{} retain six-block, 5~ps estimates. Both pure-liquid references are included. These are diagnostic standard errors, not final confidence intervals. Lines connect sampled compositions.}
\label{fig:mixing}
\end{figure}'''+t[end:]
start=t.index('The retained single-checkpoint series show less negative mixing enthalpies')
end=t.index('These results quantify',start)
t=t[:start]+t[end:]
p.write_text(t,encoding='utf-8')
change('sections/methods.tex','PBE density, viscosity, Ti coexistence, Ti$_3$Al phonons, and mixing enthalpies at 1873 and 2500~K retain','PBE density, viscosity, Ti coexistence, and Ti$_3$Al phonons retain')
change('sections/methods.tex',r'''Calculations used 200 atoms at $x=0,0.1,\ldots,1$ and temperatures of 1873, 2000, and 2500~K, following the same liquid preparation and production protocol.
The 1873 and 2000~K series provide the principal comparison; the 2500~K series extends the temperature range.''',r'''Alloy-liquid calculations used 200 atoms at $x=0,0.1,\ldots,1$.
The composition-dependent density and excess-volume comparison uses 1873~K, whereas the mixing-enthalpy comparison uses 2000~K.''')
change('supporting_materials.tex','Mixing enthalpies at 1873 and 2500~K & 5k, retained'+r'\\'+'\n','')
change('supporting_materials.tex','In particular, the 1873, 2000, and 2500~K PBE enthalpy curves cannot yet be interpreted as a single-checkpoint temperature series, and the retained melting comparison does not establish the melting temperature of 15k.','The retained melting comparison does not establish the melting temperature of 15k.')

# Keep the general asset generator aligned with the current figure.
p=ROOT/'audit/oct26/manuscript_revision/prepare_assets.py';t=p.read_text(encoding='utf-8')
start=t.index("fig,axs=plt.subplots(1,2,figsize=(7,3.1)")
end=t.index("save(fig,FIG/'mixing_enthalpy')",start)+len("save(fig,FIG/'mixing_enthalpy')")
t=t[:start]+"import runpy\nrunpy.run_path(str(ROOT/'audit/oct26/remove_enthalpy_temperature/update_figure.py'),run_name='__main__')"+t[end:]
p.write_text(t,encoding='utf-8')
print('Removed temperature panel, results discussion, and obsolete scope statements.')
