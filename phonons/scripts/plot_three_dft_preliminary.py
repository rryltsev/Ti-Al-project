"""Overlay current, electronically converged DFT central-pair DOS (preliminary)."""
import build_pbesol_phonons as b
import numpy as np
import matplotlib.pyplot as plt
import json

sources={
    'PBE':b.ROOT/'results/PBE-matched-DFT-DPA4-20261101-01',
    'PBEsol':b.ROOT/'results/PBEsol-matched-DFT-DPA4-20261031-02',
    'r2SCAN':b.ROOT/'results/r2SCAN-matched-DFT-DPA4-20261101-01'}
out=b.ROOT/'results/three-DFT-DOS-preliminary-20261101';out.mkdir(exist_ok=False)
colors=['#16658e','#d66a19','#25896e'];cols={};sourcehash={}
grid=np.arange(-.3,9.302,.002)
plt.rcParams.update({'font.size':11,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(9,5.4))
for (xc,path),color in zip(sources.items(),colors):
    data=np.load(path/'comparison.npz')
    x,y=data['frequency_THz'],data['DOS_DFT']
    assert abs(np.trapezoid(y,x)-3)<1e-10
    ax.plot(x,y,lw=1.6,color=color,label=xc)
    cols[xc]=np.interp(grid,x,y,left=0,right=0)
    assert abs(np.trapezoid(cols[xc],grid)-3)<1e-6
    sourcehash[xc]=dict(path=str(path/'comparison.npz'),sha256=b.digest(path/'comparison.npz'))
ax.set(xlim=(0,9.15),ylim=(0,None),xlabel='Frequency (THz)',ylabel='DOS (modes / atom / THz)')
ax.set_title(r'$\alpha$-Ti | DFT PBE / PBEsol / r2SCAN | Preliminary',loc='left',pad=18)
ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
fig.text(.1,.045,r'$4\times4\times3$ cells; own DFT equilibria; $q=50^3$; $\sigma=0.05$ THz; 47.867 amu.',fontsize=9)
fig.text(.1,.012,'Reference diagnostics (PBE/r2SCAN) and numerical/supercell convergence pending.',fontsize=9,color='#8a5a16')
fig.subplots_adjust(left=.1,right=.98,top=.88,bottom=.17)
for ext in ['png','pdf']:fig.savefig(out/('three-DFT-DOS-preliminary.'+ext),dpi=220)
plt.close(fig)
b.write_csv(out/'DOS.csv',['frequency_THz',*cols],zip(grid,*cols.values()))
(out/'provenance.json').write_text(json.dumps(dict(sources=sourcehash,
    qualification='Preliminary, not convergence-validated. Each functional at its own DFT-relaxed geometry. PBE/r2SCAN reference force cancels in central-pair derivative but reference diagnostics still pending.',
    csv_note='Each native 0.002THz Gaussian DOS interpolated to common frequency grid; integrals checked within1e-6 of3.'),indent=2)+'\n')
print(out)
