from pathlib import Path
import sys,json,csv
OUT=Path(__file__).parent
sys.path.insert(0,str(OUT/'deps'))
import numpy as np
from ase.io.trajectory import Trajectory
from ase import units
ROOT=Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project')
BASE=ROOT/'results-scripts-logs/models_check_wsl2/results/mixing_enthalpy/dpa4/neo/finetune/AlTi'
CONV=96.48533212331
def block_stats(a,block=50):
    # 100 MD steps between saved frames, dt=1 fs: block=50 is 5 ps.
    a=a[1:];n=len(a)//block;b=a[:n*block].reshape((n,block)).mean(axis=1)
    return float(b.std(ddof=1)/np.sqrt(n)) if n>1 else None
rows=[]
for T in [1873,2000,2500]:
  for model in ['pbe','scan','sol']:
    folder=BASE/f'{T}K'/model
    source_summary=json.loads((folder/'summary.json').read_text())
    group=[]
    for p in sorted(folder.glob('*.traj')):
      traj=Trajectory(str(p));a0=traj[0];N=len(a0);x=float(sum(a0.numbers==13)/N)
      E=[];U=[];P=[];V=[];Temp=[];origH=[];Hext=[];dens=[]
      frac=[];cells=[]
      for a in traj:
        e=a.get_potential_energy()/N;u=a.get_total_energy()/N;v=a.get_volume()/N
        s=a.get_stress(include_ideal_gas=True);pressure=-s[:3].mean()
        E.append(e);U.append(u);P.append(pressure/units.GPa);V.append(v);Temp.append(a.get_temperature());dens.append(a.get_masses().sum()/a.get_volume()*1.66053906660)
        origH.append((u+v*(s[:3].mean()/units.GPa)*6.242e-3)*96.485)
        Hext.append((u+units.bar*v)*CONV)
        frac.append(a.get_scaled_positions(wrap=True));cells.append(a.cell.array)
      E,U,P,V,Temp,origH,Hext,dens=map(np.array,[E,U,P,V,Temp,origH,Hext,dens])
      ff=np.array(frac);steps=np.diff(ff,axis=0);steps-=np.round(steps)
      dr=np.einsum('tni,tij->tnj',steps,np.array(cells)[1:]);disp=np.cumsum(dr,axis=0);disp-=disp.mean(axis=1)[:,None,:]
      msd=float(np.mean(np.sum(disp[-1]**2,axis=1)))
      comp=p.stem.rsplit('_',1)[0];raw=source_summary[comp]
      r=dict(T=T,dft=model,x_Al=x,N=N,n_frames=len(E),source=p.relative_to(ROOT).as_posix(),T_mean=float(Temp.mean()),E_pot_eV=float(E.mean()),E_tot_eV=float(U.mean()),rho=float(dens.mean()),rho_sem_5ps=block_stats(dens),H_reported=raw['H_kJ_per_mol'],H_legacy_recomputed=float(origH.mean()),H_external_pressure=float(Hext.mean()),H_external_sem_5ps=block_stats(Hext),E_pot_sem_5ps_kJmol=block_stats(E)*CONV,P_mean_GPa=float(P.mean()),P_sem_5ps_GPa=block_stats(P),V_atom_A3=float(V.mean()),Epot_half_drift_kJmol=float((E[len(E)//2:].mean()-E[:len(E)//2].mean())*CONV),rho_half_drift=float(dens[len(E)//2:].mean()-dens[:len(E)//2].mean()),T_half_drift=float(Temp[len(E)//2:].mean()-Temp[:len(E)//2].mean()),MSD_30ps_A2=msd,cell_aspect_ratio=float(np.max(np.linalg.norm(cells[-1],axis=1))/np.min(np.linalg.norm(cells[-1],axis=1))),Epot_block_sem_kJmol={str(b):block_stats(E,b)*CONV for b in [5,10,20,30,50]})
      group.append(r)
    endpoints={r['x_Al']:r for r in group if r['x_Al'] in [0,1]}
    for r in group:
      x=r['x_Al'];a=endpoints[1];t=endpoints[0]
      r['dH_reported']=r['H_reported']-x*a['H_reported']-(1-x)*t['H_reported']
      r['dH_external_pressure']=r['H_external_pressure']-x*a['H_external_pressure']-(1-x)*t['H_external_pressure']
      r['dE_pot_kJmol']=(r['E_pot_eV']-x*a['E_pot_eV']-(1-x)*t['E_pot_eV'])*CONV
      r['dH_external_sem_5ps']=0 if x in [0,1] else float(np.sqrt(r['H_external_sem_5ps']**2+x*x*a['H_external_sem_5ps']**2+(1-x)**2*t['H_external_sem_5ps']**2))
      r['dH_change']=r['dH_external_pressure']-r['dH_reported']
      r['Vm_cm3mol']=r['V_atom_A3']*.602214076
      r['Vex_cm3mol']=(r['V_atom_A3']-x*a['V_atom_A3']-(1-x)*t['V_atom_A3'])*.602214076
    rows.extend(group)
    mins={field:min(group,key=lambda r:r[field]) for field in ['dH_reported','dH_external_pressure','Vex_cm3mol']}
    print(T,model,'H minima',[(k,round(v[k],3),v['x_Al']) for k,v in mins.items()], 'maxHchange',round(max(abs(r['dH_change']) for r in group),3),'P range',tuple(round(f(r['P_mean_GPa'] for r in group),3) for f in [min,max]),'MSD range',tuple(round(f(r['MSD_30ps_A2'] for r in group),1) for f in [min,max]),flush=True)
(OUT/'thermo_reprocessed.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
with (OUT/'thermo_reprocessed.csv').open('w',newline='',encoding='utf8') as f:
    fields=[k for k in rows[0] if k!='Epot_block_sem_kJmol'];w=csv.DictWriter(f,fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
print('REPRODUCE LEGACY H max error',max(abs(r['H_legacy_recomputed']-r['H_reported']) for r in rows))
print('LARGEST CORRECTIONS',[(r['T'],r['dft'],r['x_Al'],round(r['dH_change'],3)) for r in sorted(rows,key=lambda r:abs(r['dH_change']),reverse=True)[:12]])
