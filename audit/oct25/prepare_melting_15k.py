from pathlib import Path
import json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'audit/oct26/pbe15k_melting';OUT.mkdir(parents=True,exist_ok=True)
source=Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project\MD-results\melting\beta_Ti_two-phase\pbe')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfg=json.loads((ROOT/'audit/oct25/pbe_full_enthalpy_gpu/protocol.json').read_text())
files={'initial.data':source/'data/Ti_1296atoms.data','reference_two_phase.data':source/'T1750K/cache/001/cool.lammps'}
manifest={}
for name,p in files.items():
 dest=OUT/name
 if not dest.exists():shutil.copyfile(p,dest)
 assert sha(dest)==sha(p)
 manifest[name]={'source':str(p),'sha256':sha(p)}
p={'remote':'/home6/u3129/tial-testsets/pbe15k-melting-20261026-01','checkpoint':cfg['models']['15000'],
 'temperatures_K':[1700,1750,1800,1850],'workers':{'0':[1700,1750],'1':[1850,1800]},'atoms':1296,
 'dt_ps':.001,'Tdamp_ps':.1,'Pdamp_ps':1.,'pressure_bar':0,'barostat':'aniso',
 'stages':[{'name':'equil','steps':10000},{'name':'heat','steps':20000},{'name':'cool','steps':10000},{'name':'melt','steps':50000}],
 'overheat_factor':1.5,'seed':4928459,'thermo_every':100,'dump_every':100,'restart_every':5000,
 'inputs':manifest,'allocation_wall_hours':22,
 'notes':['Original 5k geometry and four-stage LAMMPS preparation retained; positions minimized before equilibration.',
 'Heat and cool integrate the liquid group only, with global box dilation, matching original implementation. Solid group is not independently integrated in preparation.',
 'Separate LAMMPS instances and data files between stages; group membership rebuilt from lower half of z, as in original runner.',
 'All trajectories, separate stage logs, input scripts, data and restart files retained. No random fallback structural classifier.',
 'Initial temperature scan; refine bracket and/or extend borderline trajectories after interface analysis.']}
(OUT/'protocol.json').write_text(json.dumps(p,indent=2),encoding='utf-8')
(OUT/'README_RU.txt').write_text('PBE checkpoint 15k: двухфазное плавление beta-Ti.\n1296 атомов, 1700/1750/1800/1850 K, 0 bar, dt=1 fs.\nПодготовка 10+20+10 ps, наблюдение границы 50 ps. Две A100 в одной задаче; по две температуры на GPU последовательно.\nЭкспорт pt2 и сверка E/F/вириала с исходным checkpoint обязательны до длинных MD.\nВсе стадии и рестарты сохраняются; точная температура плавления определяется после структурного анализа.\n',encoding='utf-8')
print(OUT)
