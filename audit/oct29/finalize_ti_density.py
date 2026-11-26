"""Verify the finished single allocation and write final density provenance/table."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];RUN=ROOT/'audit/oct29/pbe15k_ti_density_temperature'
OUT=RUN/'analysis';DATA=ROOT/'manuscript/supporting_data'
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cfg=load(RUN/'protocol.json');status=load(RUN/'status.json');registry=load(OUT/'density_registry_15k.json')
summary=status['metadata']['summary.json'];expected=[s['id'] for s in cfg['states']]
assert summary['status']=='completed' and summary['completed']==expected and not summary['failed']
assert registry['completed_new']==expected and len(expected)==15
assert '20964836|COMPLETED|0:0|09:52:26|' in (RUN/'scheduler-latest.txt').read_text()
manifest=load(RUN/'collection-manifest.json')
assert len(manifest)==109 and all(sha(RUN/'collected'/n)==v['sha256'] for n,v in manifest.items())
for state in cfg['states']:
    folder=RUN/'collected/results'/state['id'];m=load(folder/'summary.json')
    assert m['job_id']=='20964836' and m['model_sha256']==cfg['models']['15000']['sha256']
    assert m['worker_sha256']==sha(RUN/'workflow.py') and m['common_sha256']==sha(RUN/'md_common.py')
    for stage,key in [('equilibration','equil_steps'),('production','production_steps')]:
        rows=[json.loads(s) for s in (folder/(stage+'-thermo.jsonl')).read_text().splitlines()]
        assert [r['step'] for r in rows]==list(range(0,state[key]+1,10))
points=sorted([r for r in registry['points'] if r['kind']=='homogeneous_NPT' and r['x_Al']==0],key=lambda r:r['T_K'])
assert len(points)==17
structure=load(OUT/'beta_structural_qc.json')
table=[r'\begin{table}[htbp]',r'\centering\small',
 r'\caption{Completed PBE 15k pure-Ti densities. The phase denotes the starting cell; $\beta^\ast$ at 1900 K melted before production. All averages use 30 ps production. SEM is a block standard error; $\bar P$ is the logged mean pressure, with a target of 0.1 MPa. MSD removes affine cell change and center-of-mass motion.}',
 r'\label{tab:ti_density_temperature15k}',r'\begin{tabular}{lrrrrr}',r'\toprule',
 r'Start & $T$ (K) & $\rho$ (g/cm$^3$) & SEM (g/cm$^3$) & $\bar P$ (MPa) & MSD$_{10\,\mathrm{ps}}$ (\AA$^2$)\\',r'\midrule']
for p in points:
    phase={'alpha':r'$\alpha$','beta':r'$\beta$','liquid':'Liquid'}[p['phase_initial']]
    if p['T_K']==1900:phase=r'$\beta^\ast$'
    table.append(f"{phase} & {p['T_K']} & {p['rho']['mean']:.5f} & {p['rho']['sem']:.5f} & {p['qc']['P_GPa']['mean']*1000:.2f} & {p['qc']['MSD_10ps_A2']:.2f}"+r'\\')
table += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
(DATA/'ti_density_temperature_final.tex').write_text('\n'.join(table)+'\n',encoding='utf-8')
shutil.copyfile(OUT/'beta_structural_qc.json',DATA/'pbe15k_ti_beta_structural_qc.json')
result=dict(job_id='20964836',slurm_state='COMPLETED',exit_code='0:0',elapsed='09:52:26',gpu='NVIDIA A100-PCIE-40GB',
    completed_points=15,primary_Ti_points_including_prior=17,registry_records=len(registry['points']),collected_files=len(manifest),
    protocol_sha256=sha(RUN/'protocol.json'),model_sha256=cfg['models']['15000']['sha256'],
    verified_utc=datetime.now(timezone.utc).isoformat(),beta_structural_qc_sha256=sha(OUT/'beta_structural_qc.json'),
    primary_points=points,figure=load(OUT/'figure_state.json')['article_figure_pdf'])
(OUT/'final_summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
report=['Итог расчёта плотности чистого Ti, PBE 15k',
 'Slurm 20964836: COMPLETED, exit 0:0, 09:52:26 на одной A100.',
 'Завершены все 15 новых точек; с ранее готовыми жидкостями 1873/2000 K — 17 точек Ti.',
 'Проверены 109 файлов, SHA256 модели/протокола/кода, полные стадии equilibration и production.',
 'Все исходные точки 5k сохранены на рис. 3. Новые расчёты не запускались.','',
 'T (K) | начальная фаза | плотность ± SEM (г/см³) | среднее P (МПа) | MSD10 (Å²)']
for p in points:report.append(f"{p['T_K']} | {p['phase_initial']} | {p['rho']['mean']:.6f} ± {p['rho']['sem']:.6f} | {p['qc']['P_GPa']['mean']*1000:.3f} | {p['qc']['MSD_10ps_A2']:.3f}")
report+=['','Контроль фаз:',
 'β-инициализация при 1800 K: сохраняется кристаллический порядок; доля qbar6>0.25 за последние 10 пс — 0.9684.',
 'β-инициализация при 1900 K: уже к 1 пс equilibration теряется кристаллический порядок; production жидкая.',
 'Доля qbar6>0.25 за последние 10 пс при 1900 K — 0.0004; MSD10 — 30.30 Å².',
 'Это поведение однородной ячейки не задаёт равновесную температуру плавления.',
 '', 'Ограничения:',
 'SEM учитывает конечную выборку, но не модельную ошибку, размер ячейки, подготовку, смещение давления и период записи.',
 'Среднее давление новых точек — 1.48–9.71 МПа при цели 0.1 МПа; полные интервалы приведены в реестре.',
 'Запись 5k и 15k различается, поэтому разность плотностей нельзя приписывать только чекпойнту.',
 '', 'Рис. 3: '+result['figure'],
 'Полный реестр (44 записи, включая сплавы, поздние окна сосуществования и 0 K): density_registry_15k.json/.csv.']
(OUT/'FINAL_REPORT_RU.txt').write_text('\n'.join(report)+'\n',encoding='utf-8')
# Explicitly state the melted point in the figure caption without changing its preparation marker.
path=ROOT/'manuscript/sections/results.tex';txt=path.read_text(encoding='utf-8')
anchor='these symbols specify preparation, not persistence of the crystalline phase.'
addition=' The PBE 15k cell initialized as $\\beta$-Ti at 1900~K is liquid during production.'
assert txt.count(anchor)==1
if addition not in txt:txt=txt.replace(anchor,anchor+addition);path.write_text(txt,encoding='utf-8')
print('Verified complete 15-point allocation and 17-point Ti density dataset.')
