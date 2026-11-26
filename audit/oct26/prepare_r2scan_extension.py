"""Freeze last archived r2SCAN frames for an additional 30 ps per composition."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
from ase import units
from ase.io import read, write

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'r2scan_enthalpy_extend30k'
SOURCE = Path(r'C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project')
BASE = SOURCE / 'results-scripts-logs/models_check_wsl2/results/mixing_enthalpy/dpa4/neo/finetune/AlTi/2000K/scan'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')

OUT.mkdir(exist_ok=False)
model = json.loads((ROOT / 'audit/oct25/all_xc_audit/protocol.json').read_text())['models']['r2SCAN-23200']
assert sha(model['local_path']) == model['sha256']
states, baseline, checks = [], {}, []
for i in range(11):
    x = i / 10
    name = 'Ti' if i == 0 else 'Al' if i == 10 else f'Al{x:.1f}Ti{1-x:.1f}'
    source = BASE / (name + '_2000K.traj')
    frames = read(source, index=':')
    assert len(frames) == 301
    a = frames[-1]
    assert len(a) == 200 and a.get_chemical_symbols().count('Al') == 20 * i
    assert a.has('momenta') and np.isfinite(a.get_momenta()).all()
    assert np.max(abs(np.tril(a.cell.array, -1))) < 1e-12
    assert np.linalg.norm(a.get_momenta().sum(axis=0)) < 1e-6
    filename = name + '.traj'
    write(OUT / filename, a)
    saved = read(OUT / filename)
    for field in ['positions', 'momenta', 'numbers']:
        assert np.array_equal(a.arrays[field], saved.arrays[field])
    assert np.array_equal(a.cell.array, saved.cell.array)
    assert np.array_equal(a.get_forces(), saved.get_forces())
    rows = []
    for j, f in enumerate(frames):
        n, volume = len(f), f.get_volume()
        energy, kinetic = f.get_potential_energy(), f.get_kinetic_energy()
        rows.append(dict(step=j * 100, time_ps=j * .1,
                         Epot_eV_atom=energy/n, Ekin_eV_atom=kinetic/n,
                         H_external_eV_atom=(energy+kinetic+units.bar*volume)/n,
                         T_K=f.get_temperature(), V_A3_atom=volume/n,
                         rho_g_cm3=float(f.get_masses().sum()/volume*1.66053906660)))
    baseline[name] = dict(source=str(source), sha256=sha(source), x_Al=x, rows=rows)
    states.append(dict(composition=name, x_Al=x, initial_file=filename,
                       initial_sha256=sha(OUT / filename), initial_source=str(source),
                       source_sha256=sha(source), initial_frame=300,
                       old_production_step=30000, bulk_modulus_GPa=110*(1-x)+76*x))
    checks.append(dict(composition=name, frames=301, atoms=len(a), T_K=a.get_temperature(),
                       momenta_sum=a.get_momenta().sum(axis=0).tolist(),
                       exact_positions_cell_momenta=True))

save('baseline.json', baseline)
cfg = dict(id='r2scan-enthalpy-extend30k-20261026-01',
           remote='/home6/u3129/tial-testsets/r2scan-enthalpy-extend30k-20261026-01',
           model=model, states=states, T_K=2000, P_bar=1, dt_fs=1, ttime_fs=25,
           additional_steps=30000, equil_steps=0, thermo_interval=10,
           trajectory_interval=100, n_atoms=200, allocation_wall_hours=12,
           baseline_sha256=sha(OUT / 'baseline.json'),
           initialization='Exact final archived positions, cell and momenta. No velocity reseeding, heating, minimization or equilibration. Thermostat/barostat variables reset because they were not archived.',
           integrator='ASE MelchionnaNPT; fraction_traceless=0; pfactor=(25 fs)^2 B(x)',
           statistics='301 original frames plus 300 new frames; exclude duplicate boundary. Block diagnostics use original frames 1..300 and new frames 1..300, twelve 5 ps blocks. Store new-segment restart sensitivity separately.',
           enthalpy='H=(Epot+Ekin+P_external V)/N; deltaH=H(x)-x H(Al)-(1-x) H(Ti)')
save('protocol.json', cfg)
save('preparation-verification.json', dict(status='passed', checks=checks,
     model_sha256=model['sha256'], protocol_sha256=sha(OUT / 'protocol.json')))
for src, dst in [('r2scan_extension_worker.py', 'worker.py'),
                 ('summarize_r2scan_extension.py', 'summarize.py'),
                 ('r2scan_extension.sbatch', 'job.sbatch')]:
    shutil.copyfile(HERE / src, OUT / dst)
(OUT / 'README_RU.txt').write_text(
    'Продолжение энтальпии смешения r2SCAN при 2000 K.\n'
    '11 составов x(Al)=0, 0.1, ..., 1; 200 атомов; NPT, 1 бар; шаг 1 фс.\n'
    'Контрольная точка r2SCAN 23200, та же модель, что в исходной серии.\n'
    'Начало: кадр 300 (30000 шагов production) каждой исходной траектории.\n'
    'Дополнительно по 30000 шагов (30 пс); всего по 60000 шагов (60 пс).\n'
    'Координаты, ячейка и импульсы сохранены точно. Скорости не генерируются заново.\n'
    'Внутренние переменные термостата/баростата не сохранились и инициализируются заново.\n'
    'Дополнительная термализация отсутствует. Все 30000 новых шагов относятся к production.\n'
    'Одна A100, четыре CPU, 32 GiB, лимит 12 часов; составы выполняются последовательно.\n'
    'Перед MD: проверка CPU/GPU и соответствия энергии, сил, напряжений архивным кадрам.\n'
    'Термодинамика каждые 10 шагов; траектория каждые 100; первые 100 шагов входят в production.\n'
    'После серии автоматически рассчитывается статистика старых и новых участков.\n'
    'Исходные файлы сохраняются без изменений; повтор граничного кадра в среднем исключается.\n',
    encoding='utf-8')
print(json.dumps(dict(prepared=str(OUT), states=len(states), additional_steps=330000,
                     protocol_sha256=sha(OUT/'protocol.json'))))
