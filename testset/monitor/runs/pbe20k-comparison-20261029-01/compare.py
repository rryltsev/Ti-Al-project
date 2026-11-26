"""User-requested four-model comparison; immutable n29 Test50 subset, raw PBE labels."""
from pathlib import Path
import hashlib
import html
import json
import sys

HERE = Path(__file__).resolve().parent
MON = HERE.parents[1]
WORK = MON.parents[1]
sys.path.insert(0, str(MON))
import analyze
import numpy as np

read = analyze.read
sha = analyze.sha
archive = MON / 'results/n29-7a071c141a4730cd/metrics.json'
reference = read(archive)
bundle_path = MON / 'data/triplets-73a98ec8f0b1d963.json'
bundle = read(bundle_path)
assert bundle['common_ids'] == reference['common_ids'] and len(bundle['records']) == 29
assert not bundle['problems']
cache = read(MON / 'prediction-manifest.json')
registry = read(MON / 'model_registry.json')
outputs = read(HERE / 'collected/outputs.json')
for name, digest in outputs.items():
    assert sha(HERE / 'collected' / name) == digest, name
assert read(HERE / 'collected/verification.json')['status'] == 'passed'
models = {
    'Pretrained DPA4-Neo': {'sha256': read(MON/'pretrained_registry.json')['sha256'], 'directory': MON/'predictions/pretrained'},
    'PBE 5k': {**registry['model_selection']['previous_PBE'], 'directory': MON/'predictions/PBE'},
    'PBE 15k': {**registry['models']['PBE'], 'directory': MON/'predictions/PBE-15000'},
    'PBE 20k': {**read(HERE/'registry-template.json')['models']['PBE'], 'directory': HERE/'collected/predictions/PBE'},
}
provenance = {str(p.relative_to(WORK)): sha(p) for p in [archive, bundle_path, MON/'analyze.py', MON/'model_registry.json', MON/'prediction-manifest.json', HERE/'collected/outputs.json']}
for row in bundle['records']:
    assert sha(WORK/'testset/TiAl-test50-v1'/row['metadata']['poscar']) == row['metadata']['poscar_sha256']
    assert all(row['labels'][xc]['scf_converged'] and row['labels'][xc]['ionic_calculations'] == 1 for xc in analyze.XC)
metrics = {}
reproduction = {}
for name, model in models.items():
    predictions = {}
    for row in bundle['records']:
        sid = row['metadata']['structure_id']
        path = model['directory'] / (sid + '.json')
        prediction = read(path)
        digest = sha(path)
        if name != 'PBE 20k':
            assert cache[path.relative_to(MON).as_posix()] == digest
        assert prediction['model_sha256'] == model['sha256']
        assert prediction['structure_id'] == sid and prediction['symbols'] == row['symbols']
        assert prediction['poscar_sha256'] == row['metadata']['poscar_sha256']
        assert prediction['energy_offset_applied_eV'] == 0
        predictions[sid] = prediction
        provenance[str(path.relative_to(WORK))] = digest
    values, groups = analyze.compare(bundle['records'], {'PBE': predictions}, 'PBE')
    metrics[name] = groups
    if name == 'PBE 20k':
        sweep = WORK/'audit/oct25/pbe_checkpoint_sweep'
        old_rows = read(sweep/'rows.json')
        old_ids = {r['id']:r['i'] for r in old_rows if r['split']=='test20'}
        old_geometry=np.load(sweep/'geometries.npz',allow_pickle=False)
        by_sid={r['metadata']['structure_id']:r for r in bundle['records']}
        for sid,i in old_ids.items():
            row=by_sid[sid]
            assert np.allclose(np.asarray(row['fractional_positions'])@row['cell_A'],old_geometry[f'coords_{i}'],rtol=0,atol=1e-12)
            assert np.array_equal(row['cell_A'],old_geometry[f'cell_{i}'])
            assert [['Al','Ti'][int(t)] for t in old_geometry[f'types_{i}']]==row['symbols']
        old_predictions = {}
        archived_manifest = read(sweep/'full-collection.json')['files']
        for task in range(24,32):
            f = sweep/f'result-{task}.npz'
            assert sha(f) == archived_manifest[f.name]
            data = np.load(f,allow_pickle=False)
            old_predictions.update({k:data[k] for k in data.files})
        max_e=max_f=max_v=0.0
        for sid,i in old_ids.items():
            pred=predictions[sid]
            max_e=max(max_e,abs(pred['energy_eV']-float(old_predictions[f'energy_{i}'])))
            max_f=max(max_f,float(np.max(np.abs(np.asarray(pred['forces_eV_A'])-old_predictions[f'forces_{i}']))))
            max_v=max(max_v,float(np.max(np.abs(np.asarray(pred['virial_eV'])-old_predictions[f'virial_{i}']))))
        # Float32 model: CPU reductions can vary slightly between runs/thread layouts.
        # Bounds are negligible relative to Test50 errors; preserve actual differences.
        assert max_e<1e-4 and max_f<1e-5 and max_v<1e-4
        reproduction={'n_structures':len(old_ids),'geometry_check':'cell/types identical; coordinates within1e-12A','energy_max_difference_eV':max_e,'force_max_difference_eV_A':max_f,'virial_max_difference_eV':max_v,'tolerances_eV_eV_A_eV':[1e-4,1e-5,1e-4],'qualification':'Numerical reproduction, not bitwise equality; same checkpoint SHA256.'}
    energy_errors = np.array([1000*(v['e_pred']-v['e_ref']) for v in values])
    assert np.isclose(np.sqrt(np.mean(energy_errors**2)), groups['all']['energy_meV_atom']['rmse'], rtol=1e-13)
assert metrics['PBE 15k'] == reference['metrics']['PBE']
assert metrics['Pretrained DPA4-Neo'] == reference['pretrained_metrics']['PBE']
for file, digest in read(HERE/'monitor-before-hashes.json').items():
    assert sha(Path(file)) == digest, 'Active monitor artifact changed'
result = {
    'common_ids': bundle['common_ids'], 'n_structures': 29,
    'reference_functional': 'PBE', 'energy_definition': reference['energy_definition'],
    'force_definition': reference['force_definition'], 'coverage': reference['coverage'],
    'models': {name: {k:str(v) if isinstance(v,Path) else v for k,v in model.items()} for name,model in models.items()},
    'metrics': metrics, 'source_sha256': provenance,
    '20k_reproduction_of_archived20': reproduction,
    'qualification': 'Same 29 validated Test50 triplets: 20 liquid, 3 hcp, 3 bcc, 3 gamma. Raw energies; no shifts or fitting. Descriptive 95% bootstrap intervals resample entire structures. Checkpoint comparison occurs after Test50 inspection; it is not blind model selection. Active PBE15k monitor remains unchanged.',
}
analyze.write(HERE/'comparison.json', result)
props = [('energy_meV_atom','Энергия, meV/атом'), ('force_eV_A','Силы, eV/Å'), ('stress_GPa','Напряжения, GPa')]
rows=[]
for name, groups in metrics.items():
    cells=[html.escape(name)]
    for prop,_ in props:
        m=groups['all'][prop]
        cells += [f"{m['mae']:.5g}",f"{m['rmse']:.5g}",f"[{m['rmse_ci95'][0]:.5g}; {m['rmse_ci95'][1]:.5g}]"]
    rows.append('<tr>'+''.join('<td>'+v+'</td>' for v in cells)+'</tr>')
header='<tr><th rowspan="2">Модель</th>'+''.join(f'<th colspan="3">{title}</th>' for _,title in props)+'</tr><tr>'+('<th>MAE</th><th>RMSE</th><th>95% интервал RMSE</th>'*3)+'</tr>'
body='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Test50: PBE checkpoint comparison</title><style>body{font:16px system-ui;margin:32px;max-width:1400px}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{border:1px solid #cbd5e1;padding:10px;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#edf2f7}p{line-height:1.5}</style><h1>Test50: pretrained DPA4-Neo и PBE 5k / 15k / 20k</h1><p>Общие 29 структур с валидированными PBE/PBEsol/r2SCAN: 20 жидких, по 3 hcp/bcc/γ. Все модели сравниваются с одними и теми же метками PBE. Исходные энергии без сдвигов; силы — все декартовы компоненты; напряжения — шесть независимых компонент.</p><table>'''+header+''.join(rows)+'''</table><p>Интервалы описательные: 2000 bootstrap-выборок целых структур. Сравнение checkpoint выполнено после просмотра Test50; это не слепой выбор модели. Полные 50 DFT-троек ещё не завершены. Активная модель мониторинга PBE 15k сохранена.</p><p><a href="comparison.json">Полные метрики, группы Al/Ti и семейств, ID и SHA256</a></p></html>'''
(HERE/'comparison.html').write_text(body,encoding='utf-8')
print(json.dumps({name:{prop:{k:g['all'][prop][k] for k in ('mae','rmse','rmse_ci95')} for prop,_ in props} for name,g in metrics.items()},indent=2))
