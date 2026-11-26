"""Read-only sensitivity analysis of the manuscript's frozen common20 subset."""
from pathlib import Path
import json, math, statistics, hashlib, shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
bundle = read(HERE / 'test20_bundle.json')
rows = []
for rec in bundle['records']:
    m = rec['metadata']
    sid = m['structure_id']
    row = {'id': sid, 'x_Ti': m['n_Ti']/m['n_atoms'],
           'volume_A3_atom': m['volume_per_atom_A3']}
    for role in ['PBE', 'pretrained']:
        pred = read(ROOT / 'testset/monitor/predictions' / role / (sid+'.json'))
        assert pred['poscar_sha256'] == m['poscar_sha256']
        row[role] = 1000*(pred['energy_eV']-rec['labels']['PBE']['energy_eV'])/m['n_atoms']
    rows.append(row)

def stats(rr, role):
    e = [r[role] for r in rr]
    bias = statistics.mean(e)
    return {'n': len(e), 'mae': statistics.mean(map(abs,e)),
            'rmse': math.sqrt(statistics.mean([x*x for x in e])),
            'bias': bias, 'median': statistics.median(e),
            'centered_rmse_diagnostic_only': math.sqrt(statistics.mean([(x-bias)**2 for x in e])),
            'n_positive': sum(x>0 for x in e)}

without = [r for r in rows if r['id'] != 'TiAl-test-009']
summary = {name: {role: stats(rr,role) for role in ['PBE','pretrained']}
           for name,rr in [('all20',rows),('without009',without),
                           ('Ti_rich_xTi_ge_half', [r for r in rows if r['x_Ti']>=.5]),
                           ('Al_rich_xTi_lt_half', [r for r in rows if r['x_Ti']<.5])]}
summary['leave_one_out'] = {r['id']: {role: stats([v for v in rows if v['id']!=r['id']],role)
                                    for role in ['PBE','pretrained']} for r in rows}
err009 = next(r['PBE'] for r in rows if r['id']=='TiAl-test-009')
summary['009_fraction_squared_error'] = err009**2 / sum(r['PBE']**2 for r in rows)
summary['fine_tuned_worse_absolute_count'] = sum(abs(r['PBE'])>abs(r['pretrained']) for r in rows)
summary['per_configuration_meV_atom'] = rows

out = HERE/'pbe_energy_diagnostic'
out.mkdir(exist_ok=True)
rec009 = next(r for r in bundle['records'] if r['metadata']['structure_id']=='TiAl-test-009')
source = ROOT/'testset/TiAl-test50-v1'/rec009['metadata']['poscar']
destination = out/'TiAl-test-009'/'POSCAR'
destination.parent.mkdir(exist_ok=True)
shutil.copy2(source,destination)
assert hashlib.sha256(destination.read_bytes()).hexdigest()==rec009['metadata']['poscar_sha256']
summary['poscar'] = {'original': str(source), 'copy': str(destination), 'sha256': rec009['metadata']['poscar_sha256']}
(out/'sensitivity.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k not in ['leave_one_out','per_configuration_meV_atom']},indent=2))
print('Per-configuration: ID, xTi, V/N, fine-tuned error, pretrained error (meV/atom)')
for r in sorted(rows,key=lambda r:r['x_Ti']):
    print(f"{r['id']} {r['x_Ti']:.4f} {r['volume_A3_atom']:.4f} {r['PBE']:+.5f} {r['pretrained']:+.5f}")
print('Leave-one-out PBE RMSE range:',min(v['PBE']['rmse'] for v in summary['leave_one_out'].values()), max(v['PBE']['rmse'] for v in summary['leave_one_out'].values()))
