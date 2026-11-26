"""Merge supplied STACIE replacements by state point, preserving old evidence."""
from pathlib import Path
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
ROOT = Path(r"C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project")
SOURCES = [ROOT / f"results-scripts-logs/{folder}/results/summary.json" for folder in
           ["stacie_pipeline_22.10.2026", "stacie_pipline_wsl2_22.10.2026"]]
old = json.loads((WORK / 'audit/oct20/viscosity_merged.json').read_text())
key = lambda r: (r['T'], r['dft'], round(r['c_Ti'], 6))
merged = {key(r): dict(r) for r in old}
changes, seen = [], set()
incoming_groups = [(source, t, models) for source in SOURCES
                   for t, models in json.loads(source.read_text())['temperatures'].items()]
for NEW, t, models in incoming_groups:
    for dft, group in models.items():
        for r in group['points']:
            k = key(r)
            assert k not in seen
            seen.add(k)
            assert int(t) == r['T'] and dft == r['dft']
            detail = NEW.parent / f"AlTi/{t}/{dft}/c_Ti_{r['c_Ti']:.2f}/result.json"
            assert json.loads(detail.read_text()) == r
            assert math.isclose(r['eta_pa_s']*1000, r['eta_mPa_s'])
            assert math.isclose(r['eta_std_pa_s']*1000, r['eta_std_mPa_s'])
            before = merged.get(k)
            if before and all(before.get(f) == r.get(f) for f in r):
                continue
            replacement = dict(r, source=str(NEW.relative_to(ROOT)),
                               result_file=str(detail), updated_on='2026-10-22')
            if before:
                replacement['previous_estimate'] = before
                # Earlier TDM used the earlier trajectory ensemble, not this update.
                replacement['tdm_comparison_status'] = 'not matched to updated trajectory set'
            merged[k] = replacement
            changes.append(dict(key=k, before=before, after=replacement,
                                result_sha256=hashlib.sha256(detail.read_bytes()).hexdigest()))
rows = sorted(merged.values(), key=key)
expected = [(1873,'pbe',.7), (1873,'scan',.4), (2000,'pbe',1.)]
pending = [k for k in expected if k not in [tuple(c['key']) for c in changes]]
refs = json.loads((WORK/'audit/oct20/reference_values.json').read_text())
experimental = []
for c, (a,b,lo,hi) in refs['takeda_table3'].items():
    c = float(c); exp = 10**(a+b/1873)
    pbe, scan = (merged[(1873,m,c)]['eta_mPa_s'] for m in ['pbe','scan'])
    experimental.append(dict(x_Ti=c,x_Al=1-c,experiment=exp,pbe=pbe,scan=scan,
                             closer='pbe' if abs(pbe-exp)<abs(scan-exp) else 'scan'))
ranking = []
for r in rows:
    ranking.append(dict(key=key(r), x_Al=round(1-r['c_Ti'],6),
        eta=r['eta_mPa_s'], std=r['eta_std_mPa_s'],
        relative_std_percent=100*r['eta_std_mPa_s']/r['eta_mPa_s'],
        n_trajectories=r['n_replicas'], duration_ps=(r['n_steps']-1)*r['timestep_ps'],
        neff=r['neff'], tdm_delta_percent_matched=r.get('tdm_delta_pct'),
        ideal_total_trajectories_for_3pct=math.ceil(r['n_replicas']*(r['eta_std_mPa_s']/r['eta_mPa_s']/.03)**2)))
pure_comparison=[]
for t in [1873,1941,2000,2100]:
    ref=.61*math.exp(27000/(8.31*t))
    item=dict(T=t,provisional_reference=ref)
    for model in ['pbe','scan']:
        eta=merged[(t,model,1.0)]['eta_mPa_s']
        item[model]=eta
        item[model+'_deviation_percent']=100*(eta/ref-1)
    item['closer']='pbe' if abs(item['pbe']-ref)<abs(item['scan']-ref) else 'scan'
    pure_comparison.append(item)
report = dict(sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in SOURCES],
              n_incoming=len(seen),n_merged=len(rows),changes=changes,pending_updates=pending,
              alloy_experiment_comparison=experimental,
              pure_ti_comparison=pure_comparison,
              precision_ranking=sorted(ranking,key=lambda r:r['relative_std_percent'],reverse=True),
              scaling_note='N estimates use ideal independent sampling and sigma proportional to N^-1/2; not a convergence guarantee. neff is a spectral diagnostic, not the number of independent MD replicas.',
              historical_tdm_note='Historical 25-case TDM/STACIE comparison remains on the old matched ensembles; do not combine new STACIE values with old TDM.')
(HERE/'viscosity_merged.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
(HERE/'viscosity_update_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:report[k] for k in ['n_incoming','n_merged','pending_updates','alloy_experiment_comparison']},indent=2))
for c in changes:
    print('UPDATED',c['key'],c['before']['eta_mPa_s'],'->',c['after']['eta_mPa_s'])
