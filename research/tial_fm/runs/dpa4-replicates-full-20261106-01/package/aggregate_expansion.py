"""Automatic, failure-visible seed/size comparison after the Slurm QC arrays."""
from pathlib import Path
import argparse,datetime,hashlib,json,os
import numpy as np

ROOT=Path(__file__).resolve().parent
METRICS=['E_raw_meV_atom','E_aligned_meV_atom','F_component_eV_A','stress_component_GPa']

def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(x):
    x=np.asarray(x,dtype=float)
    return dict(n=len(x),mean=float(x.mean()),sample_SD=float(x.std(ddof=1)) if len(x)>1 else None,
                min=float(x.min()),max=float(x.max()),values=x.tolist())

def collect(path,expected):
    t=load(path/'training-summary.json');e=load(path/'evaluation/summary.json');q=load(path/'evaluation/numerical-QC/summary.json')
    for v in [t,e,q,load(path/'zero-step.json'),load(path/'parameter-update-check.json')]:assert v['status']=='PASS'
    assert not t['test_access'] and not e['test_access']
    c=t['case'];assert c['N']==expected['N'] and c['seed']==expected['seed']
    assert t['steps']==50*c['N'] and (e['train_evaluated'],e['validation_evaluated'])==(c['N'],149)
    model=Path(t['final_checkpoint']);mh=sha(model);assert mh==e['model_sha256']==q['sha256']
    if 'model_sha256' in expected:assert mh==expected['model_sha256']
    m=load(path/'evaluation/metrics/metrics.json')[c['XC']];h=load(path/'evaluation/harmonics/summary.json');assert h['status']=='PASS'
    rr=list(map(json.loads,(path/'evaluation/metrics/per_structure.jsonl').read_text().splitlines()))
    assert len(rr)==c['N']+149 and {r['split'] for r in rr}=={'train','validation'}
    gf=np.array(h['model']['gamma_THz']);pol=np.array(h['model']['gamma_c_axis_polarization']);singlet=3+int(np.argmax(pol[3:]));doublet=[i for i in range(3,6) if i!=singlet]
    values={k:m['metrics']['validation'][k]['RMSE'] for k in METRICS}
    values.update(phonon_sorted_RMSE_THz=h['sorted_band_RMSE_THz'],Gamma_singlet_THz=float(gf[singlet]),Gamma_doublet_THz=float(gf[doublet].mean()),imaginary_modes=h['model']['imaginary_modes_weighted'])
    initial=load(path/'parameter-inventory.json')['parameters']
    adapter_sha=hashlib.sha256(json.dumps({k:v['initial_sha256'] for k,v in initial.items() if v['adapter']},sort_keys=True).encode()).hexdigest()
    additional=load(path/'evaluation/physical-diagnostics/summary.json') if (path/'evaluation/physical-diagnostics/summary.json').exists() else None
    if 'name' in expected:assert additional and additional['status']=='PASS'
    return dict(path=str(path),case=c,model_sha256=mh,values=values,metrics=m,harmonics=h,physical_diagnostics=additional,
                exposures_sha256=sha(path/'training-exposures.jsonl'),initial_adapters_sha256=adapter_sha,training=t),rr

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--group',choices=['N200','N694'],required=True);a=ap.parse_args()
    plan=load(ROOT/'plan.json');wanted=[c for c in plan['cases'] if c['name'] in plan['groups'][a.group]]
    items=[(ROOT/'training-plan01/cases'/c['name'],c) for c in wanted]
    # Completed seed10 N200 is the common reference for both seed and size comparisons.
    items += [(Path(c['remote_path']),c) for c in plan['existing_seed10'].values()]
    out=ROOT/'aggregates'/a.group;out.mkdir(parents=True,exist_ok=False)
    cases={};rows={};failures=[]
    for path,expected in items:
        try:
            c,rr=collect(path,expected);key=f"{c['case']['XC']}-{c['case']['method']}-N{c['case']['N']}-seed{c['case']['seed']}"
            cases[key]=c;rows[key]=rr
        except Exception as e:failures.append(dict(path=str(path),error=repr(e)))
    checks=[];grouped={};paired={};size_control={}
    if not failures:
        for n in sorted({c['case']['N'] for c in cases.values()}):
            seeds=sorted({c['case']['seed'] for c in cases.values() if c['case']['N']==n})
            for seed in seeds:
                cc=[c for c in cases.values() if c['case']['N']==n and c['case']['seed']==seed]
                assert len(cc)==4
                if len({c['exposures_sha256'] for c in cc})!=1:
                    failures.append(dict(error=f'Optimizer exposure order differs within N{n}/seed{seed}.'))
            for xc in ['PBE','r2SCAN']:
                for method in ['lora16','full']:
                    keys=[f'{xc}-{method}-N{n}-seed{s}' for s in seeds];cc=[cases[k] for k in keys]
                    grouped[f'{xc}-{method}-N{n}']={metric:stats([c['values'][metric] for c in cc]) for metric in cc[0]['values']}
                    train_ids=[sorted(r['id'] for r in rows[k] if r['split']=='train') for k in keys]
                    assert all(ids==train_ids[0] for ids in train_ids)
                    if len(seeds)>1:
                        if len({c['exposures_sha256'] for c in cc})!=len(seeds):failures.append(dict(error=f'Non-independent training sampling seeds: {keys}'))
                        if method=='lora16' and len({c['initial_adapters_sha256'] for c in cc})!=len(seeds):failures.append(dict(error=f'LoRA initialization hashes identical across seeds: {keys}'))
                paired[f'{xc}-N{n}']={metric:stats([cases[f'{xc}-lora16-N{n}-seed{s}']['values'][metric]-cases[f'{xc}-full-N{n}-seed{s}']['values'][metric] for s in seeds]) for metric in cases[f'{xc}-full-N{n}-seed{seeds[0]}']['values']}
        valsets=[sorted(r['id'] for r in rr if r['split']=='validation') for rr in rows.values()]
        assert all(ids==valsets[0] and len(ids)==149 for ids in valsets)
        checks=['Same validation149 IDs','Same N200 membership across seeds','Matched exposure order within size/seed','Distinct exposure streams and LoRA initialization for seeds10/20/30']
        if a.group=='N694':
            for xc in ['PBE','r2SCAN']:
                for method in ['lora16','full']:
                    small=cases[f'{xc}-{method}-N200-seed10']['values'];large=cases[f'{xc}-{method}-N694-seed10']['values']
                    size_control[f'{xc}-{method}']={k:dict(N200=small[k],N694=large[k],delta_N694_minus_N200=large[k]-small[k]) for k in small}
    result=dict(status='PASS' if not failures else 'INCOMPLETE_OR_FAILED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        job_id=os.environ.get('SLURM_JOB_ID'),group=a.group,cases=cases,failures=failures,checks=checks,
        seed_summary=grouped,paired_LoRA_minus_full=paired,size_control_seed10=size_control,
        interpretation='Mean and sample SD over training seeds; not uncertainty of individual structures. Three seeds are a technical replication, not proof of universal method superiority. N694 has seed10 only, paired to N200 seed10; same50 epochs, different step counts. Test remains unused.')
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    text=f"# {a.group}: автоматическое сравнение\n\nСтатус: **{result['status']}**.\n\n"
    if failures:text+='Сбои сохранены, неполные кандидаты не исключены молча:\n\n'+''.join(f"- {f}\n" for f in failures)
    else:
        text+='Среднее ± sample SD по training seeds; при одном seed показано одно значение. Энергия выровнена только на соответствующем train.\n\n| Вариант | E aligned, meV/atom | F, eV/Å | Stress, GPa | Phonon sorted RMSE, THz | Γ singlet, THz |\n|---|---:|---:|---:|---:|---:|\n'
        for name,metrics in grouped.items():
            values=[]
            for metric in ['E_aligned_meV_atom','F_component_eV_A','stress_component_GPa','phonon_sorted_RMSE_THz','Gamma_singlet_THz']:
                s=metrics[metric];values.append(f"{s['mean']:.5f}"+(f" ± {s['sample_SD']:.5f}" if s['sample_SD'] is not None else ''))
            text+='| '+name+' | '+' | '.join(values)+' |\n'
        text+='\nПолные метрики каждого seed, проверки и парные разности сохранены в summary.json. Ни один checkpoint не выбран по test. N694 seed10 — контроль размера данных; вывод о его разбросе между seeds пока невозможен.\n'
    (out/'report.md').write_text(text,encoding='utf8')
    print(json.dumps(dict(status=result['status'],group=a.group,complete_cases=len(cases),failures=failures)))
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
