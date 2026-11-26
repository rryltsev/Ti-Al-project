"""Paired N694/N200 evaluation with independent metric and provenance checks."""
from pathlib import Path
import datetime,json
import numpy as np
from grace_mask_aggregate import case_result,load,sha,dump
ROOT=Path(__file__).resolve().parent

def main():
    out=ROOT/'comparison';out.mkdir(exist_ok=False);plan=load(ROOT/'plan.json');ref=np.load(ROOT/'reference_figure.npz')
    candidates=[(ROOT,c,ROOT/'ti3al',c['name']) for c in plan['cases']]
    candidates += [(Path(p['root']),p['case'],Path(p['ti3al_root']),p['ti3al_name']) for p in plan['parents']]
    expected={694:load(ROOT/'membership.json'),200:load(ROOT/'membership-N200.json')};cases={};failures={}
    assert expected[694]['validation']==expected[200]['validation'] and set(expected[200]['train'])<set(expected[694]['train'])
    geometry_sha=load(ROOT/'ti3al/prepared/summary.json')['geometry_sha256']
    for root,c,ti,name in candidates:
        try:
            assert c['XC']=='PBE' and c['method'] in ['full','restricted_radial']
            r=case_result(root,c,ref);assert r['ids']==expected[c['N']]
            ph=load(ti/'analysis'/name/'summary.json');a=load(ti/'analysis-asr'/name/'summary.json');assert ph['status']==a['status']=='PASS'
            assert ph['geometry_sha256']==geometry_sha and ph['source_summary_sha256']==sha(ti/'forces'/name/'summary.json')
            fs=load(ti/'forces'/name/'summary.json');assert fs.get('checkpoint_hashes',fs['case'].get('hashes'))==r['checkpoint_hashes']
            rows=np.load(ti/'analysis'/name/'dos_extended.npz');df=np.loadtxt(ti/'source/Ti3Al_D019/total_dos.dat')
            x=rows['frequency'];d=rows['dos'];dd=np.interp(x,df[:,0],df[:,1]/8,left=0,right=0)
            d=d/np.trapezoid(d,x);dd=dd/np.trapezoid(dd,x)
            cdf=np.r_[0,np.cumsum((d[1:]+d[:-1])/2*np.diff(x))];dcdf=np.r_[0,np.cumsum((dd[1:]+dd[:-1])/2*np.diff(x))]
            w1=float(np.trapezoid(abs(cdf-dcdf),x));l1=float(np.trapezoid(abs(d-dd),x))
            assert np.isclose(w1,ph['metrics']['DOS_Wasserstein_THz'],atol=1e-12)
            assert np.isclose(l1,ph['metrics']['normalized_DOS_L1'],atol=1e-12)
            r['Ti3Al']=dict(W1_THz=w1,L1=l1,ASR_W1_THz=a['ASR_DOS_metrics']['DOS_Wasserstein_THz'],mesh_imaginary=ph['mesh14']['imaginary_modes_weighted'],ASR_band_imaginary=a['ASR_band_imaginary_count'],residual_force_eV_A=ph['max_residual_force_eV_A'],displacement_convergence=ph['displacement_convergence'])
            r['values']={**r['metrics']['validation'],'TiAl_band_RMSE_THz':r['phonons']['band_RMSE_THz'],'TiAl_Gamma_singlet_error_THz':r['phonons']['gamma_singlet_abs_error_THz'],'Ti3Al_DOS_W1_THz':w1,'Ti3Al_DOS_L1':l1,'Ti3Al_DOS_ASR_W1_THz':r['Ti3Al']['ASR_W1_THz']}
            cases[c['name']]=r
        except Exception as exc:failures[c['name']]=repr(exc)
    groups={};paired={};mask_pairs={}
    if not failures:
        assert len(cases)==12
        for n in [200,694]:
            for seed in [10,20,30]:
                rr=[c for c in cases.values() if c['case']['seed']==seed and c['case']['N']==n]
                assert len(rr)==2 and len({tuple(c['first5_batch_hashes']) for c in rr})==1
            groups[str(n)]={}
            for method in ['full','restricted_radial']:
                cc=[cases[f'PBE-{method}-N{n}-seed{s}']['values'] for s in [10,20,30]]
                groups[str(n)][method]={k:dict(mean=float(np.mean([v[k] for v in cc])),sample_SD=float(np.std([v[k] for v in cc],ddof=1)),values=[v[k] for v in cc]) for k in cc[0]}
            mask_pairs[str(n)]={k:[cases[f'PBE-restricted_radial-N{n}-seed{s}']['values'][k]-cases[f'PBE-full-N{n}-seed{s}']['values'][k] for s in [10,20,30]] for k in cc[0]}
        for method in ['full','restricted_radial']:
            paired[method]={k:[cases[f'PBE-{method}-N694-seed{s}']['values'][k]-cases[f'PBE-{method}-N200-seed{s}']['values'][k] for s in [10,20,30]] for k in cases[f'PBE-{method}-N694-seed10']['values']}
    result=dict(status='PASS' if not failures else 'INCOMPLETE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=cases,failures=failures,groups=groups,paired_N694_minus_N200=paired,paired_radial_minus_full=mask_pairs,seeds=[10,20,30],test_access=False,r2SCAN_tested=False,design='Equal epochs50; N20010000 updates versus N69434700; each train-only offset fitted separately')
    dump(out/'summary.json',result)
    lines=['# GRACE PBE N694 versus N200','',result['status'],'','Mean +/- sample SD across seeds10/20/30. Fixed validation149. Same epoch budget, different update budget. No test access.','', '| N | Mask | E aligned meV/atom | F eV/A | Stress GPa | TiAl RMSE THz | Ti3Al W1 THz | Ti3Al L1 |','|---:|---|---:|---:|---:|---:|---:|---:|']
    for n,methods in groups.items():
        for method,g in methods.items():
            keys=['E_aligned_RMSE_meV_atom','F_RMSE_eV_A','stress_RMSE_GPa','TiAl_band_RMSE_THz','Ti3Al_DOS_W1_THz','Ti3Al_DOS_L1']
            lines.append('| '+n+' | '+method+' | '+' | '.join(f'{g[k]["mean"]:.6f} +/- {g[k]["sample_SD"]:.6f}' for k in keys)+' |')
    lines+=['','Ti3Al: archived fixed Cmcm geometry; DFT DOS only, force constants unavailable. Seed SD is not uncertainty across dataset splits.',json.dumps(failures,indent=2)]
    (out/'comparison.md').write_text('\n'.join(lines)+'\n');print(result['status'],len(cases),'models')
    if failures:raise SystemExit(1)
if __name__=='__main__':main()
