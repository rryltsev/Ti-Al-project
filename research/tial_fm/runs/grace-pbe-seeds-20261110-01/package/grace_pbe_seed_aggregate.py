"""Three-seed PBE-only validation, TiAl bands and Ti3Al DOS comparison."""
from pathlib import Path
import datetime,json
import numpy as np
from grace_mask_aggregate import case_result,load,sha,dump
ROOT=Path(__file__).resolve().parent
def main():
    out=ROOT/'comparison';out.mkdir(exist_ok=False);plan=load(ROOT/'plan.json');reference=np.load(ROOT/'reference_figure.npz')
    candidates=[(ROOT,c,ROOT/'ti3al',c['name']) for c in plan['cases']]
    candidates += [(Path(p['root']),p['case'],Path(plan['ti3al_parent']),p['ti3al_name']) for p in plan['parents']]
    expected=load(ROOT/'membership.json');cases={};failures={}
    geometry_sha=load(Path(plan['ti3al_parent'])/'prepared/summary.json')['geometry_sha256']
    for root,c,ti,name in candidates:
        try:
            r=case_result(root,c,reference);assert r['ids']==expected
            ph=load(ti/'analysis'/name/'summary.json');a=load(ti/'analysis-asr'/name/'summary.json');assert ph['status']==a['status']=='PASS'
            assert ph['geometry_sha256']==geometry_sha
            assert ph['source_summary_sha256']==sha(ti/'forces'/name/'summary.json')
            force_summary=load(ti/'forces'/name/'summary.json')
            if root==ROOT:assert force_summary['checkpoint_hashes']==r['checkpoint_hashes']
            else:assert force_summary['case']['hashes']==r['checkpoint_hashes']
            rows=np.load(ti/'analysis'/name/'dos_extended.npz');df=np.loadtxt(ti/'source/Ti3Al_D019/total_dos.dat')
            x=rows['frequency'];d=rows['dos'];dd=np.interp(x,df[:,0],df[:,1]/8,left=0,right=0)
            cdf=np.r_[0,np.cumsum((d[1:]+d[:-1])/2*np.diff(x))]/np.trapezoid(d,x)
            dcdf=np.r_[0,np.cumsum((dd[1:]+dd[:-1])/2*np.diff(x))]/np.trapezoid(dd,x)
            w1=float(np.trapezoid(abs(cdf-dcdf),x));assert np.isclose(w1,ph['metrics']['DOS_Wasserstein_THz'],atol=1e-12)
            r['Ti3Al']=dict(W1_THz=w1,L1=ph['metrics']['normalized_DOS_L1'],ASR_W1_THz=a['ASR_DOS_metrics']['DOS_Wasserstein_THz'],mesh_imaginary=ph['mesh14']['imaginary_modes_weighted'],ASR_band_imaginary=a['ASR_band_imaginary_count'],residual_force_eV_A=ph['max_residual_force_eV_A'],displacement_convergence=ph['displacement_convergence'])
            r['values']={**r['metrics']['validation'],'TiAl_band_RMSE_THz':r['phonons']['band_RMSE_THz'],'TiAl_Gamma_singlet_error_THz':r['phonons']['gamma_singlet_abs_error_THz'],'Ti3Al_DOS_W1_THz':w1,'Ti3Al_DOS_L1':r['Ti3Al']['L1'],'Ti3Al_DOS_ASR_W1_THz':r['Ti3Al']['ASR_W1_THz']}
            cases[c['name']]=r
        except Exception as exc:failures[c['name']]=repr(exc)
    groups={};paired={}
    if not failures:
        assert len(cases)==12
        for seed in [10,20,30]:
            rr=[c for c in cases.values() if c['case']['seed']==seed];assert len(rr)==4 and len({tuple(c['first5_batch_hashes']) for c in rr})==1
        for method in ['full','restricted','restricted_mlp','restricted_radial']:
            cc=[cases[f'PBE-{method}-N200-seed{s}']['values'] for s in [10,20,30]]
            groups[method]={k:dict(mean=float(np.mean([v[k] for v in cc])),sample_SD=float(np.std([v[k] for v in cc],ddof=1)),values=[v[k] for v in cc]) for k in cc[0]}
            paired[method]={k:[cases[f'PBE-{method}-N200-seed{s}']['values'][k]-cases[f'PBE-full-N200-seed{s}']['values'][k] for s in [10,20,30]] for k in cc[0]}
    result=dict(status='PASS' if not failures else 'INCOMPLETE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cases=cases,failures=failures,groups=groups,paired_minus_full=paired,seeds=[10,20,30],test_access=False,r2SCAN_tested=False)
    dump(out/'summary.json',result)
    lines=['# GRACE PBE N200: three seeds / four masks','',result['status'],'','Metrics are validation149 RMSE; energy aligned on train200 only; TiAl sorted band RMSE; Ti3Al archived Cmcm DOS W1. Fixed final epoch50, fp64.','', '| Mask | E aligned meV/atom | F eV/A | Stress GPa | TiAl bands THz | Ti3Al W1 THz |','|---|---:|---:|---:|---:|---:|']
    for method,g in groups.items():
        keys=['E_aligned_RMSE_meV_atom','F_RMSE_eV_A','stress_RMSE_GPa','TiAl_band_RMSE_THz','Ti3Al_DOS_W1_THz']
        lines.append('| '+method+' | '+' | '.join(f'{g[k]["mean"]:.6f} ± {g[k]["sample_SD"]:.6f}' for k in keys)+' |')
    lines+=['','Sample SD across training seeds only; no uncertainty across dataset splits. Ti3Al DFT force constants unavailable; no DFT band RMSE claimed.',json.dumps(failures,indent=2)]
    (out/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(result['status'],len(cases),'models')
    if failures:raise SystemExit(1)
if __name__=='__main__':main()
