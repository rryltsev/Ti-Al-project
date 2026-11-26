from pathlib import Path
import json,sys,os,time,hashlib
base=Path(__file__).resolve().parent;path=base/'scan_common_cutoff.py'
ns={'__file__':str(path),'__name__':'cutoff_helpers'}
exec(compile(path.read_text(encoding='utf-8-sig').split("previous=read(BASE/'combined_results.json')")[0],str(path),'exec'),ns)
np=ns['np'];fm=ns['fit_model_spectrum'];L=ns['LorentzModel'];C=ns['CV2LCriterion']
rows=[]
for T in [1873,2000]:
 for k in ['old20','new_split20','new40','pooled20','new_first20','new_second20']:
  for fc in [1.23,1.25,1.26]:
   rr=[]
   for seed in [1,42,123]:
    r=fm(ns['spectra'][T][k],L(),fc*1e12,8.,C(),np.random.default_rng(seed),100)
    assert np.isfinite(r['criterion']) and 'msg' not in r, (T,k,fc,r.get('msg'))
    rr.append({a:float(r[a]) for a in ['acint','acint_std','criterion_zscore','cost_zscore']})
   rows.append({'T':T,'group':k,'cutoff_THz':fc,'eta_range_over_seeds_mPa_s':1000*(max(r['acint'] for r in rr)-min(r['acint'] for r in rr)),'cutoff_z_range_over_seeds':max(r['criterion_zscore'] for r in rr)-min(r['criterion_zscore'] for r in rr)})
out={'finite_criterion_and_no_fit_errors':True,'seeds':[1,42,123],'checks':rows,'max_eta_spread_mPa_s':max(r['eta_range_over_seeds_mPa_s'] for r in rows),'max_cutoff_z_spread':max(r['cutoff_z_range_over_seeds'] for r in rows)}
(base/'common_cutoff/numerical_validation.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='checks'}))

