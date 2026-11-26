from pathlib import Path
import json,hashlib,time
O=Path(__file__).resolve().parent;R=O.parent;W=R.parents[2]
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protected=[R/'figure_profile.json',W/'manuscript/figures/viscosity.pdf',W/'manuscript/build/ti-temperature/main.pdf',W/'manuscript/sections/results.tex',W/'manuscript/supporting_data/viscosity_article_profile.json',W/'manuscript/supporting_data/viscosity_temperature_current.json',R/'current_campaign_20261105/figure_profile.json',R/'current_campaign_20261105/figure7b_current_campaign.pdf']
(O/'protected_files_before.json').write_text(json.dumps({str(p):sha(p) for p in protected},indent=2)+'\n')
p=read(R/'figure_profile.json');p.update(version='2026-11-05-current-campaign-with-1941-preview',created_unix=time.time(),authoritative_article_selection=False);p['temperature']['pbe']=[]
for root,T,N in [('audit/oct29/pbe15k_ti_temperature',1873,30),('audit/nov05/pbe15k_ti_1941',1941,10),('audit/oct29/pbe15k_ti_temperature',2000,30),('audit/oct29/pbe15k_ti_temperature',2100,35)]:
 r=W/root;f=r/f'collected/analysis/Ti_{T}K/batch_{N:04d}/summary.json';d=read(f)
 assert d==read(r/f'collected/analysis/latest_Ti_{T}K.json') and read(r/'collected/analysis/latest.json')['points'][f'Ti_{T}K']['n_replicas']==N
 assert d['protocol_sha256']==sha(r/'protocol.json') and d['checkpoint_sha256']==read(r/'protocol.json')['checkpoint']['sha256'] and len(d['points'][0]['sources'])==N
 review=read(r/f'reviews/Ti_{T}K_N{N}/assessment.json');assert review['visual_review']['notes'] and review['n_replicas']==N
 a=d['points'][0]['windows']['40ps'];p['temperature']['pbe'].append(dict(T=T,eta_mPa_s=a['eta_mPa_s'],eta_std_mPa_s=a['eta_std_mPa_s'],n_replicas=N,production_ps=40,checkpoint=15000,source=str(f),source_sha256=sha(f),cost_z=a['cost_z'],cutoff_z=a['criterion_z'],stop_recommended=d['stop_recommended'],pass_streak=d['pass_streak'],preliminary=True));p['sources_sha256'][str(f)]=sha(f)
p['selection'].update(temperature_frozen_snapshot=False,temperature_current_campaign_preview=True,temperature_includes_new_1941_series=True,article_profile_modified=False)
p['intermediate_temperature_search']=dict(status='new1941K15k_campaign_started_20261105',historical_15k_exists=False,historical_5k_relabelled=False,new_campaign_N=10,new_campaign_source=p['temperature']['pbe'][1]['source'],included=True)
p['notes']=['User requested standalone Figure7b including the new1941K series. Latest complete new40ps batches:1873N30,1941N10,2000N30,2100N35.','PBE temperature points are preliminary new40ps results without historical pooling;1941K is new15k evidence, not relabelled5k.','r2SCAN and experimental references retain selected article data/style. Weighted Arrhenius curves are recomputed for displayed values.','Formal SE excludes systematics;1873/2100 fail cutoff,1941 fails N/precision/duration,2000 has one complete pass of two.','Preview does not publish or compile the article or replace its fixed profile.']
p['caption_tex']=r'\caption{Current standalone preview of Figure7b for pure Ti. Symbols show the latest fully analyzed PBE 40~ps campaign estimates at1873,1941,2000 and2100~K and the selected \rscan{} estimates. Solid lines are weighted Arrhenius fits in ln(viscosity), with weights $(\eta/\sigma_\eta)^2$. PBE points remain preliminary; error bars are formal estimator standard errors and exclude systematic uncertainties. The dashed Ishikawa correlation and its10.3\% band retain the selected article styling. This preview does not change the manuscript data selection.}'
(O/'figure_profile.json').write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(O/'render_preview.py').write_bytes((R/'current_campaign_20261105/render_preview.py').read_bytes())
print('Inputs:',[(x['T'],x['n_replicas'],round(x['eta_mPa_s'],6),round(x['eta_std_mPa_s'],6)) for x in p['temperature']['pbe']])
