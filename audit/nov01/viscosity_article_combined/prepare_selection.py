from pathlib import Path
import json,hashlib,shutil,time
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[2]
BASE=WORK/'audit/oct29/pbe15k_ti_temperature'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
backup=ROOT/'before';backup.mkdir(exist_ok=True)
for rel,name in [('manuscript/figures/viscosity.pdf','viscosity.pdf'),('manuscript/sections/results.tex','results.tex'),('audit/oct29/pbe15k_ti_temperature/update_figure.py','update_figure.py'),('audit/oct29/pbe15k_ti_temperature/figure7_data.json','figure7_data.json')]:
 p=WORK/rel
 if p.exists() and not (backup/name).exists():shutil.copy2(p,backup/name)
merged_path=WORK/'audit/oct25/viscosity/viscosity_merged.json'
archived_path=BASE/'before/viscosity_current_oct29.json'
temp_path=WORK/'audit/nov01/pbe15k_legacy_pooling_v2/temperature_plot/plot_data.json'
refs_path=WORK/'audit/oct20/reference_values.json'
merged=read(merged_path);archived=read(archived_path);temp=read(temp_path)
comp={m:sorted([r for r in merged if r['T']==1873 and r['dft']==m],key=lambda r:1-r['c_Ti']) for m in ['pbe','scan']}
scan_temp=sorted([r for r in archived if r['dft']=='scan' and r['c_Ti']==1],key=lambda r:r['T'])
legacy_root=Path(r'C:/Users/Roman Ryltsev/Desktop/STARTUPS/Ti-Al-X/TiAl-Project/results-scripts-logs')
candidate_path=legacy_root/'stacie_pipeline_cluster/stacie_pipline/results/AlTi/1941/pbe/c_Ti_1.00/result.json'
candidate=read(candidate_path)
caption=r'''\caption{STACIE viscosity as a function of composition at 1873~K (a) and temperature for pure Ti (b). Colored symbols show PBE and \rscan{} estimates; lines connect sampled states. The composition data use 20~ps production trajectories. In (b), the PBE values at 1873 and 2000~K combine archived 20~ps trajectories with two 20~ps segments from each new 40~ps trajectory, using a common spectral cutoff of 1.25~THz; the 2100~K value uses full 40~ps trajectories. These pooled estimates remain exploratory because preparation differs and spectral diagnostics are not uniformly satisfied. Error bars are formal estimator standard errors and exclude systematic uncertainties; data provenance is retained in the supporting records. Black squares in (a) are the experimental correlations of Takeda \textit{et al.}\cite{Takeda2025}; open squares require a 3~K extrapolation. The dashed curve with an asterisk in (b) is a provisional transcription of the Ishikawa \textit{et al.}\cite{Ishikawa2012} correlation; shading denotes its reported 10.3\% uncertainty, not a specified confidence level.}'''
profile=dict(version='2026-11-01-r1',created_unix=time.time(),authoritative_article_selection=True,
 composition=comp,temperature=dict(pbe=temp['pbe15k'],scan=scan_temp),references=read(refs_path),caption_tex=caption,
 selection=dict(composition_PBE_checkpoint=5000,temperature_PBE_checkpoint=15000,temperature_frozen_snapshot=True,
 display_labels=['PBE','r2SCAN'],checkpoint_identity_in_provenance_only=True,production_protocol_changed=False),
 intermediate_temperature_search=dict(status='no_verified_15k_result_found',candidate=candidate,candidate_checkpoint=5000,candidate_source=str(candidate_path),included=False,
 local_export_15k_temperatures=[1873,2000],remote_campaign_temperatures=[1873,2000,2100],
 note='1941 K legacy result is the previously archived 5k baseline. Do not silently relabel as 15k.'),
 sources_sha256={str(p):sha(p) for p in [merged_path,archived_path,temp_path,refs_path,candidate_path]},
 notes=['User requested original composition PBE 5k points, updated pure-Ti temperature values, and no checkpoint labels in the figure or caption.',
 'The two panels use distinct PBE parameter snapshots and trajectory ensembles; do not interpret their pure-Ti endpoints as identical measurements.',
 'This fixed article selection does not modify official 40 ps monitoring estimates or stopping criteria.'])
assert len(comp['pbe'])==11 and len(comp['scan'])==11
assert [r['T'] for r in profile['temperature']['pbe']]==[1873,2000,2100]
(ROOT/'figure_profile.json').write_text(json.dumps(profile,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(ROOT/'caption.tex').write_text(caption+'\n',encoding='utf-8')
text=(WORK/'manuscript/sections/results.tex').read_text(encoding='utf-8')
start=text.index('The STACIE results compare PBE and')
end=text.index('\\begin{figure*}[t]',start)
intro=r'''The STACIE results compare PBE and \rscan{} transport properties (Fig.~\ref{fig:viscosity}).
At 1873~K, the PBE composition series gives $3.259\pm0.153$~mPa\,s for pure Ti and $4.059\pm0.113$, $4.883\pm0.155$, and $1.761\pm0.036$~mPa\,s at $x_{\mathrm{Al}}=0.20$, 0.45, and 0.80, respectively.
These estimates use 20~ps trajectories, with 15 or 50 replicas depending on composition.
The largest sampled PBE value is $5.021\pm0.193$~mPa\,s at $x_{\mathrm{Al}}=0.40$; the sampled grid does not locate a continuous maximum precisely.
The \rscan{} viscosities exceed the PBE values at all eleven matched compositions and form a broad high-viscosity region over approximately $x_{\mathrm{Al}}=0.40$--0.60.
The largest sampled \rscan{} value is $7.19\pm0.28$~mPa\,s at $x_{\mathrm{Al}}=0.45$, close to $7.05\pm0.26$~mPa\,s at 0.40.
The composition and temperature panels use separate PBE parameter snapshots and trajectory ensembles, whose identities and source files are retained in the supporting data.

'''
text=text[:start]+intro+text[end:]
start=text.index('The higher \\rscan{} alloy viscosity')
end=text.index('The experimental alloy series does not extend',start)
comparison=r'''The higher \rscan{} alloy viscosity does not improve agreement at four of the five compositions for which PBE and experimental correlations are available.
At $x_{\mathrm{Al}}=0.45$, the Takeda correlation gives 5.30~mPa\,s, compared with 4.88 for PBE and 7.19 for \rscan{}.
At $x_{\mathrm{Al}}=0.80$, the corresponding values are 1.764, 1.761, and 2.528~mPa\,s.
PBE is closer at $x_{\mathrm{Al}}=0.45$, 0.60, 0.70, and 0.80; at $x_{\mathrm{Al}}=0.90$, the \rscan{} value of 1.218~mPa\,s is closer to the experimental correlation of 1.437~mPa\,s than the PBE value of 0.951~mPa\,s.
The values at 0.45 and 0.60 involve a 3~K extrapolation beyond the stated measurement range; the values at 0.70, 0.80, and 0.90 are interpolated.
'''
text=text[:start]+comparison+text[end:]
start=text.index('For pure Ti, PBE decreases')
end=text.index('An archived comparison of 25 matched trajectory ensembles',start)
pure=r'''For pure Ti, the selected PBE temperature series decreases from $3.200\pm0.084$~mPa\,s at 1873~K through $2.830\pm0.077$~mPa\,s at 2000~K to $2.528\pm0.068$~mPa\,s at 2100~K.
The first two values combine archived and new trajectories at a common cutoff of 1.25~THz; the last retains the separate 40~ps estimate from 15 replicas.
The \rscan{} series decreases from 3.70~mPa\,s at 1873~K to 2.99~mPa\,s at 2100~K, with 3.30~mPa\,s at 2000~K.
Against the provisional Ishikawa correlation, PBE and \rscan{} lie on opposite sides at 1873~K (approximately $-7.4\%$ and $+7.0\%$).
At 2000 and 2100~K, \rscan{} is closer to the central correlation.
These comparisons do not establish a uniform functional ranking across the liquid states studied.

The statistical interpretation remains sensitive to stress-correlation sampling.
The pooled PBE estimates are an exploratory comparison: the archived and new trajectories have different preparation histories, and not every temporal window passes spectral diagnostics at the common cutoff.
The narrow acceptable cutoff interval for the aggregate fits was identified using the same data and is not an independent convergence test.
Formal error bars exclude preparation, finite-size, and model uncertainties; the official new-trajectory convergence assessment remains separate from the selected article points.
'''
text=text[:start]+pure+text[end:]
(WORK/'manuscript/sections/results.tex').write_text(text,encoding='utf-8')
# Keep the hourly updater on the user's explicit article selection.
u=BASE/'update_figure.py';s=u.read_text(encoding='utf-8')
marker='# User-selected article profile, 2026-11-01.'
if marker not in s:
 prefix='''# User-selected article profile, 2026-11-01.
from pathlib import Path as _ProfilePath
import sys as _profile_sys, runpy as _profile_runpy
_profile_builder = _ProfilePath(__file__).resolve().parents[3] / 'audit/nov01/viscosity_article_combined/build_figure.py'
if _profile_builder.with_name('figure_profile.json').exists():
    _profile_sys.argv = [str(_profile_builder), '--publish'] + _profile_sys.argv[1:]
    _profile_runpy.run_path(str(_profile_builder), run_name='__main__')
    raise SystemExit(0)

'''
 u.write_text(prefix+s,encoding='utf-8')
print(json.dumps({'profile':str(ROOT/'figure_profile.json'),'PBE_temperatures':[r['T'] for r in profile['temperature']['pbe']],'intermediate_found':False}))
