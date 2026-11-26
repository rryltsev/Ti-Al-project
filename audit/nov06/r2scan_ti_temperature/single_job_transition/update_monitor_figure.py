"""Scientific monitoring plot; never writes article figures or profiles."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[3]
sys.path.insert(0,str(WORK/'audit/nov01/r2scan_viscosity_diagnostics/deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
reg=json.loads((ROOT/'registry.json').read_text());rows=[];old=[]
for sid,entry in reg['runs'].items():
    d=ROOT/sid;legacy=json.loads((d/'legacy/manifest.json').read_text())
    old.append({**legacy['published'],'T':entry['temperature_K']})
    latest=d/f'collected/analysis/latest_{sid}.json'
    if not latest.exists():continue
    batch=json.loads(latest.read_text());n=batch['n_replicas']
    assert n>=5 and n%5==0
    assert batch['protocol_sha256']==entry['protocol_sha256']
    point=batch['points'][0]
    rows.append(dict(T=entry['temperature_K'],N=n,new40=point['windows']['40ps'],first20=point['windows']['first20ps'],second20=point['windows']['second20ps'],combined20=point['combined_20ps'],checks=batch['checks'],stop=batch['stop_recommended'],review=batch['review_required']))
if not rows:
    print('No fully analyzed new r2SCAN batch yet');raise SystemExit(0)
fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for ax,key,title in zip(axes,['combined20','new40'],['Accumulated: legacy + new (20 ps spectra)','New 40 ps only: duration diagnostic']):
    ax.errorbar([x['T'] for x in old],[x['eta_mPa_s'] for x in old],yerr=[x['eta_std_mPa_s'] for x in old],fmt='o--',color='.6',capsize=3,label='Verified legacy 20 ps')
    ax.errorbar([x['T'] for x in rows],[x[key]['eta_mPa_s'] for x in rows],yerr=[x[key]['eta_std_mPa_s'] for x in rows],fmt='s-',color='#C04C35',capsize=3,label='Accumulated, adopted SE' if key=='combined20' else 'New 40 ps, formal SE')
    for x in rows:ax.annotate((f"{x['combined20']['legacy_replicas']} old + {x['N']} new" if key=='combined20' else f"{x['N']} new only"),(x['T'],x[key]['eta_mPa_s']),xytext=(0,8),textcoords='offset points',fontsize=8,ha='center')
    ax.set(xlabel='Temperature (K)',ylabel='Viscosity (mPa s)',title=title,xlim=(1840,2140));ax.set_xticks([1873,1941,2000,2100]);ax.legend(fontsize=8)
fig.savefig(ROOT/'monitor_current.png',dpi=180);plt.close(fig)
(ROOT/'monitor_current.json').write_text(json.dumps(dict(points=rows,legacy=old,article_profile_changed=False),indent=2)+'\n')
print(str((ROOT/'monitor_current.png').resolve()))
