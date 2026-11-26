from pathlib import Path
import json, numpy as np
OUT=Path(__file__).parent
rows=json.loads((OUT/'thermo_reprocessed.json').read_text())
st=json.loads((OUT/'viscosity_merged.json').read_text());tdm=json.loads((OUT/'tdm_merged.json').read_text())
# Source: Brillo et al. 2025, Table 7, printed rho(1873) column.
# Multiple rows are separate published series, not extra independent compositions.
rho={0:[4.184],.1:[3.96,3.84],.2:[3.66,3.69],.25:[3.73,3.67,3.65],.3:[3.55,3.54],.4:[3.36],.5:[3.36,3.32],.6:[3.04],.7:[2.85,2.82],.8:[2.57,2.57],.9:[2.36],1:[2.08]}
# Read from raster of Witusiewicz 2008 Fig.7 at scale 1.6, dark circle centers.
# x=0 axis pixel559, x=1 pixel880; y=0 pixel775, y=-30 pixel1097.
# Visual digitization only; conservative reading tolerance ~0.2 kJ/mol.
mix_y=[870,942,990,1051,1092,1094,1064,997,897]
mix_ref={round((i+1)/10,1):-30*(y-775)/322 for i,y in enumerate(mix_y)}
# Source: Takeda et al. 2025, Table 3. log10(eta/mPa.s)=a+b/T.
takeda={.1:(-1.251,2638,1665,1878),.2:(-1.084,2492,1674,1884),.3:(-1.185,3021,1705,1888),.4:(-1.311,3487,1714,1870),.55:(-.884,3013,1822,1870)}
refs=dict(density_table7={str(k):v for k,v in rho.items()},mixing_digitized_kJmol={str(k):v for k,v in mix_ref.items()},mixing_digitization_note='Provisional manual raster digitization; reading tolerance about 0.2 kJ/mol, not experimental uncertainty; original experiments unavailable.',takeda_table3={str(k):v for k,v in takeda.items()},pureTi_ishikawa_note='Formula transcribed from supplied GPT Work review, original full paper not yet independently checked: eta=0.61 exp(27000/(8.31 T)); uncertainty 10.3% is not specified as 1sigma.')
(OUT/'reference_values.json').write_text(json.dumps(refs,indent=2),encoding='utf8')
metrics={}
for m in ['pbe','scan','sol']:
    group=[r for r in rows if r['T']==1873 and r['dft']==m]
    diff=np.array([r['rho']-np.mean(rho[r['x_Al']]) for r in group]);ref=np.array([np.mean(rho[r['x_Al']]) for r in group])
    d=dict(density_MAE_g_cm3=float(np.mean(abs(diff))),density_MAPE_pct=float(np.mean(abs(diff)/ref)*100),density_max_pct=float(np.max(abs(diff)/ref)*100))
    group=[r for r in rows if r['T']==2000 and r['dft']==m and r['x_Al'] in mix_ref]
    for field in ['dH_reported','dH_external_pressure']:
        dif=np.array([r[field]-mix_ref[r['x_Al']] for r in group]);d[field+'_MAE']=float(np.mean(abs(dif)));d[field+'_RMSE']=float(np.sqrt(np.mean(dif**2)))
    metrics[m]=d;print(m,d)
print('VISCOSITY VS TAKEDA1873: cTi experiment PBE scan (mPa.s)')
for c,(a,b,tmin,tmax) in takeda.items():
    e=10**(a+b/1873);vals={r['dft']:(r['eta_mPa_s'],100*(r['eta_mPa_s']/e-1)) for r in st if r['T']==1873 and r['c_Ti']==c}
    print(c,round(e,4),vals,'extrapolated',1873>tmax)
print('PURE TI: T reference PBE scan; reference is secondary transcription')
for T in [1873,1941,2000,2100]:
    e=.61*np.exp(27000/(8.31*T));print(T,round(e,5),[(r['dft'],round(r['eta_mPa_s'],5),round(100*(r['eta_mPa_s']/e-1),2)) for r in st if r['T']==T and r['c_Ti']==1])
ratios=[abs(r['tdm_delta_pct']) for r in st if 'tdm_delta_pct' in r]
print('TDM COMPARE',len(ratios),'median abs %',np.median(ratios),'max',max(ratios))
print('NEFF<60',sum(r['neff']<60 for r in st),'/',len(st),'minimum',min(r['neff'] for r in st))
print('TDM alpha at bound',[(r['T'],r['dft'],r['c_Ti'],r.get('alpha')) for r in tdm if r.get('alpha',0)>9.99])
print('DRIFT/SE SAMPLE')
for T in [1873,2000]:
 for m in ['pbe','scan','sol']:
    a=[r for r in rows if r['T']==T and r['dft']==m]
    print(T,m,'rho max blocksem',max(r['rho_sem_5ps'] for r in a),'Hmix max blocksem',max(r['dH_external_sem_5ps'] for r in a),'Epot max halfdrift',max(abs(r['Epot_half_drift_kJmol']) for r in a),'rho maxhalfdrift',max(abs(r['rho_half_drift']) for r in a))
(OUT/'comparison_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf8')
