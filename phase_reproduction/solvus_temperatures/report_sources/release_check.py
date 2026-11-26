from pathlib import Path
import hashlib,json,csv,re
from pypdf import PdfReader
s=Path('phase_reproduction/solvus_temperatures')
info=json.loads((s/'results.json').read_text(encoding='utf-8'))
src=s/'report/TiAl_600K_1273K.tex'
pdf=s/'report/TiAl_600K_1273K.pdf'
def digest(f): return hashlib.sha256(f.read_bytes()).hexdigest()
def check(x):
    if isinstance(x,list):
        for i in x: check(i)
    if isinstance(x,dict):
        if 'path' in x and 'sha256' in x: assert digest(s/x['path'])==x['sha256'],x['path']
        for v in x.values(): check(v)
check(info)
assert info['pressure_GPa']==0 and info['MTP_sha256']==digest(s.parent/'inputs/fitted.mtp')
assert [r['temperature_K'] for r in info['MTP_solvus']]==[600,1000,1273.15]
assert info['MTP_solvus'][2]['Al_at_pct'] is None
assert len(info['low_checks'])==9 and len(info['high_profile_files'])==4
assert all([x['T_K']==1273.15 for x in info['high_checks']])
assert all(r['band_at_pct']==[19.5,23.] for r in info['large_cell_high_response'])
assert all(r['scale_for_experimental_gap_equal_total_weights']>3*r['max_dx_dmu_eV_inv'] for r in info['large_cell_high_response'])
inv=list(csv.DictReader((s/'atomistic/series_summary.csv').open()))
assert len(inv)==34 and sum(int(r['windows']) for r in inv)==563 and sum(int(r['production_records']) for r in inv)==475710
assert all(r['status']=='all_windows_ok_and_archive_sha256_verified' for r in inv)
assert all(p.exists() for p in [s/'c46/T600/c5120/result.json', s/'c46/T1273.15/c5120/result.json',s/'analysis/600_independent_endpoints.json'])
reader=PdfReader(pdf)
assert len(reader.pages)==6 and all(len(p.extract_text() or '')>300 for p in reader.pages)
assert all(tag not in src.read_text(encoding='utf-8') for tag in ('% HIGH_SUMMARY','% HIGH_LARGE_CONTROL','% LOW_COMPARISON','% CONTROL_TABLE'))
print('Pages and extracted text lengths', [len(p.extract_text()) for p in reader.pages])
print('Verified inputs, results and PDF',digest(pdf),digest(src))
