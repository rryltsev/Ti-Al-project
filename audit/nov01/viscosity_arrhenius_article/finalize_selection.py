"""Refresh the selected preview and validate data preservation and article output."""
from pathlib import Path
import hashlib,json,time,math
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[2]
CANON=WORK/'audit/nov01/viscosity_article_combined'
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
read=lambda f:json.loads(f.read_text(encoding='utf-8'))
old=read(ROOT/'figure_manifest.json');current=read(CANON/'figure_manifest.json')
archive=ROOT/'before_selection/arrhenius_figure_manifest.json'
if not archive.exists():archive.write_text(json.dumps(old,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert old['composition']==current['composition'] and old['temperature']==current['temperature']
for key in ['pbe','scan']:
    for field in ['a','B_K']:
        assert math.isclose(old['fits'][key][field],current['arrhenius_fits'][key][field],rel_tol=1e-12,abs_tol=1e-10)
for ext in ['png','pdf','svg']:
    source=CANON/f'viscosity_combined.{ext}';dest=ROOT/f'viscosity_arrhenius_eta_T.{ext}'
    dest.write_bytes(source.read_bytes());old['output_sha256'][dest.name]=sha(dest)
    assert sha(source)==sha(dest)
assert sha(WORK/'manuscript/figures/viscosity.pdf')==sha(CANON/'viscosity_combined.pdf')
(ROOT/'caption_eta_T.tex').write_text((CANON/'caption.tex').read_text(encoding='utf-8'),encoding='utf-8')
old['updated_unix']=time.time();old['selected_variant']='eta_T'
old['selected_profile_version']=current['profile_version'];old['selected_rendering']=current['rendering']
old['selected_profile_sha256']=current['profile_sha256']
old['notes']=[n for n in old['notes'] if 'Primary article files remain unchanged' not in n]
old['notes'].append('The eta_T variant is now selected and published to the article, with uniform black Takeda squares. The lneta_inverse_T files remain the archived alternative from the previous turn.')
(ROOT/'figure_manifest.json').write_text(json.dumps(old,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
article=WORK/'manuscript/build/ti-temperature/main.pdf'
reader=PdfReader(article);pages=[i+1 for i,page in enumerate(reader.pages) if 'STACIE viscosity as a function' in (page.extract_text() or '')]
assert len(pages)==1,pages
page_text=reader.pages[pages[0]-1].extract_text()
assert 'Arrhenius' in page_text and 'Extrapolated by 3 K' not in page_text
assert '??' not in page_text
log=(article.parent/'staging/main.log').read_text(encoding='utf-8')
assert 'Missing character' not in log and 'Overfull' not in log
record={'selected_variant':'eta_T','profile_version':current['profile_version'],'figure_page':pages[0],
        'all_points_and_errors_unchanged':True,'arrhenius_coefficients_unchanged':True,
        'takeda_symbols':'all five filled black squares','extrapolation_legend_removed':True,
        'article':str(article),'figure_sha256':sha(WORK/'manuscript/figures/viscosity.pdf'),
        'article_sha256':sha(article),'updated_unix':time.time()}
(ROOT/'selection_check.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (ROOT/'README_RU.txt').open('a',encoding='utf-8') as f:
    f.write('\nВыбран пользователем и установлен в статью вариант eta(T). Все точки Takeda показаны одинаковыми чёрными квадратами, дополнительная легенда об экстраполяции убрана. Основной воспроизводящий скрипт: ../viscosity_article_combined/build_figure.py; профиль: ../viscosity_article_combined/figure_profile.json. Предыдущие варианты и данные сохранены.\n')
print(json.dumps(record,ensure_ascii=False))
