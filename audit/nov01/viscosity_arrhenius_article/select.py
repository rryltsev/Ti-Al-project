"""Record the user's selected eta(T) figure and uniform Takeda symbols."""
from pathlib import Path
import json,time
ROOT=Path(__file__).resolve().parent;WORK=ROOT.parents[2]
path=WORK/'audit/nov01/viscosity_article_combined/figure_profile.json'
p=json.loads(path.read_text(encoding='utf-8'))
old_points=json.dumps([p['composition'],p['temperature']],sort_keys=True)
p['version']='2026-11-01-r2-arrhenius'
p['updated_unix']=time.time()
p['rendering']={'temperature_curve':'weighted_arrhenius','temperature_coordinates':'eta_T','takeda_symbols':'uniform_black_squares'}
p['selection']['article_figure_variant']='eta_T_arrhenius_uniform_takeda'
p['caption_tex']=(ROOT/'caption_eta_T.tex').read_text(encoding='utf-8').strip().replace('; open squares require a 3~K extrapolation.','.')
assert 'solid colored lines show weighted Arrhenius fits' in p['caption_tex']
assert 'open squares' not in p['caption_tex']
note='User selected the eta versus temperature Arrhenius variant and identical filled black squares for all Takeda correlations, without the separate extrapolation legend. Input values, uncertainties and experimental correlation formulas are unchanged.'
if note not in p['notes']:p['notes'].append(note)
assert json.dumps([p['composition'],p['temperature']],sort_keys=True)==old_points
path.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Selected',p['version'])
