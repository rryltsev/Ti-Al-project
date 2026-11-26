"""Build the two user-requested article figure variants without publishing either."""
from pathlib import Path
import copy, hashlib, json, subprocess, sys
ROOT=Path(__file__).resolve().parent
WORK=ROOT.parents[2]
OUT=ROOT/'variants'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
save=lambda p,d:Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
base=load(ROOT/'figure_profile.json')
candidate_path=Path(base['intermediate_temperature_search']['candidate_source'])
assert sha(candidate_path)==base['sources_sha256'][str(candidate_path)]
candidate=load(candidate_path)
assert candidate['T']==1941 and candidate['dft']=='pbe' and candidate['c_Ti']==1.0
assert candidate==base['intermediate_temperature_search']['candidate']
protected=[ROOT/'figure_profile.json',ROOT/'viscosity_combined.pdf',ROOT/'viscosity_combined.png',WORK/'manuscript/figures/viscosity.pdf',WORK/'manuscript/build/ti-temperature/main.pdf']
before={str(p):sha(p) for p in protected}
profiles=[]
for name,include in [('three_points',False),('four_points',True)]:
    dest=OUT/name
    dest.mkdir(parents=True,exist_ok=True)
    p=copy.deepcopy(base)
    p['version']=base['version']+'-'+name
    p['authoritative_article_selection']=False
    p['variant']=name
    p['notes'].append('User requested both alternatives; neither alternative changes the currently published article selection or the production monitor.')
    if include:
        candidate_row=copy.deepcopy(candidate)
        candidate_row.update(checkpoint_step=5000,kind='archived_20ps_comparison_point',source=str(candidate_path))
        p['temperature']['pbe'].append(candidate_row)
        p['temperature']['pbe'].sort(key=lambda x:x['T'])
        p['selection'].pop('temperature_PBE_checkpoint',None)
        p['selection']['temperature_PBE_checkpoint_by_T']={str(r['T']):r['checkpoint_step'] for r in p['temperature']['pbe']}
        p['intermediate_temperature_search']['included']=True
        p['intermediate_temperature_search']['inclusion_authorized_by_user']=True
        p['notes'].append('The additional 1941 K PBE value is the archived checkpoint 5000 result; the temperature curve mixes parameter snapshots. It is not a checkpoint 15000 result.')
        target='the 2100~K value uses full 40~ps trajectories.'
        replacement=target+' The additional PBE value at 1941~K is an archived 20~ps estimate from an earlier parameter set and is included for comparison.'
        assert target in p['caption_tex']
        p['caption_tex']=p['caption_tex'].replace(target,replacement)
    save(dest/'figure_profile.json',p)
    subprocess.run([sys.executable,'-B',str(ROOT/'build_figure.py'),'--profile',str(dest/'figure_profile.json'),'--output-dir',str(dest)],check=True)
    profiles.append(p)
assert profiles[0]['composition']==profiles[1]['composition']==base['composition']
assert profiles[0]['temperature']['scan']==profiles[1]['temperature']['scan']==base['temperature']['scan']
assert [r for r in profiles[1]['temperature']['pbe'] if r['T']!=1941]==base['temperature']['pbe']
assert sha(OUT/'three_points/viscosity_combined.png')==sha(ROOT/'viscosity_combined.png')
assert before=={str(p):sha(p) for p in protected}
save(OUT/'comparison.json',dict(variants=[{'name':p['variant'],'pbe_temperature_points':p['temperature']['pbe']} for p in profiles],original_article_files_unchanged=True,three_point_png_matches_original=True))
(OUT/'README_RU.txt').write_text('Два варианта объединённого рисунка для статьи.\n\nthree_points: PBE при 1873, 2000 и 2100 K.\nfour_points: дополнительно архивная точка PBE при 1941 K, 3.049946163 ± 0.117315378 мПа·с (15 реплик по 20 пс, checkpoint 5000). Остальные температурные точки PBE относятся к checkpoint 15000.\n\nКонцентрационная панель в обоих вариантах использует PBE checkpoint 5000; данные r2SCAN и экспериментальные корреляции совпадают. Обозначения чекпойнтов в легендах и подписях отсутствуют; происхождение записано в JSON. В подписи четырёхточечного варианта указано, что промежуточная точка получена для более раннего набора параметров.\n\nДля каждого варианта сохранены PDF, PNG, SVG, caption.tex и JSON с источниками. В статье сохранён ранее опубликованный вариант с тремя температурными точками.\n',encoding='utf-8')
print('Both variants created; primary article unchanged.')
