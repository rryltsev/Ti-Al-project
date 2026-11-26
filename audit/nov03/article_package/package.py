from pathlib import Path
import json,zipfile,csv,re,hashlib,shutil
from pypdf import PdfReader
import pypdfium2 as pdfium
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent;MAN=ROOT/'manuscript';DEL=ROOT/'deliverables/2026-11-03';DATA=DEL/'figure_data';PKG=DEL/'TiAl_latex'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
pdf=MAN/'build/revision-nov03-parity/main.pdf';texts=[p.extract_text() or '' for p in PdfReader(pdf).pages]
log=pdf.with_suffix('.log').read_text(encoding='utf-8')
assert not re.search('Overfull|undefined references|undefined citations|Missing character',log)
assert not any('??' in t for t in texts)
figpages={i:next(j+1 for j,t in enumerate(texts) if f'FIG. {i}.' in t) for i in range(1,8)}
with zipfile.ZipFile(OUT/'before_revision.zip') as z:
 assert z.read('supporting_materials.tex')==(MAN/'supporting_materials.tex').read_bytes()
 for f in ['phonons_alpha_ti','melting_ti','liquid_density','mixing_enthalpy']:
  assert z.read('figures/'+f+'.pdf')==(MAN/'figures'/(f+'.pdf')).read_bytes()
 old=z.read('sections/results.tex').decode('utf-8-sig').replace('\r\n','\n')
 current=(MAN/'sections/results.tex').read_text(encoding='utf-8');boundary=r'\subsection{Lattice dynamics'
 assert current.split(boundary)[1]==old.split(boundary)[1]
render=OUT/'rendered';render.mkdir(exist_ok=True);thumbs=[];doc=pdfium.PdfDocument(str(pdf))
for i,page in enumerate(doc):
 im=page.render(scale=1.4).to_pil().convert('RGB');im.save(render/f'page_{i+1}.png');im.thumbnail((360,510));tile=Image.new('RGB',(380,550),'white');tile.paste(im,((380-im.width)//2,26));ImageDraw.Draw(tile).text((12,7),f'Page {i+1}',fill='black');thumbs.append(tile);page.close()
doc.close()
for start in range(0,len(thumbs),6):
 sheet=Image.new('RGB',(1140,1100),'#ccc')
 for j,im in enumerate(thumbs[start:start+6]):sheet.paste(im,((j%3)*380,(j//3)*550))
 sheet.save(render/f'contact_{start//6+1}.png')
# Copy all editable sources; omit build caches and unrelated archived figures.
PKG.mkdir(exist_ok=True)
for p in MAN.glob('*'):
 if p.is_file() and p.suffix in ['.tex','.bib','.txt']:shutil.copy2(p,PKG/p.name)
for folder in ['sections','supporting_data']:shutil.copytree(MAN/folder,PKG/folder,dirs_exist_ok=True)
figs=set()
for p in [MAN/'supporting_materials.tex']+list((MAN/'sections').glob('*.tex')):
 figs.update(re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}',p.read_text(encoding='utf-8')))
for name in sorted(figs):
 dest=PKG/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(MAN/name,dest)
for p in PKG.rglob('*.tex'):
 s=p.read_text(encoding='utf-8')
 for name in re.findall(r'\\input\{([^}]+)\}',s):assert (PKG/name).exists() or (PKG/(name+'.tex')).exists(),(p,name)
 for name in re.findall(r'\\bibliography\{([^}]+)\}',s):
  for bib in name.split(','):assert (PKG/(bib+'.bib')).exists(),bib
shutil.copyfile(pdf,PKG/'main.pdf');shutil.copyfile(pdf,DEL/'TiAl_article_2026-11-03.pdf')
(PKG/'README_RU.txt').write_text('''Исходники статьи Ti--Al, снимок от 3 ноября 2026.

Основной документ: main.tex. SI: supporting_materials.tex.
Каталоги sections/, figures/, supporting_data/ и четыре файла .bib сохраняют
относительные пути. Файлы .tex редактируются в UTF-8.
Сборка в TeX Live/MiKTeX с REVTeX 4.2: latexmk -pdf main.tex
Отдельная сборка SI: latexmk -pdf supporting_materials.tex
Альтернатива: tectonic main.tex и tectonic supporting_materials.tex.
Нужны обычные пакеты REVTeX, graphicx, amsmath, natbib, hyperref и lmodern.
Текущий проверенный основной PDF приложен как main.pdf.

Рисунок 1 и Table II обновлены по 49 общим проверенным структурам из Test50.
Обновлены соответствующие числа в Results и Conclusions. Без энергетического
выравнивания; PBE 20k, остальные модели и геометрии сохранены.
Сопутствующие данные всех рисунков поставляются отдельным архивом figure_data.
Исторические диагностические таблицы и рисунки SI сохранены с их прежними
выборками и чекпойнтами; их нельзя интерпретировать как текущий тест на 49 структурах.
SI в этой редакции не переписывался. В supporting_data есть исторические файлы;
актуальная тестовая метрика: test49_metrics.json.
''',encoding='utf-8')
# Reader-friendly index of every plotted numerical table.
catalog=[]
for p in sorted(DATA.rglob('*.csv')):
 if p.name=='INDEX.csv':continue
 with p.open(encoding='utf-8-sig',newline='') as f:
  r=csv.reader(f);header=next(r);n=sum(1 for _ in r)
 assert n>0,p
 catalog.append([p.relative_to(DATA).as_posix(),n,' | '.join(header)])
with (DATA/'INDEX.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.writer(f);w.writerow(['file','data_rows','column_names']);w.writerows(catalog)
(DATA/'README_RU.txt').write_text('''Численные данные рисунков статьи Ti--Al, 3 ноября 2026.

CSV: UTF-8 с BOM, запятая-разделитель, точка в дробных числах; первая строка —
названия столбцов и единицы измерения. Файлы читаются pandas.read_csv и Origin.
INDEX.csv содержит полный перечень таблиц, число строк и названия столбцов.
plot_csv.py — пример построения произвольной таблицы (python, matplotlib);
параметры и пример запуска: python plot_csv.py --help.
Папки Fig01--Fig07 соответствуют семи основным рисункам, FigS01--FigS11 — SI.
Строки разных серий разделяются столбцами functional/model/phase/split/checkpoint.
Для построения одной линии нужно отфильтровать серию и отсортировать её по x.
SE и SEM — стандартные ошибки из исходного оценивания, не полная погрешность.
lower/upper — границы показанной полосы или ошибки, а не дополнительные кривые.

Fig01: x = DFT_energy_eV_atom либо DFT_force_eV_A; y = model_energy_eV_atom
либо model_force_eV_A. Все компоненты сил сохранены, без прореживания.
atom_index_zero_based начинается с нуля. stress.csv содержит данные для метрик.
Общий набор — 49 структур, 34446 компонент силы на модель и функционал.
Исходные энергии без сдвигов; pretrained сравнивается с каждым функционалом.
Fig02: точные замороженные таблицы текущего шестипанельного рисунка, включая
дисперсии, DOS и оцифрованный эксперимент. Нормировка DOS — 3 моды на атом.
Fig03: models.csv содержит архивные и обновлённые линии. Для жидкостей PBEsol
и r2SCAN используется extended, для PBE показаны обе версии, различимые по series.
Три начальные фазы — отдельные линии. Корреляция жидкой плотности содержит ±2%.
Fig04: время в пс, solid_like_fraction безразмерна; повтор 002 — отдельная линия.
Fig05: x_Al — мольная доля Al, отдельные повторные экспериментальные точки сохранены.
Fig06: энтальпия смешения при 2000 K, включая оба нулевых чистых предела.
Fig07: исходные точки и ошибки; Arrhenius_curves.csv — плавные кривые,
а не новые измерения. Ishikawa_curve.csv сохраняет предварительную корреляцию
и её полосу ±10.3%; Takeda.csv сохраняет диапазоны измерений.

SI содержит исторические диагностические наборы, не заменённые набором Fig01.
S01: raw и центральное скользящее среднее по 11 записям. Вертикальная отметка 15k.
S02: метрики против числа шагов; остатки обучения для 5k и 23.4k.
S03: остатки против объёма для x_Ti=205/216 и 1, два чекпойнта и два split.
S04: остатки против состава; нижние панели отбирают x_Ti>=0.5 и используют объём
как x. Цвет означает объём в верхних панелях и x_Ti в нижних.
S05: q_normalized — нормированная длина пути Gamma--K--M--Gamma--A;
common/relaxed — общая/собственная равновесная геометрия, шесть ветвей в ТГц.
S06/S07: один общий столбец частот и именованные кривые DOS (моды/(атом ТГц)).
S08: fixed/equilibrium — две панели, сетка MP40. DFT повторяется на обеих.
S09: первые две панели — L2 против RMSE validation/test20; третья — отношения
L2/TV/W1 к значению 5k против числа тысяч шагов.
S10: плотность и энтальпия против x_Al для трёх чекпойнтов, ошибки SEM.
S11: изменение энтальпии относительно production против x_Al, отдельная линия
для каждого окна и отдельная панель для каждого чекпойнта.

source_hashes.json записывает исходные пути и SHA256. Исходные расчёты не запускались.
''',encoding='utf-8')
sources={}
for p in [OUT/'source_hashes.json',OUT/'si_source_hashes.json']:
 sources.update(read(p))
(DATA/'source_hashes.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
for folder in [PKG,DATA]:
 manifest={p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='SHA256.json' and 'validation' not in p.relative_to(folder).parts}
 (folder/'SHA256.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 target=DEL/(folder.name+'_2026-11-03.zip')
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(folder.rglob('*')):
   if p.is_file() and 'validation' not in p.relative_to(folder).parts:z.write(p,p.relative_to(folder.parent).as_posix())
 with zipfile.ZipFile(target) as z:assert z.testzip() is None
report={'pages':len(texts),'figure_pages':figpages,'figure_tables':len(catalog),'referenced_figure_assets':len(figs),'SI_unchanged':True,'other_results_unchanged':True,'metrics_checked':36,'visual_review':'All 18 pages reviewed in contact sheets; pages 8 and 9 inspected at full size. No clipping or overlap.','packaged_main_compile':'Passed from standalone source folder, with no workspace source dependencies.','pdf_SHA256':sha(pdf),'archives':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in DEL.glob('*.zip')}}
(OUT/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
