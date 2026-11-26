from pathlib import Path
import json,re,zipfile,hashlib
import argparse
from pypdf import PdfReader
import pypdfium2 as pdfium
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[3]
MAN=ROOT/'manuscript'
OUT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--build-dir',default='build/revision-oct29')
args=parser.parse_args()
BUILD=MAN/args.build_dir
report={}
for name in ['main','supporting_materials']:
    dest=OUT/'rendered'/name;dest.mkdir(parents=True,exist_ok=True)
    text=[p.extract_text() or '' for p in PdfReader(BUILD/(name+'.pdf')).pages]
    (dest/'text.txt').write_text('\n\n'.join(text),encoding='utf-8')
    log=(BUILD/(name+'.log')).read_text(errors='replace')
    assert not any(x in log for x in ['Overfull','Missing character','undefined references','undefined citations'])
    assert not any('??' in t for t in text)
    doc=pdfium.PdfDocument(str(BUILD/(name+'.pdf')));thumbs=[]
    for i,page in enumerate(doc):
        im=page.render(scale=1.6).to_pil().convert('RGB');im.save(dest/f'page_{i+1}.png')
        im.thumbnail((360,510));tile=Image.new('RGB',(380,550),'white');tile.paste(im,((380-im.width)//2,26))
        ImageDraw.Draw(tile).text((12,7),f'{name} -- {i+1}',fill='black');thumbs.append(tile)
        page.close()
    doc.close()
    for start in range(0,len(thumbs),6):
        sheet=Image.new('RGB',(1140,1100),'#cccccc')
        for j,im in enumerate(thumbs[start:start+6]):sheet.paste(im,((j%3)*380,(j//3)*550))
        sheet.save(dest/f'contact_{start//6+1}.png')
    report[name]={'pages':len(text),'undefined_references':False,'overfull_boxes':False,
        'page_starts':[t[:85].replace('\n',' ') for t in text],
        'pdf_sha256':hashlib.sha256((BUILD/(name+'.pdf')).read_bytes()).hexdigest()}
(OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
