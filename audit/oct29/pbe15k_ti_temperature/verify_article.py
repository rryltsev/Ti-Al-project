"""Verify fixed article selection and byte-identical Figure7 page rendering after build."""
from pathlib import Path
import json,hashlib,subprocess,time,sys
from pypdf import PdfReader
M=Path(__file__).resolve().parent;W=M.parents[2];C=W/'audit/nov01/viscosity_article_combined'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
b=read(M/'article-build.json');assert b['status']=='compiled';pdf=Path(b['article_pdf']);r=PdfReader(pdf);pages=[i+1 for i,p in enumerate(r.pages) if 'FIG. 7.' in p.extract_text()];assert len(r.pages)==18 and pages==[14]
p=read(C/'figure_profile.json');m=read(C/'figure_manifest.json');assert sha(C/'figure_profile.json')=='8ca5307ab484cf7963593cdd0c36bd6dca3fd2800cb8e1893fa164874365cfe8';assert m['composition']==p['composition'] and m['temperature']==p['temperature'];cap=m['manuscript_caption_tex'];assert cap in (W/'manuscript/sections/results.tex').read_text(encoding='utf-8');assert not any(x in cap.lower() for x in ['checkpoint','5k','15k'])
im=C/'article_current_review.png';before=sha(im);pop='C:/Users/Roman Ryltsev/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin/pdftoppm.exe';subprocess.run([pop,'-f','14','-l','14','-singlefile','-r','100','-png',str(pdf),str(im.with_suffix(''))],check=True,capture_output=True);assert before==sha(im),'Changed render requires visual inspection before QA PASS'
qa=dict(checked_unix=time.time(),pages=18,figure_page=14,render_identical_to_previously_inspected=True,profile_preserved=True,caption_preserved=True,article_build=b)
(M/'article-qa-latest.json').write_text(json.dumps(qa,indent=2)+'\n')
for folder in sys.argv[1:]:
 out=M/folder;assert out.is_dir() and out.is_relative_to(M);(out/'article-qa.json').write_text(json.dumps(qa,indent=2)+'\n')
print('Article QA PASS;fixed profile/caption and Figure7 render preserved.')
