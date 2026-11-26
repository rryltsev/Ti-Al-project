from pathlib import Path
import json
from pypdf import PdfReader

OUT = Path(__file__).resolve().parent
OUT.mkdir(exist_ok=True)
paths = sorted(Path(r"C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\Potentials\ML").glob("*.pdf"))
paths.append(Path(r"C:\SCIENCE\MY_WORK\MY ARTICLES\REPRINTS\2024\2024 JCP Al-Cu Transfer Learning.pdf"))
index = []
for path in paths:
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    target = OUT / (path.stem + ".txt")
    target.write_text("\n\n".join(f"=== PDF PAGE {i+1} ===\n{text}" for i, text in enumerate(pages)), encoding="utf-8")
    index.append(dict(path=str(path), pages=len(pages), text=str(target),
                      metadata={str(k): str(v) for k,v in (reader.metadata or {}).items()}))
    print("\nFILE:", path.name, "\nPAGES:", len(pages), "\n", pages[0][:3300])
(OUT/"source_index.json").write_text(json.dumps(index,indent=2,ensure_ascii=False),encoding="utf-8")
