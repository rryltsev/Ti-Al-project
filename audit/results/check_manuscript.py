"""Check citations/labels and render the compiled manuscript for visual review."""
from pathlib import Path
import json
import re
import argparse
from pypdf import PdfReader
import pypdfium2 as pdfium

HERE = Path(__file__).resolve().parent
MANUSCRIPT = HERE.parents[1] / "manuscript"
parser = argparse.ArgumentParser()
parser.add_argument("--pdf", type=Path, default=MANUSCRIPT / "build/results/main.pdf")
parser.add_argument("--out", type=Path, default=HERE / "rendered")
args = parser.parse_args()
pdf = args.pdf.resolve()
out = args.out.resolve()
out.mkdir(parents=True, exist_ok=True)
text = "\n".join(p.extract_text() or "" for p in PdfReader(pdf).pages)
(out / "main.txt").write_text(text, encoding="utf-8")
tex = "\n".join(p.read_text(encoding="utf-8") for p in [
    MANUSCRIPT / "main.tex", *sorted((MANUSCRIPT / "sections").glob("*.tex"))])
labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", tex))
cites = {x.strip() for group in re.findall(r"\\(?:cite|onlinecite)\{([^}]+)\}", tex)
         for x in group.split(",")}
bib = "\n".join(p.read_text(encoding="utf-8") for p in MANUSCRIPT.glob("*.bib"))
keys = set(re.findall(r"@\w+\{([^,]+),", bib))
log = (pdf.parent / "main.log").read_text(encoding="utf-8")
report = {
    "pdf": str(pdf),
    "pages": len(PdfReader(pdf).pages),
    "citations": len(cites),
    "missing_labels": sorted(refs - labels),
    "missing_citations": sorted(cites - keys),
    "unresolved_pdf_markers": text.count("??"),
    "overfull": "Overfull" in log,
    "missing_characters": "Missing character" in log,
}
assert not report["missing_labels"] and not report["missing_citations"]
assert not report["unresolved_pdf_markers"] and not report["overfull"]
assert not report["missing_characters"]
doc = pdfium.PdfDocument(str(pdf))
for i in range(len(doc)):
    page = doc[i]
    page.render(scale=1.55).to_pil().save(out / f"page_{i+1}.png")
    page.close()
doc.close()
(HERE / "build_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
(out / "build_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
