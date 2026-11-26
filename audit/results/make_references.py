"""Fetch DOI metadata for the Results section using the shared formatter."""
from pathlib import Path
import importlib.util

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("methods_bib", HERE.parent / "methods/make_references.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.HERE = HERE
refs = {
    "Souvatzis2007": ("10.1103/PhysRevLett.99.015901", "Phys. Rev. Lett."),
    "Wang2012": ("10.1007/s11434-011-4945-6", "Chin. Sci. Bull."),
    "Brillo2025": ("10.1007/s10765-025-03651-0", "Int. J. Thermophys."),
    "Witusiewicz2008": ("10.1016/j.jallcom.2007.10.061", "J. Alloys Compd."),
    "Takeda2025": ("10.1007/s10765-025-03624-3", "Int. J. Thermophys."),
    "Ishikawa2012": ("10.1088/0957-0233/23/2/025305", "Meas. Sci. Technol."),
}
entries = ["% Results references: DOI metadata checked on 20 October 2026.\n"]
for key, (doi, journal) in refs.items():
    _, d = module.fetch((key, (doi, journal)))
    fields = {
        "author": " and ".join(module.tex(a["family"]) + ", " + module.tex(a.get("given", "")) for a in d["author"]),
        "title": "{" + module.tex(d["title"][0]) + "}",
        "journal": journal, "volume": d["volume"],
        "pages": d.get("page", d.get("article-number", "")).replace("-", "--"),
        "year": str(d.get("published-print", d["published"])["date-parts"][0][0]),
        "doi": doi,
    }
    entries.append("@article{" + key + ",\n" +
                   ",\n".join("  " + k + " = {" + v + "}" for k, v in fields.items()) + "\n}\n")
    print(key, fields["year"], fields["volume"], fields["pages"], flush=True)
(HERE.parents[1] / "manuscript/results_references.bib").write_text("\n".join(entries), encoding="utf-8")
