from pathlib import Path
import html
import json
import re
import unicodedata

HERE=Path(__file__).resolve().parent
OUT=HERE.parents[1]/"manuscript"/"references.bib"
JOURNALS={
    "Zope2003":"Phys. Rev. B", "Kim2016":"Comput. Mater. Sci.",
    "Seko2020":"Phys. Rev. B", "Zhao2024":"Phys. Rev. B",
    "Shen2025":"J. Mater. Chem. C", "Khazieva2024":"J. Chem. Phys.",
    "Song2024":"Nat. Commun.", "Batatia2025":"J. Chem. Phys.",
    "Jinnouchi2019":"Phys. Rev. B", "Perdew1996":"Phys. Rev. Lett.",
    "Perdew2008":"Phys. Rev. Lett.", "Furness2020":"J. Phys. Chem. Lett.",
}
SPECIAL={"ø":r"{\o}","Ø":r"{\O}","ł":r"{\l}","Ł":r"{\L}","ß":r"{\ss}","æ":r"{\ae}"}
ACCENTS={"\u0301":"'","\u0300":"`","\u0302":"^","\u0308":'"',"\u0303":"~",
         "\u030c":"v","\u0306":"u","\u030b":"H","\u0327":"c","\u030a":"r","\u0304":"="}
def tex(text):
    text=html.unescape(re.sub("<[^>]*>","",text))
    text=text.replace("–","--").replace("—","---").replace("&",r"\&").replace("’","'")
    text=unicodedata.normalize("NFD",text)
    out=[]
    for char in text:
        if char in ACCENTS and out:
            out[-1]="{\\"+ACCENTS[char]+"{"+out[-1]+"}}"
        else: out.append(SPECIAL.get(char,char))
    return re.sub(r"\s+"," ","".join(out)).strip()

entries=[r"""% Bibliographic metadata checked against publisher PDFs / Crossref, 20 October 2026.
% Public preprints are explicitly marked as preprints.
@article{Clemens2013,
  author = {Helmut Clemens and Svea Mayer},
  title = {{Design, Processing, Microstructure, Properties, and Applications of Advanced Intermetallic TiAl Alloys}},
  journal = {Adv. Eng. Mater.},
  volume = {15},
  pages = {191--215},
  year = {2013},
  doi = {10.1002/adem.201200231}
}
"""]
for key,journal in JOURNALS.items():
    d=json.loads((HERE/(key+".metadata.json")).read_text(encoding="utf-8"))
    authors=" and ".join(tex(a["family"])+", "+tex(a.get("given","")) for a in d["author"])
    title=tex(d["title"][0])
    if key=="Furness2020": title=title.replace("r2SCAN",r"$r^{2}$SCAN")
    fields=dict(author=authors,title="{"+title+"}",journal=journal,volume=d["volume"],
                pages=d.get("page",d.get("article-number","")).replace("-","--"),
                year=str(d["published"]["date-parts"][0][0]),doi=d["DOI"])
    entries.append("@article{"+key+",\n"+",\n".join("  "+k+" = {"+v+"}" for k,v in fields.items())+"\n}\n")
entries.append(r"""@misc{Kato2026,
  author = {Yuna Kato and J{\"u}rgen Brillo and Dirk Holland-Moritz and Fan Yang and Thomas C. Hansen and Thomas Voigtmann and Linnea Heitmeier},
  title = {{Molecular Dynamics simulations of Al-Ti metallic alloy melts using a transferable machine-learning potential}},
  year = {2026},
  eprint = {2604.26362},
  archivePrefix = {arXiv},
  primaryClass = {cond-mat.mtrl-sci},
  doi = {10.48550/arXiv.2604.26362},
  note = {Preprint}
}

@misc{Li2026DPA4,
  author = {Tiancheng Li and Wentao Li and Anyang Peng and Jianming Xue and Linfeng Zhang and Duo Zhang and Han Wang},
  title = {{DPA4: Pushing the Accuracy-Cost Frontier of Interatomic Potentials with EMFA SO(2) Convolution}},
  year = {2026},
  eprint = {2606.02419},
  archivePrefix = {arXiv},
  primaryClass = {physics.chem-ph},
  doi = {10.48550/arXiv.2606.02419},
  note = {Preprint, version 4}
}
""")
OUT.write_text("\n".join(entries),encoding="utf-8")
print("Wrote",OUT,"with",len(JOURNALS)+3,"entries")
intro=(OUT.parent/"sections"/"introduction.tex").read_text()
keys=set(re.findall(r"@\w+\{([^,]+)",OUT.read_text(encoding="utf-8")))
cited={key for group in re.findall(r"\\cite\{([^}]+)\}",intro) for key in group.split(",")}
assert cited<=keys,cited-keys
plain=re.sub(r"\\cite\{[^}]*\}","",intro)
print("Approximate introduction word count:",len(plain.split()))
print("Cited keys:",len(cited))
