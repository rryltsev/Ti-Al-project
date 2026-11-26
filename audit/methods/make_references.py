"""Fetch public DOI metadata and write the Methods bibliography."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import html
import json
import re
import unicodedata
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
REFS = {
    "Larsen2017": ("10.1088/1361-648X/aa680e", "J. Phys.: Condens. Matter"),
    "Cheng2020": ("10.1021/acs.accounts.0c00403", "Acc. Chem. Res."),
    "Bartok2013": ("10.1103/PhysRevB.87.184115", "Phys. Rev. B"),
    "Kresse1996": ("10.1103/PhysRevB.54.11169", "Phys. Rev. B"),
    "Blochl1994": ("10.1103/PhysRevB.50.17953", "Phys. Rev. B"),
    "Kresse1999": ("10.1103/PhysRevB.59.1758", "Phys. Rev. B"),
    "Thompson2022": ("10.1016/j.cpc.2021.108171", "Comput. Phys. Commun."),
    "Lechner2008": ("10.1063/1.2977970", "J. Chem. Phys."),
    "Togo2015": ("10.1016/j.scriptamat.2015.07.021", "Scr. Mater."),
    "Toraman2025": ("10.1021/acs.jcim.5c01475", "J. Chem. Inf. Model."),
    "Zhang2015": ("10.1021/acs.jctc.5b00351", "J. Chem. Theory Comput."),
}

def fetch(item):
    key, (doi, journal) = item
    cache = HERE / (key + ".metadata.json")
    if cache.exists():
        return key, json.loads(cache.read_text(encoding="utf-8"))
    req = Request("https://api.crossref.org/works/" + quote(doi, safe=""),
                  headers={"User-Agent": "TiAl-Manuscript-Bibliography/1.0"})
    for attempt in range(3):
        try:
            with urlopen(req, timeout=25) as response:
                data = json.load(response)["message"]
            break
        except HTTPError as error:
            if error.code != 429 or attempt == 2:
                raise
            time.sleep(4)
    cache.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return key, data

SPECIAL = {"ø":r"{\o}", "Ø":r"{\O}", "ł":r"{\l}", "Ł":r"{\L}", "ß":r"{\ss}", "æ":r"{\ae}"}
ACCENTS = {"\u0301":"'", "\u0300":"`", "\u0302":"^", "\u0308":'"', "\u0303":"~",
           "\u030c":"v", "\u0306":"u", "\u030b":"H", "\u0327":"c", "\u030a":"r", "\u0304":"="}

def tex(text):
    text = html.unescape(re.sub("<[^>]*>", "", text))
    text = text.replace("–", "--").replace("—", "---").replace("−", "-").replace("&", r"\&").replace("’", "'")
    out = []
    for char in unicodedata.normalize("NFD", text):
        if char in ACCENTS and out:
            out[-1] = "{\\" + ACCENTS[char] + "{" + out[-1] + "}}"
        else:
            out.append(SPECIAL.get(char, char))
    return re.sub(r"\s+", " ", "".join(out)).strip()

if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=1) as pool:
        metadata = dict(pool.map(fetch, REFS.items()))
    entries = ["% Public metadata checked on 20 October 2026.\n"]
    for key, (doi, journal) in REFS.items():
        d = metadata[key]
        fields = {
            "author": " and ".join(tex(a["family"]) + ", " + tex(a.get("given", "")) for a in d["author"]),
            "title": "{" + tex(d["title"][0]) + "}",
            "journal": journal,
            "volume": d["volume"],
            "pages": d.get("page", d.get("article-number", "")).replace("-", "--"),
            "year": str(d.get("published-print", d["published"])["date-parts"][0][0]),
            "doi": doi,
        }
        entries.append("@article{" + key + ",\n" +
                       ",\n".join("  " + k + " = {" + v + "}" for k, v in fields.items()) + "\n}\n")
        print(key, fields["year"], fields["volume"], fields["pages"], d["title"][0], flush=True)
    entries.append(r"""@misc{Hu2021,
  author = {Hu, Edward J. and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  title = {{LoRA: Low-Rank Adaptation of Large Language Models}},
  year = {2021},
  eprint = {2106.09685},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL}
}
""")
    (HERE.parents[1] / "manuscript" / "methods_references.bib").write_text("\n".join(entries), encoding="utf-8")
