"""Fetch public bibliographic metadata; no private files are uploaded."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import quote
import json

OUT=Path(__file__).resolve().parent
DOIS={
    "Clemens2013":"10.1002/adem.201200231",
    "Zope2003":"10.1103/PhysRevB.68.024102",
    "Kim2016":"10.1016/j.commatsci.2016.03.038",
    "Seko2020":"10.1103/PhysRevB.102.174104",
    "Zhao2024":"10.1103/PhysRevB.110.184115",
    "Shen2025":"10.1039/D5TC02960K",
    "Khazieva2024":"10.1063/5.0222355",
    "Song2024":"10.1038/s41467-024-54554-x",
    "Batatia2025":"10.1063/5.0297006",
    "Jinnouchi2019":"10.1103/PhysRevB.100.014105",
    "Perdew1996":"10.1103/PhysRevLett.77.3865",
    "Perdew2008":"10.1103/PhysRevLett.100.136406",
    "Furness2020":"10.1021/acs.jpclett.0c02405",
}
def get(item):
    key,doi=item
    path=OUT/(key+".metadata.json")
    try:
        if path.exists():
            data=json.loads(path.read_text(encoding="utf-8"))
        else:
            req=Request("https://api.crossref.org/works/"+quote(doi,safe=""),headers={"User-Agent":"TiAl-manuscript-bibliography/1.0"})
            with urlopen(req,timeout=20) as f:
                data=json.load(f)["message"]
            path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf-8")
        print(key, data.get("title"), data.get("volume"), data.get("page"),data.get("article-number"),data.get("published"),flush=True)
    except Exception as exc:
        print(key,type(exc).__name__,str(exc),flush=True)
with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(get,DOIS.items()))
