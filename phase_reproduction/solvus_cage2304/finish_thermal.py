"""Rebuild the separate and joint temperature estimates on all verified new data."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from threadpoolctl import threadpool_limits
from pooled_wall import HERE, fit, source_paths, difference
from pooled_contacts import main as fit_contacts

OUT = HERE/"analysis"
GRID = [1242,1243,1244,1245]
HIGH = [1254,1262,1269]


def fresh(dest, paths, stride, draws,contacts=False):
    if not dest.is_file():
        return False
    base = HERE if contacts else HERE.parent
    z = json.loads(dest.read_text(encoding="utf8"))
    return (set((base/p).resolve() for p in z["manifests"])==set(paths)
            and z.get("thinning_stride",z.get("stride"))==stride and z["draws"]==draws
            and not z.get("pending_at_fit"))


def estimates(wall,temps,name,stride=12,draws=64):
    paths,_=source_paths(wall,temps,None,False,False)
    dest=OUT/(name+".json")
    if not fresh(dest,paths,stride,draws):
        a=SimpleNamespace(wall=wall,temperatures=temps,fields=None,available=False,previous=False,
                          stride=stride,cut=.37,bounds=[1200.,1320.,3.43,3.68],draws=draws,out=dest)
        fit(a)


def contacts(wall):
    paths,_=source_paths(wall,GRID,None,False,False)
    dest=OUT/f"contacts_W{round(100*wall)}_1243.json"
    if not fresh(dest,paths,12,64,True):
        fit_contacts(SimpleNamespace(wall=wall,sources=GRID,stride=12,cut=.37,target=1243.,draws=64,out=dest))


def main(task):
    if task in ("wide","all"):
        estimates(.40,GRID,"W40_grid")
        estimates(.40,[1239,*GRID],"W40_all")
        estimates(.40,GRID,"W40_stride6",stride=6,draws=0)
        contacts(.40)
    if task in ("tight","all"):
        estimates(.35,HIGH,"W35_near")
        estimates(.35,[*GRID,*HIGH],"W35_all")
        estimates(.35,HIGH,"W35_stride6",stride=6,draws=0)
        contacts(.35)
    if task in ("difference","all"):
        difference(SimpleNamespace(tight=OUT/"W35_all.json",loose=OUT/"W40_all.json",out=OUT/"wall_shift.json"))
        difference(SimpleNamespace(tight=OUT/"W35_near.json",loose=OUT/"W40_grid.json",out=OUT/"wall_shift_local.json"))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task",nargs="?",choices=["wide","tight","difference","all"],default="all")
    with threadpool_limits(limits=1):
        main(parser.parse_args().task)
