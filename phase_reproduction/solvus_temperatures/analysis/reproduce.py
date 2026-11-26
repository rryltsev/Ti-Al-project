"""List (--list), display (--show NAME), or recompute (NAME ...) the reported analyses.

Run with the same Python used for analysis, with NumPy and SciPy installed.
The raw manifests and trajectories are resolved relative to this source file.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
SERIES = HERE.parent / "atomistic" / "series"
RECIPES = {}


def dirs(prefix, suffixes):
    return [SERIES / (prefix + suffix) / "windows.csv" for suffix in suffixes.split()]


def add(name, script, inputs, *options):
    RECIPES[name] = [sys.executable, HERE/(script+".py"), *inputs, *options, "--out", HERE/(name+".json")]


def profile(name, prefix, suffixes, bounds, bootstrap=0, *options):
    add(name, "isotherm", dirs(prefix,suffixes), "--field-bounds",*bounds,
        "--bootstrap",bootstrap,"--seed",26713,"--bandwidth",.04,*options)


profile("600_n96","at600-n96-","l",(3.135,3.213),96)
profile("600_n288","at600-n288-","a b",(3.13,3.205),192)
profile("600_n768_merged","at600-n768-","a b",(3.13,3.19),128)
profile("600_n1400_merged","at600-n1400-","a b",(3.143,3.170),192)
profile("600_n1400_with_conditioned_k0","at600-n1400-","a b pure",(3.143,3.170),96,"--restrict-k0-cut",.53)
profile("600_n288_tight","at600-n288-","tight",(3.12,3.21),128)
profile("600_n768_tight","at600-n768-","tight",(3.137,3.177),96)
profile("600_n1400_tight","at600-n1400-","tight",(3.143,3.170),128)
profile("600_n288_fixed","at600-n288-","fixed-a fixed-b",(3.13,3.2),96)
profile("600_n768_fixed","at600-n768-","fixed",(3.153,3.205),128)

high_prefix = "at1273-n"
free_high = {96:("l h",(3.34,3.70)),288:("l h",(3.37,3.67)),
             768:("u v o p peak sh ex",(3.34,3.73)),
             1400:("pure q r sh rich",(3.39,3.73))}
for n,(names,bounds) in free_high.items():
    tag = f"1273_n{n}" + ("_merged" if n>=768 else "")
    profile(tag,high_prefix+str(n)+"-",names,bounds)
    if n>=768:
        add(f"1273_n{n}_response","response",dirs(high_prefix+str(n)+"-",names),
            "--field-bounds",*bounds,"--bootstrap",64,"--seed",31735)
        for h in ("first","last"):
            add(f"1273_n{n}_response_{h}_half","response",dirs(high_prefix+str(n)+"-",names),
                "--field-bounds",*bounds,"--half",h)
add("1273_small_responses","response",dirs("at1273-n96-","l h")+dirs("at1273-n288-","l h"),
    "--bootstrap",64,"--seed",31735)
profile("1273_n288_fixed","at1273-n288-","fixed-l fixed-h",(3.37,3.73),64)
profile("1273_n288_tight","at1273-n288-","tight",(3.42,3.51))
profile("1273_n768_tight","at1273-n768-","tight",(3.478,3.502))
for n,subset in ((768,"u v"),(1400,"pure rich")):
    for part,program in (("direct_pairs","direct_pairs"),("unbiased_domains","spatial_order")):
        add(f"1273_n{n}_{part}",program,dirs(f"at1273-n{n}-",subset))
for n,subset,bounds in ((768,"o p peak sh ex",(3.413,3.552)),(1400,"q r sh",(3.49,3.547))):
    add(f"1273_n{n}_umbrella_response","response",dirs(f"at1273-n{n}-",subset),
        "--field-bounds",*bounds)
add("1273_n288_fixed_response","response",dirs("at1273-n288-","fixed-l fixed-h"))
add("1273_size_comparison","compare_sizes",[HERE/(x+".json") for x in
    ("1273_n96","1273_n288","1273_n768_merged","1273_n1400_merged")])
add("1273_response_table","response_table",[HERE/(x+".json") for x in
    ("1273_small_responses","1273_n768_response","1273_n1400_response")])
RECIPES["1273_response_table"][-1]=HERE/"1273_response_table.csv"
add("600_size_and_cage","binodal_summary",[HERE/(x+".json") for x in
    ("600_n96","600_n288","600_n768_merged","600_n1400_merged","600_n288_tight",
     "600_n768_tight","600_n1400_tight","600_n288_fixed","600_n768_fixed")])


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names",nargs="*")
    parser.add_argument("--show",action="store_true")
    parser.add_argument("--list",action="store_true")
    args=parser.parse_args()
    if args.list or not args.names:
        print(*RECIPES,sep="\n")
    for name in args.names:
        if name not in RECIPES:
            parser.error("Unknown analysis "+name)
        command=list(map(str,RECIPES[name]))
        if args.show:
            print(subprocess.list2cmdline(command))
        else:
            subprocess.run(command,check=True,env={**os.environ,"OPENBLAS_NUM_THREADS":"1"})
