"""Seed six native ATAT 64-site cells at the compositions specified in v16 S1."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def run(cmd, cwd, output=None):
    if output is None:
        subprocess.run(cmd,cwd=cwd,check=True)
    else:
        with output.open("w",encoding="ascii",newline="\n") as stream:
            subprocess.run(cmd,cwd=cwd,check=True,stdout=stream)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mcsqs",type=Path)
    p.add_argument("corrdump",type=Path)
    p.add_argument("lattice",type=Path)
    p.add_argument("clusters",type=Path,help="full cluster list from the ATAT enumeration job")
    p.add_argument("output",type=Path)
    args = p.parse_args()
    mcsqs, corr, lat, clusters = map(lambda path:str(path.resolve(strict=True)),
                                     (args.mcsqs,args.corrdump,args.lattice,args.clusters))
    args.output.mkdir(parents=True,exist_ok=False)
    coordinate_system = Path(lat).read_text().splitlines()[:3]
    metadata = []
    for n_al in (17,20,22):
        for ordered in (False, True):
            label = "preferred" if ordered else "random"
            name = f"{'control' if n_al==22 else 'train'}{n_al}_{label}"
            dest = args.output/name
            dest.mkdir()
            # The specified coordinate system is unchanged in the doubled D019 parent.
            # Only the set of occupations on its eight sites is symmetry-restricted.
            repeat = 2 if ordered else 1
            lines = [*coordinate_system,f"{repeat} 0 0",f"0 {repeat} 0","0 0 1"]
            p_al = (n_al-16)/48 if ordered else n_al/64
            pref = 0
            for i in range(repeat):
                for j in range(repeat):
                    for b in range(2):
                        preferred = ordered and (i+b)%2==0 and (j+b)%2==0
                        species = "Al=1" if preferred else f"Ti={1-p_al:.15g},Al={p_al:.15g}"
                        lines.append(f"{i+b/3:.15g} {j+b/3:.15g} {b/2:.15g} {species}")
                        pref += preferred
            if ordered and pref!=2:
                raise ValueError("unexpected D019 sublattice")
            (dest/"rndstr.in").write_text("\n".join(lines)+"\n",encoding="ascii")
            (dest/"sqscell.out").write_text("1\n\n4 0 0\n0 4 0\n0 0 2\n",encoding="ascii")
            cmd1 = [corr,"-ro","-noe","-nop","-clus","-2=1.02","-l=rndstr.in","-sig=14"]
            run(cmd1,dest)
            seed = 20261124+10*n_al+int(ordered)
            # 3.52 saves the initialized cell before checking the stopsqs sentinel.
            # Take a genuine random permutation; there is no correlation optimization.
            (dest/"stopsqs").touch()
            cmd2 = [mcsqs,"-n=64","-rc",f"-sd={seed}","-ms=0","-sig=14"]
            run(cmd2,dest)
            shutil.copyfile(dest/"bestsqs.out",dest/"str.out")
            atoms = [row.split() for row in (dest/"str.out").read_text().splitlines()
                     if row.split() and row.split()[-1] in ("Ti","Al")]
            if len(atoms)!=64 or sum(row[-1]=="Al" for row in atoms)!=n_al:
                raise ValueError(f"wrong stoichiometry for {name}")
            n_pref_al=0
            for row in atoms:
                x,y,z = map(float,row[:3])
                b = round(2*z)%2
                i,j = round(x-b/3),round(y-b/3)
                n_pref_al += int(row[-1]=="Al" and (i+b)%2==0 and (j+b)%2==0)
            if ordered and n_pref_al!=16:
                raise ValueError("unexpected occupation of preferred sublattice")
            cmd3 = [corr,f"-l={lat}","-c",f"-cf={clusters}","-s=str.out","-sig=14"]
            run(cmd3,dest,dest/"corr.out")
            metadata.append({"id":name,"n_Al":n_al,"n_Al_on_preferred":n_pref_al,"seed":seed,
                             "str_sha256":hashlib.sha256((dest/"str.out").read_bytes()).hexdigest(),
                             "commands":[cmd1,cmd2,cmd3]})
    (args.output/"generation.json").write_text(json.dumps(metadata,indent=2)+"\n")


if __name__=="__main__":
    main()
