"""Compare the temperature-dependent executable with the archived 1000 K kernel."""
import argparse
import csv
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

HERE=Path(__file__).resolve().parent


def check(old, new):
    model=HERE.parents[1]/"inputs/fitted.mtp"
    settings=["4","3","0.4","3.29","12",".48","27516","window","10","35","1","1"]
    tables={}
    with tempfile.TemporaryDirectory() as tmp:
        dest=Path(tmp)
        for name,exe,t in [("original",old,()),("t1000",new,("1000",)),("t600",new,("600",)),
                           ("t1273",new,("1273.15",))]:
            subprocess.run([str(exe),str(model),*settings,name,*t],cwd=dest,check=True)
            with (dest/(name+".csv")).open() as stream:
                rows=list(csv.DictReader(stream))
            expected = 1000. if not t else float(t[0])
            if t and any(float(row.pop("T_K"))!=expected for row in rows):
                raise ValueError("Recorded T differs from requested T")
            tables[name]=rows
        assert tables["original"]==tables["t1000"]
        assert (dest/"original.state").read_bytes()==(dest/"t1000.state").read_bytes()
        assert tables["t600"]!=tables["t1000"] and tables["t1273"]!=tables["t1000"]
        return {"validation":"Same local compiler, seed, starting configuration and all proposal parameters",
                "reference_at_1000_all_45_frames_and_final_state_identical":True,
                "new_temperatures_have_distinct_trajectories":True,
                "all_trace_temperatures_verified_K":[600,1000,1273.15],"settings":settings,
                "potential_sha256":hashlib.sha256(model.read_bytes()).hexdigest(),
                "source_sha256":{str(f.relative_to(HERE.parents[1])):hashlib.sha256(f.read_bytes()).hexdigest()
                                 for f in [HERE/"sgmc.cpp",HERE.parents[1]/"solvus_1000/sgmc.cpp"]}}


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("old",type=Path)
    cli.add_argument("new",type=Path)
    a=cli.parse_args()
    r=check(a.old.resolve(),a.new.resolve())
    (HERE/"temperature_validation.json").write_text(json.dumps(r,indent=2)+"\n")
    print(json.dumps(r))
