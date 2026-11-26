"""Refresh all direct-physics tables, profile curves, controls, and cross-cage comparisons."""
import argparse
import time

from isotherms import HERE,STUDY
from run import combine_csv
from temperature_zeros import generate as generate_T
from matched_cages import compare as compare_cages
from bracket_checks import assess
from profiles import generate as generate_profiles
from free_chain_check import compare as compare_free
from archive_comparison import compare as compare_archive
from resolution import calculate
from field_profiles import update as write_field_curves


def export():
    combine_csv("scout");combine_csv("final")
    if not list(HERE.glob("final_T[0-9]*.json")):return
    generate_T();compare_cages();assess();generate_profiles();compare_free();compare_archive();calculate();write_field_curves()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--watch",action="store_true")
    args=p.parse_args()
    seen=None
    while True:
        finals=sorted(HERE.glob("final_T[0-9]*.json"))
        current=[(f.name,f.stat().st_mtime_ns) for f in finals]
        if current!=seen:
            if args.watch:
                if finals:
                    generate_profiles();compare_free();compare_archive();calculate()
            else:
                export()
            seen=current
        if not args.watch:break
        # Multiple field-control jobs continue independently of final analysis.
        manifests=list((STUDY/"atomistic"/"series").glob("*/windows.csv"))
        statuses=list((STUDY/"atomistic"/"series").glob("*/window_status.json"))
        if len(manifests)>=52 and len(manifests)==len(statuses) and len(finals)>=8:
            from isotherms import discover
            groups,pending=discover(STUDY/"atomistic"/"series")
            if not pending and len(finals)==len(groups):
                export();break
        time.sleep(30)
