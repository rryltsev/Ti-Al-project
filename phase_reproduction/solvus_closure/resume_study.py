"""Retrieve the already submitted UMT series and finish numerical postprocessing."""
import os
from pathlib import Path
import subprocess
import sys


STUDY = Path(__file__).resolve().parent


def run(relative_script, *arguments):
    cmd = [sys.executable, "-B", str(STUDY / relative_script), *arguments]
    subprocess.run(cmd, cwd=STUDY, check=True, env={**os.environ, "OPENBLAS_NUM_THREADS": "1"})


def main():
    run("atomistic/closure_runs.py", "resume")
    run("analysis/resume_direct.py", "--require-all")
    run("check_thermal.py")
    run("thermal_cases.py", "--ready", "--workers", "2")
    half_cases = ["direct_1245_n768", "direct_1245_n1400", "direct_1245_n2304"]
    for part in ("early", "late"):
        run("thermal_cases.py", *half_cases, "--part", part, "--draws", "0", "--ready", "--workers", "2")
    run("size_limit.py")
    run("analysis/n2304_widths.py")
    run("collect_results.py")
    print("Numerical results are in size_dependence.csv, congruent_temperatures.csv and analysis/vertex_size_limit.json.")
    print("The reviewed report and its LaTeX source are under report/.")


if __name__ == "__main__":
    main()
