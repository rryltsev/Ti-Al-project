"""Check composition and volume measures against exact finite-N probabilities."""
import argparse
import csv
import math
from pathlib import Path
import subprocess
import tempfile

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("executable", type=Path)
    args = p.parse_args()
    motifs = Path(__file__).resolve().parent / "static/motifs.txt"
    n, t, c0, kappa = 96, 1000, .19, 100.
    kt = 8.617333262145e-5*t
    # Independent occupations plus the exact finite-size quadratic restraint.
    counts = np.arange(n+1)
    log_p = np.array([math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1) for k in counts])
    log_p -= n*kappa*(counts/n-c0)**2
    p_n = np.exp(log_p - log_p.max())
    p_n /= p_n.sum()
    with tempfile.TemporaryDirectory() as folder:
        tmp = Path(folder)
        values = np.zeros((5,21))
        values[:,0] = np.arange(15.5, 20., 1)
        # Pure affine slope in v gives an exact truncated exponential,
        # which tests the flat dv convention independently of chemical moves.
        values[:,1] = .003*(values[:,0]-17.5)
        np.savetxt(tmp/"coeff",values,fmt="%.17g")
        cmd = [str(args.executable.resolve()),str(motifs),str(tmp/"coeff"),"4","4","3",str(t),
               str(c0),str(kappa),"0","500","6500","1","20261124","random","ideal","1",str(tmp/"out.csv")]
        subprocess.run(cmd,check=True)
        with (tmp/"out.csv").open() as f:
            rows=list(csv.DictReader(f))
        measured_n=np.array([int(r["n_Al"]) for r in rows])
        measured_v=np.array([float(r["volume_A3_atom"]) for r in rows])
        mean_n=counts@p_n
        rate=n*.003*.5/kt
        width=values[-1,0]-values[0,0]
        mean_v=values[0,0]+1/rate-width/math.expm1(rate*width)
        print(f"nAl exact {mean_n:.6f}, sampled {measured_n.mean():.6f}")
        print(f"v exact {mean_v:.6f}, sampled {measured_v.mean():.6f}")
        if abs(measured_n.mean()-mean_n)>.09 or abs(measured_v.mean()-mean_v)>.1:
            raise RuntimeError("sampler fails finite-system equilibrium check")


if __name__=="__main__":
    main()
