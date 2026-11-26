"""Create window manifests for independent order-parameter umbrella runs at 1000 K."""
import argparse
import csv
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("series")
    p.add_argument("nx", type=int)
    p.add_argument("nz", type=int)
    p.add_argument("cap", type=float)
    p.add_argument("mu", type=float)
    p.add_argument("rep", type=int)
    p.add_argument("burn", type=int)
    p.add_argument("sweeps", type=int)
    p.add_argument("--axial-free", type=int, choices=[0, 1], default=1)
    p.add_argument("--windows", type=int, default=8)
    p.add_argument("--center-start", action="store_true")
    args = p.parse_args()
    n = 2 * args.nx**2 * args.nz
    series = Path(__file__).resolve().parent / "series" / args.series
    series.mkdir(parents=True, exist_ok=True)
    fields = "run N shape cap mu_eV K_eV q0 seed start burn sweeps stride axial_free".split()
    with (series / "windows.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(fields)
        for j in range(args.windows):
            center = .035 + .875*j/(args.windows-1)
            start = "window" if args.center_start else ("random" if args.rep % 2 == 0 else "ordered")
            seed = 810000 + n*3331 + round(args.cap*100)*239 + args.rep*173 + j
            writer.writerow([f"{args.series}-q{j:02}", n, f"{args.nx}x{args.nx}x{args.nz}",
                             args.cap, args.mu, round(5+n*.064, 4), round(center, 6), seed, start,
                             args.burn, args.sweeps, 1, args.axial_free])


if __name__ == "__main__":
    main()
