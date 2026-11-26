"""Compare N8 direct MTP, classical anharmonic TI and bulk harmonic C46."""
import json
import math
from io import BytesIO
from pathlib import Path
from statistics import mean
import tarfile

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "cluster" / "runs"
N = 8
KT = 1000 * 8.617333262145e-5


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def chemistry(energies, degeneracies):
    e0 = min(energies.values())
    z = {key: degeneracies[key]*math.exp(-(e-e0)/KT) for key, e in energies.items()}
    return e0-KT*math.log(sum(z.values())), z["d019"]/sum(z.values())


def jackknife_se(values):
    center = mean(values)
    return math.sqrt((len(values)-1)/len(values)*sum((v-center)**2 for v in values))


def main():
    case = read(RUNS/"alloy-ti-n8-01"/"outputs"/"pilot.json")
    bulk = read(RUNS/"alloy-ti-bulkha-01"/"outputs"/"bulk_ha.json")
    ce = read(HERE/"c46_N8_1000K.json")["orbits"]
    data = {r["name"]: r for r in case["results"]}
    g = {key: row["degeneracy"] for key, row in ce.items()}
    if sum(g.values()) != math.comb(N, 2) or set(g) != set(data) or set(g) != set(bulk):
        raise ValueError("Different decoration sets")
    if any(not r["stable_mesh"] for r in bulk.values()):
        raise ValueError("Invalid bulk harmonic spectrum")
    if any(r["model_sha256"] != case["model_sha256"] for r in bulk.values()):
        raise ValueError("Different potentials")
    f = {name: {} for name in ("C46_static", "direct_static", "C46_bulk_qHA", "direct_bulk_qHA",
                                "direct_bulk_classical_HA", "direct_N8_classical_HA",
                                "direct_N8_quantum_HA", "direct_N8_classical_anharmonic")}
    drops = {k: [] for k in data}
    checks = {}
    for k, row in data.items():
        sta = row["static_eV_cell"]
        f["C46_static"][k] = N*ce[k]["static_eV_atom"]
        f["C46_bulk_qHA"][k] = N*ce[k]["quantum_HA_eV_atom"]
        f["direct_static"][k] = sta
        if abs(sta-N*bulk[k]["static_eV_atom"]) > 1.e-6:
            raise ValueError("Static energies differ in harmonic and TI jobs")
        f["direct_bulk_qHA"][k] = sta + N*bulk[k]["quantum_vib_eV_atom"]
        f["direct_bulk_classical_HA"][k] = sta + N*bulk[k]["classical_vib_eV_atom"]
        f["direct_N8_quantum_HA"][k] = sta + row["harmonic_quantum_eV_cell"]
        harmonic = sta + row["harmonic_classical_eV_cell"]
        f["direct_N8_classical_HA"][k] = harmonic
        f["direct_N8_classical_anharmonic"][k] = harmonic + row["fep_eV"]
        ratios = [math.exp(-r/KT) for r in row["replica_fep_eV"]]
        if not math.isclose(-KT*math.log(mean(ratios)), row["fep_eV"], abs_tol=1e-12):
            raise ValueError("Error reconstructing the pooled endpoint")
        drops[k] = [harmonic - KT*math.log(mean(ratios[:i]+ratios[i+1:])) for i in range(len(ratios))]
        checks[k] = dict(draws=row["draws"], ess_min=row["ess_min"], max_weight=row["max_weight_endpoint"],
                         excluded=row["excluded_draws"], near_wall=row["target_p_within_ten_pct_of_wall"],
                         correction_meV_atom=row["fep_eV"]*1000/N,
                         quadrature_minus_endpoint_meV_atom=(row["ti_eV"]-row["fep_eV"])*1000/N,
                         delta16_vs_64_intervals_meV_atom=(row["coarse_ti_eV"]-row["ti_eV"])*1000/N,
                         step_change_meV_atom=row["hessian_step_change_meV_atom"])
    table = {}
    for method, energies in f.items():
        table[method] = dict(gaps_meV_atom={k: (v-energies["d019"])*1000/N for k,v in energies.items()},
                             probability_D019=chemistry(energies, g)[1])
    # One independent block removed from every orbit for each jackknife replicate.
    drop_energies = [{k: values[i] for k, values in drops.items()} for i in range(case["replicas"])]
    errors = {k: 1000/N*jackknife_se([d[k]-d["d019"] for d in drop_energies]) for k in data}
    p_error = jackknife_se([chemistry(d, g)[1] for d in drop_energies])
    summary = dict(temperature_K=case["temperature_K"], degeneracies=g,
                   scope="Initial two-Gaussian proposal. The anharmonic W<1 estimates only describe the sampled neighborhood.",
                   validation="An independent broad proposal found collapsed Ti-Ti configurations. Use cage_sensitivity.json for defined HCP basins and the report for the standalone reproduction.",
                   gaps_and_probabilities=table, gaps_anharmonic_SE_meV_atom=errors,
                   probability_anharmonic_SE=p_error, importance_sampling=checks,
                   canonical_anharmonic_minus_classical_HA_meV_atom=1000/N*(
                   chemistry(f["direct_N8_classical_anharmonic"],g)[0] -
                   chemistry(f["direct_N8_classical_HA"],g)[0]))
    (HERE/"summary_n8.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")


def check_sampling():
    import numpy as np
    from target_check import comparison
    target = read(RUNS/"alloy-ti-mcmc-01"/"outputs"/"target_check.json")
    output = {}
    with tarfile.open(RUNS/"alloy-ti-n8-01"/"outputs"/"draws.tar.gz", "r:gz") as samples:
        for k, row in target["orbits"].items():
            sample = samples.extractfile(f"samples_{k}.npz").read()
            check = comparison(row["target_integrand_eV_cell"], BytesIO(sample), KT)
            with np.load(BytesIO(sample)) as data:
                delta = data["delta_eV_cell"]
                weights = np.exp(data["log_harmonic_to_proposal"])
            # The proposal alternates equal counts from q0 and q1. The stored
            # balance weight w = 2*q0/(q0+q1) gives q0/q1 = w/(2-w).
            inside = weights[:,::2] > 0
            narrow = (inside*np.exp(-delta[:,::2]/KT)).mean(axis=1)
            wide_w = weights[:,1::2]
            wide = (wide_w/(2-wide_w)*np.exp(-delta[:,1::2]/KT)).mean(axis=1)
            def diff(a,b):
                return -1000*KT/N*np.log(np.mean(a)/np.mean(b))
            drop = [diff(np.delete(wide,i),np.delete(narrow,i)) for i in range(len(wide))]
            check.update(wide_minus_narrow_free_energy_meV_atom=float(diff(wide,narrow)),
                         wide_minus_narrow_jackknife_se_meV_atom=jackknife_se(drop))
            output[k] = check
    (HERE/"target_comparison.json").write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-target", action="store_true", help="also compare direct MCMC; needs NumPy")
    args = parser.parse_args()
    main()
    if args.check_target:
        check_sampling()
