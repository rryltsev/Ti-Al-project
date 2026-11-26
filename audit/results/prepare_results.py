"""Rebuild manuscript figures and numerical summaries from archived results.

No MD/DFT calculation or fitting of an interatomic potential is performed.
Composition is x_Al throughout. Uncertainties retain their source meaning.
"""
from pathlib import Path
import csv
import hashlib
import json
import os
import sys

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
ROOT = Path(r"C:\Users\Roman Ryltsev\Desktop\STARTUPS\Ti-Al-X\TiAl-Project")
sys.path.insert(0, str(WORK / "audit/oct20/deps"))
os.environ["MPLBACKEND"] = "Agg"
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

FIG = WORK / "manuscript/figures"
FIG.mkdir(exist_ok=True)
SOURCES = {}
SUMMARY = {}
MODELS = ["pbe", "sol", "scan"]
STYLE = {"pbe": ("#2878B5", "PBE"), "sol": ("#39835B", "PBEsol"),
         "scan": ("#D55E00", r"r$^2$SCAN")}
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.dpi": 220,
})


def track(path):
    path = Path(path)
    SOURCES[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def data(path):
    return json.loads(track(path).read_text(encoding="utf-8"))


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(HERE / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


thermo = data(WORK / "audit/oct20/thermo_reprocessed.json")
refs = data(WORK / "audit/oct20/reference_values.json")
visc = data(WORK / "audit/oct25/viscosity/viscosity_merged.json")
visc_update = data(WORK / "audit/oct25/viscosity/viscosity_update_report.json")
for viscosity_source in visc_update["sources"]:
    track(viscosity_source["path"])
SUMMARY["viscosity_current"] = visc
SUMMARY["viscosity_update_pending"] = visc_update["pending_updates"]
SUMMARY["comparison_metrics"] = data(WORK / "audit/oct20/comparison_metrics.json")

# Training logs: final printed diagnostics, not a new common test-set evaluation.
SUMMARY["training_log_final"] = {}
for model in MODELS:
    a = np.loadtxt(track(ROOT / f"DPA4-finetune/lcurve_{model}.out"))
    SUMMARY["training_log_final"][model] = dict(zip(
        ["step", "mae_e_val", "mae_e_trn", "mae_f_val", "mae_f_trn",
         "mae_v_val", "mae_v_trn", "rmse_val", "rmse_trn", "lr"],
        a[-1].tolist()))

# Phonons: use raw production arrays, not old plotting CSV model columns.
phroot = ROOT / "results-scripts-logs/models_check_wsl2/results/phonons/dpa4/neo/finetune"
expfile = ROOT / "MD-results/phonons_DPA4-neo/finetune/alpha_Ti/pbe/phonon_bands_data_alpha_Ti_dpa4-neo-finetune_pbe.csv"
with track(expfile).open() as handle:
    rr = list(csv.DictReader(handle))
ex = np.array([(float(r["Exp_x"]), float(r["Exp_y"])) for r in rr
               if r["Exp_x"] and r["Exp_y"]])
fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.5), layout="constrained")
SUMMARY["phonons"] = {}
for j, model in enumerate(MODELS):
    col, label = STYLE[model]
    for system in ["alpha_Ti", "alpha2_AlTi3"]:
        p = phroot / system / model
        command = track(p / "command.txt").read_text()
        for flag in ["--delta 0.01", "--kpts-dos 50,50,50", "--dos-width 0.0002"]:
            assert flag in command, (p, flag)
        a = np.loadtxt(track(p / "phonon_bands.dat"))
        at = read(track(p / "relaxed_structure.cif"))
        path = "GKMGA" if system == "alpha_Ti" else "GMKGALHA"
        q, ticks, labels = at.cell.bandpath(path, npoints=200).get_linear_kpoint_axis()
        assert np.allclose(q, a[:, 0], atol=1e-5), p
        summary = {"min_frequency_THz": float(a[:, 1:].min()),
                   "max_frequency_THz": float(a[:, 1:].max()),
                   "gamma_frequencies_THz": a[0, 1:].tolist(),
                   "path": path, "q_special_normalized": (ticks / q[-1]).tolist(),
                   "cell_A": at.cell.cellpar().tolist()}
        SUMMARY["phonons"][system + "/" + model] = summary
        if system != "alpha_Ti":
            continue
        ax = axs.flat[j]
        ax.plot(q / q[-1], a[:, 1:], color=col, lw=1.0)
        ax.scatter(ex[:, 0], ex[:, 1], s=10, color="black", alpha=.8,
                   label="Experiment (digitized)", zorder=3)
        for t in ticks:
            ax.axvline(t / q[-1], color=".8", lw=.5, zorder=0)
        ax.set(xticks=ticks / q[-1],
               xticklabels=[r"$\Gamma$" if x == "G" else x for x in labels],
               ylabel="Frequency (THz)", xlim=(0, 1), ylim=(-.2, 10),
               title=f"({chr(97+j)}) " + label + r": $\alpha$-Ti")
        if j == 0:
            ax.legend(loc="upper right", frameon=False, fontsize=7)
        dos = np.loadtxt(track(p / "phonon_dos.dat"))
        # Stored ASE DOS frequency was converted to THz without a Jacobian.
        # Explicit area normalization supplies modes/(atom THz).
        integral = np.trapezoid(dos[:, 1], dos[:, 0])
        assert integral > 0
        y = dos[:, 1] * 3 / integral
        axs[1, 1].plot(dos[:, 0], y, color=col, lw=1.1, label=label)
        summary["raw_DOS_integral"] = float(integral)
        summary["normalized_DOS_integral"] = float(np.trapezoid(y, dos[:, 0]))
dftdos = np.loadtxt(track(ROOT / "TiAl-papers/phonons/DFT_Ti_Ti3Al/Ti_hcp/total_dos.dat"))
dfty = dftdos[:, 1] * 3 / np.trapezoid(dftdos[:, 1], dftdos[:, 0])
axs[1, 1].plot(dftdos[:, 0], dfty, "k--", lw=1.1, label="Independent PBE DFT")
axs[1, 1].set(xlabel="Frequency (THz)", ylabel=r"DOS (atom$^{-1}$ THz$^{-1}$)",
               title=r"(d) $\alpha$-Ti density of states", xlim=(0, 11))
axs[1, 1].legend(frameon=False, fontsize=7)
save(fig, "phonons_alpha_ti")

# Density vs temperature: labels identify preparation, not verified final phases.
# Experimental basis and validity ranges are retained in references.json.
density_refs = data(WORK / "audit/density_reference/references.json")
track(density_refs["solid"]["source_file"])
liquid_ref = density_refs["liquid"]
liquid_measurements = data(WORK / "audit/density_reference/reiplinger2022_points.json")
track(liquid_measurements["source_file"])


def liquid_ti_reference(temp):
    temp = np.asarray(temp)
    assert np.all((temp >= liquid_ref["valid_range_K"][0]) &
                  (temp <= liquid_ref["valid_range_K"][1]))
    return (liquid_ref["c1_kg_m3"] +
            liquid_ref["c2_kg_m3_K"] * (temp - liquid_ref["Tm_K"])) / 1000


fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.1), layout="constrained")
SUMMARY["density_ti"] = {}
SUMMARY["density_ti_reference_comparison"] = {}
for model in MODELS:
    col, label = STYLE[model]
    with track(ROOT / f"MD-results/density_DPA4-neo/density_scan_{model}.csv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    SUMMARY["density_ti"][model] = rows
    for lo, hi, marker, ls in [(300, 1100, "o", "-"), (1200, 1900, "^", "--"),
                                (2000, 2500, "s", "-")]:
        g = [r for r in rows if lo <= float(r["T_target_K"]) <= hi]
        ax = axs[1] if lo == 2000 else axs[0]
        ax.plot([float(r["T_target_K"]) for r in g],
                [float(r["density_g_cm3"]) for r in g], color=col, marker=marker,
                ls=ls, ms=3, lw=1, label=label if lo == 300 else None)
    liquid_rows = [r for r in rows if 2000 <= float(r["T_target_K"]) <= 2500]
    temperatures = np.array([float(r["T_target_K"]) for r in liquid_rows])
    densities = np.array([float(r["density_g_cm3"]) for r in liquid_rows])
    reference = liquid_ti_reference(temperatures)
    bias = 100 * (densities / reference - 1)
    rho300 = float(next(r for r in rows if float(r["T_target_K"]) == 300)["density_g_cm3"])
    SUMMARY["density_ti_reference_comparison"][model] = {
        "solid_300K_bias_percent": 100 * (rho300 / 4.50 - 1),
        "liquid_T_K": temperatures.tolist(),
        "liquid_reference_g_cm3": reference.tolist(),
        "liquid_bias_percent": bias.tolist(),
        "liquid_mean_absolute_relative_deviation_percent": float(np.abs(bias).mean()),
        "liquid_linear_slope_g_cm3_K": float(np.polyfit(temperatures, densities, 1)[0]),
    }
for phase in ["alpha", "beta"]:
    points = np.array(density_refs["solid"][phase])
    axs[0].plot(points[:, 0], points[:, 1], color="black", marker="D", ms=3,
                ls=":", lw=.9, label="Zinov'ev (1989)" if phase == "alpha" else None)
tt = np.linspace(*liquid_ref["plotted_range_K"], 200)
rr = liquid_ti_reference(tt)
u = liquid_ref["expanded_relative_uncertainty"]
axs[1].fill_between(tt, rr * (1-u), rr * (1+u), color=".85", zorder=0,
                    label=r"$\pm2\%$ (95% coverage)")
axs[1].plot(tt, rr, color="black", lw=1.2, label="Ntonti et al. (2024)")
pp = liquid_measurements["points"]
axs[1].errorbar([p["T_K"] for p in pp], [p["rho_g_cm3"] for p in pp],
                yerr=[[p["rho_g_cm3"]-p["lower_g_cm3"] for p in pp],
                      [p["upper_g_cm3"]-p["rho_g_cm3"] for p in pp]],
                fmt="o", ms=2.6, mfc="white", mec="black", mew=.6,
                ecolor=".3", elinewidth=.55, capsize=1.3, zorder=4,
                label="Reiplinger & Brillo (2022)")
SUMMARY["density_ti_experimental_points"] = liquid_measurements
for temp, label in [(1155, r"$\alpha$--$\beta$: 1155 K"), (1943, r"$T_m$: 1943 K")]:
    axs[0].axvline(temp, color=".45", ls="-.", lw=.8, zorder=0)
    axs[0].text(temp-25, 4.82, label, rotation=90, ha="right", va="top", fontsize=7, color=".3")
axs[0].set(title=r"(a) $\alpha$- and $\beta$-Ti starting cells", xlim=(220, 2010), ylim=(4.22, 4.85))
axs[1].set(title="(b) Separately prepared liquid", xlim=(1835, 2530), ylim=(3.87, 4.66))
axs[0].legend(frameon=True, facecolor="white", edgecolor="none", framealpha=1,
              ncol=2, fontsize=7, loc="lower left", borderpad=.3)
handles, labels = axs[1].get_legend_handles_labels()
axs[1].legend(handles[::-1], labels[::-1], frameon=False, fontsize=7, loc="upper right")
for ax in axs:
    ax.set(xlabel="Temperature (K)", ylabel=r"Density (g cm$^{-3}$)")
save(fig, "density_ti")

# Melting: show nearby trajectories and retain both preparations at 1850 K.
fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.65), sharey=True, layout="constrained")
chosen = {"pbe": [(1750, "001"), (1775, "001"), (1800, "001")],
          "sol": [(1730, "001"), (1750, "001"), (1770, "001")],
          "scan": [(1850, "001"), (1850, "002"), (1860, "001")]}
SUMMARY["melting_source_summary"] = data(WORK / "audit/melting/trajectory_summary.json")
for j, model in enumerate(MODELS):
    for i, (temp, run) in enumerate(chosen[model]):
        f = np.load(track(WORK / f"audit/melting/{model}_{temp}_{run}.npz"))
        axs[j].plot(f["step"] * f["dt_ps"], f["solid_fraction"][:, 1],
                    color=["#2878B5", "#9467BD", "#D55E00"][i],
                    ls="--" if run == "002" else "-", lw=1.1,
                    label=f"{temp} K" + (" (2)" if run == "002" else ""))
    axs[j].set(xlabel="Time (ps)", title=f"({chr(97+j)}) " + STYLE[model][1],
               xlim=(0, 50), ylim=(-.03, 1.04))
    axs[j].legend(fontsize=7, frameon=False, loc="center right")
axs[0].set_ylabel("Solid-like fraction")
save(fig, "melting_ti")

# Alloy density and excess molar volume.
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8), layout="constrained")
for model in MODELS:
    col, label = STYLE[model]
    g = sorted((r for r in thermo if r["T"] == 1873 and r["dft"] == model),
               key=lambda r: r["x_Al"])
    axs[0].plot([r["x_Al"] for r in g], [r["rho"] for r in g], "o-", color=col, ms=3, lw=1, label=label)
    axs[1].plot([r["x_Al"] for r in g], [r["Vex_cm3mol"] for r in g], "o-", color=col, ms=3, lw=1, label=label)
xx, yy, vv = [], [], []
for x, values in refs["density_table7"].items():
    x = float(x)
    for rho in values:
        xx.append(x); yy.append(rho)
        vv.append(((1-x)*47.867+x*26.9815385)/rho - (1-x)*47.867/4.184 - x*26.9815385/2.08)
axs[0].scatter(xx, yy, color="black", marker="s", s=14, zorder=5, label="Published densities")
axs[1].scatter(xx, vv, color="black", marker="s", s=14, zorder=5, label="From published densities")
axs[1].axhline(0, color=".6", lw=.7)
for ax in axs:
    ax.set(xlabel=r"$x_{\mathrm{Al}}$", xlim=(-.02, 1.02))
    ax.legend(frameon=False, fontsize=7)
axs[0].set(title="(a) Density, 1873 K", ylabel=r"$\rho$ (g cm$^{-3}$)")
axs[1].set(title="(b) Excess volume, 1873 K", ylabel=r"$V_m^{\mathrm{ex}}$ (cm$^3$ mol$^{-1}$)")
save(fig, "liquid_density")

# Mixing enthalpy: panel at reference T, panel showing all three isotherms.
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8), layout="constrained")
SUMMARY["thermo_extrema"] = {}
for model in MODELS:
    col, label = STYLE[model]
    for T, ls in [(1873, ":"), (2000, "-"), (2500, "--")]:
        g = sorted((r for r in thermo if r["T"] == T and r["dft"] == model),
                   key=lambda r: r["x_Al"])
        mn = min(g, key=lambda r: r["dH_external_pressure"])
        vm = min(g, key=lambda r: r["Vex_cm3mol"])
        SUMMARY["thermo_extrema"][f"{model}/{T}"] = {
            "Hmix_min": mn["dH_external_pressure"], "xAl_Hmix_min": mn["x_Al"],
            "Vex_min": vm["Vex_cm3mol"], "xAl_Vex_min": vm["x_Al"]}
        axs[1].plot([r["x_Al"] for r in g], [r["dH_external_pressure"] for r in g],
                    ls=ls, c=col, lw=1.1, label=label if T == 2000 else None)
        if T == 2000:
            axs[0].errorbar([r["x_Al"] for r in g], [r["dH_external_pressure"] for r in g],
                           yerr=[r["dH_external_sem_5ps"] for r in g], fmt="o-",
                           color=col, ms=3, lw=1, capsize=2, label=label)
mix = refs["mixing_digitized_kJmol"]
axs[0].scatter([float(x) for x in mix], list(mix.values()), c="black", marker="s",
               s=14, label="Calorimetry (reproduced)", zorder=5)
for ax in axs:
    ax.set(xlabel=r"$x_{\mathrm{Al}}$", ylabel=r"$\Delta H_{\mathrm{mix}}$ (kJ mol$^{-1}$)",
           xlim=(-.02, 1.02), ylim=(-36, 2))
axs[0].set_title("(a) 2000 K")
axs[1].set_title("(b) Temperature dependence")
axs[0].legend(fontsize=7, frameon=False, loc="upper center")
for T, ls in [(1873, ":"), (2000, "-"), (2500, "--")]:
    axs[1].plot([], [], c=".3", ls=ls, label=f"{T} K")
axs[1].legend(fontsize=7, frameon=False, ncol=2, loc="upper center")
save(fig, "mixing_enthalpy")

# Viscosity: use x_Al = 1-c_Ti consistently with all thermodynamic figures.
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9), layout="constrained")
for model in ["pbe", "scan"]:
    col, label = STYLE[model]
    g = sorted((r for r in visc if r["T"] == 1873 and r["dft"] == model),
               key=lambda r: 1-r["c_Ti"])
    axs[0].errorbar([1-r["c_Ti"] for r in g], [r["eta_mPa_s"] for r in g],
                    yerr=[r["eta_std_mPa_s"] for r in g],
                    fmt="o-", c=col, ms=3, capsize=2, lw=1, label=label)
    g = sorted((r for r in visc if r["c_Ti"] == 1 and r["dft"] == model), key=lambda r: r["T"])
    axs[1].errorbar([r["T"] for r in g], [r["eta_mPa_s"] for r in g],
                    yerr=[r["eta_std_mPa_s"] for r in g],
                    fmt="o-", c=col, ms=3, capsize=2, lw=1, label=label)
for c, (a, b, lo, hi) in refs["takeda_table3"].items():
    axs[0].scatter(1-float(c), 10**(a+b/1873), s=23, marker="s",
                    facecolors="black" if hi >= 1873 else "white",
                    edgecolors="black", zorder=5)
axs[0].plot([], [], "ks", ms=4, label="Takeda fit")
axs[0].plot([], [], "ks", mfc="white", ms=4, label="Fit extrapolated by 3 K")
t = np.linspace(1850, 2120, 150)
eta = .61*np.exp(27000/(8.31*t))
axs[1].plot(t, eta, "k--", lw=1, label="Ishikawa fit*")
axs[1].fill_between(t, eta*.897, eta*1.103, color=".7", alpha=.25)
axs[0].set(title="(a) Composition dependence, 1873 K", xlabel=r"$x_{\mathrm{Al}}$",
           ylabel=r"$\eta$ (mPa s)", xlim=(-.02, 1.02), ylim=(0, 8.3))
axs[1].set(title="(b) Pure Ti", xlabel="Temperature (K)", ylabel=r"$\eta$ (mPa s)",
           xlim=(1840, 2130), ylim=(2.1, 4.25))
axs[0].legend(fontsize=7, frameon=False, loc="lower left")
axs[1].legend(fontsize=7, frameon=False, loc="upper right")
save(fig, "viscosity")

(HERE / "numerical_summary.json").write_text(json.dumps(SUMMARY, indent=2), encoding="utf-8")
(HERE / "figure_sources.json").write_text(json.dumps(SOURCES, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in SUMMARY.items() if k in ["training_log_final", "phonons", "thermo_extrema"]}, indent=2))
print("Saved six figures, numerical summary, and source hashes.")
