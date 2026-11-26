"""Summaries and figures for independently evaluated melting trajectories."""
from analyze_melting import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 180})
LABELS = {"pbe": "PBE", "scan": r"r$^2$SCAN", "sol": "PBEsol"}


def linear_slope(t, y, lo, hi):
    ok = (t >= lo) & (t <= hi)
    return float(np.polyfit(t[ok], y[ok], 1)[0])


def slab_width(profile, threshold=.25):
    """Fractional length of largest solid interval of the periodic z profile.

    Linear interpolation of crossings after Gaussian sigma=1-bin smoothing.
    Invalid/no unique slab states return NaN; no guessed interface.
    """
    p = gaussian_filter1d(profile, 1., mode="wrap")
    # Fine periodic grid for a robust fractional width (not an atom position fit).
    z = (np.arange(len(p)) + .5) / len(p)
    xx = (np.arange(3600) + .5) / 3600
    yy = np.interp(xx, np.r_[z[-1]-1, z, z[0]+1], np.r_[p[-1], p, p[0]])
    solid = yy > threshold
    if solid.all() or not solid.any():
        return np.nan, np.nan, np.nan
    starts = np.where(solid & ~np.roll(solid, 1))[0]
    ends = np.where(~solid & np.roll(solid, 1))[0]
    lengths = np.array([min((e-s) % 3600 for e in ends) for s in starts])
    largest = int(np.argmax(lengths))
    a, width = starts[largest]/3600, lengths[largest]/3600
    # Reject additional macroscopic intervals: there is no longer one planar slab.
    if (lengths > 360).sum() > 1:
        return np.nan, np.nan, np.nan
    return float(width), float(a), float((a+width) % 1)


def main():
    paths = sorted(p for p in HERE.glob("*.npz") if ".pilot." not in p.name)
    assert len(paths) == 25, f"Only {len(paths)} complete trajectories"
    cases = {}
    summaries = []
    for path in paths:
        with np.load(path) as f:
            d = {k: f[k] for k in f.files}
        key = path.stem
        cases[key] = d
        t = d["step"]*d["dt_ps"]
        d["time"] = t
        d["width"] = np.array([slab_width(p) for p in d["profile"]])
        f = d["solid_fraction"][:, 1]
        case = dict(key=key, model=str(d["model"]), temperature=int(d["temperature"]),
                    run=str(d["run"]), frames=len(t), natoms=d["q6"].shape[1],
                    duration_ps=float(t[-1]), first_fraction=float(f[0]),
                    last_fraction=float(f[-1]), mean_first5=float(f[t<=5].mean()),
                    mean_last10=float(f[t>=40].mean()), mean_qbar6_first=float(d["qbar6"][0].mean()),
                    mean_qbar6_last=float(d["qbar6"][-1].mean()),
                    lengths_initial_A=d["lengths"][0].tolist(),
                    lengths_final_A=d["lengths"][-1].tolist(),
                    source=str(d["source"]), sha256=str(d["source_sha256"]))
        case["slopes_fraction_per_ps"] = {f"{a}_{b}": linear_slope(t,f,a,b)
                                         for a,b in [(5,25),(5,50),(10,50),(25,50)]}
        case["threshold_check"] = {}
        for i, th in enumerate(d["thresholds"]):
            y = d["solid_fraction"][:, i]
            case["threshold_check"][str(th)] = dict(first=float(y[0]), last=float(y[-1]),
                slope_10_50=linear_slope(t,y,10,50))
        case["coherent_bonds_7"] = dict(first=float((d["bonds"][0]>=7).mean()),
                                      last=float((d["bonds"][-1]>=7).mean()),
                                      slope_10_50=linear_slope(t,(d["bonds"]>=7).mean(axis=1),10,50))
        case["first_all_liquid_ps"] = next((float(t[i]) for i in range(len(t))
             if np.all(f[i:] < .02)), None)
        case["first_mostly_solid_ps"] = next((float(t[i]) for i in range(len(t))
             if np.all(f[i:] > .90)), None)
        summaries.append(case)
        print(f"{key:18s} f0={f[0]:.3f} f50={f[-1]:.3f} "
              f"slope(5-25)={case['slopes_fraction_per_ps']['5_25']:+.4f} "
              f"slope(25-50)={case['slopes_fraction_per_ps']['25_50']:+.4f}", flush=True)
    (HERE/"trajectory_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    fig, axs = plt.subplots(1,3,figsize=(14,4.5),sharey=True)
    for ax, model in zip(axs, ["pbe", "sol", "scan"]):
        subset = [d for d in cases.values() if str(d["model"]) == model]
        tmin,tmax = min(int(d["temperature"]) for d in subset),max(int(d["temperature"]) for d in subset)
        for d in subset:
            temp=int(d["temperature"])
            color=plt.colormaps["coolwarm"](Normalize(tmin,tmax)(temp))
            ax.plot(d["time"],d["solid_fraction"][:,1],color=color,lw=1.5,
                    ls="--" if str(d["run"])=="002" else "-",
                    label=f"{temp} K"+(" (run 2)" if str(d["run"])=="002" else ""))
        ax.set(title=LABELS[model],xlabel="Time (ps)",xlim=(0,50),ylim=(-.02,1.04))
        ax.legend(fontsize=8,loc="center left",bbox_to_anchor=(1.01,.5),frameon=False)
        ax.grid(alpha=.2)
    axs[0].set_ylabel(r"Solid-like atom fraction ($\bar q_6>0.25$)")
    fig.suptitle("Ti coexistence trajectories: 1296 atoms, NPT, 50 ps",y=1.01)
    fig.tight_layout()
    fig.savefig(HERE/"solid_fraction_all.png",bbox_inches="tight")
    fig.savefig(HERE/"solid_fraction_all.pdf",bbox_inches="tight")
    plt.close(fig)

    chosen=["pbe_1750_001","pbe_1775_001","sol_1730_001",
            "sol_1750_001","scan_1850_001","scan_1860_001"]
    fig,axs=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True)
    for ax,key in zip(axs.T.flatten(),chosen):
        d=cases[key]
        p=gaussian_filter1d(d["profile"],1,axis=1,mode="wrap")
        im=ax.imshow(p.T,extent=[-.25,50.25,0,1],origin="lower",aspect="auto",
                     cmap="viridis",vmin=.05,vmax=.42)
        ax.contour(d["time"],(np.arange(36)+.5)/36,p.T,
                   levels=[.25],colors=["white"],linewidths=.7)
        ax.set_title(f"{LABELS[str(d['model'])]}, {int(d['temperature'])} K")
        ax.set_xlim(0,50)
    for ax in axs[-1]: ax.set_xlabel("Time (ps)")
    for ax in axs[:,0]: ax.set_ylabel("Fractional position z / Lz")
    fig.subplots_adjust(right=.90,wspace=.15,hspace=.25)
    cax=fig.add_axes([.92,.15,.015,.7])
    fig.colorbar(im,cax=cax,label=r"Local mean $\bar q_6$")
    fig.suptitle("Growth and retreat of the crystalline slab (white contour: 0.25)")
    fig.savefig(HERE/"interface_maps.png",bbox_inches="tight")
    fig.savefig(HERE/"interface_maps.pdf",bbox_inches="tight")
    plt.close(fig)

    fig,axs=plt.subplots(2,3,figsize=(13,7),sharex=True)
    for ax,key in zip(axs.T.flatten(),chosen):
        d=cases[key]
        t=d["time"]
        ax.fill_between(t,d["solid_fraction"][:,2],d["solid_fraction"][:,0],
                        color="steelblue",alpha=.18,label=r"$\bar q_6$ cutoff 0.20–0.30")
        ax.plot(t,d["solid_fraction"][:,1],label=r"$\bar q_6>0.25$",color="steelblue")
        ax.plot(t,(d["bonds"]>=7).mean(axis=1),ls="--",color="darkorange",
                label="At least 7 coherent bonds")
        ax.plot(t,d["width"][:,0],color="black",lw=1,label="Slab width / Lz")
        ax.set(title=f"{LABELS[str(d['model'])]}, {int(d['temperature'])} K",
               xlim=(0,50),ylim=(-.02,1.04))
        ax.grid(alpha=.2)
    axs[0,0].legend(fontsize=7)
    for ax in axs[-1]: ax.set_xlabel("Time (ps)")
    for ax in axs[:,0]: ax.set_ylabel("Fraction / normalized width")
    fig.suptitle("Classifier sensitivity (shading is not a statistical confidence interval)")
    fig.tight_layout()
    fig.savefig(HERE/"classifier_sensitivity.png",bbox_inches="tight")
    plt.close(fig)

    fig,axs=plt.subplots(1,3,figsize=(12,3.5),sharey=True)
    for ax, model, cold, hot in zip(axs,["pbe","sol","scan"],[1700,1730,1850],[1800,1800,1900]):
        for temp, color, label in [(cold,"steelblue","Crystalline endpoint"),
                                  (hot,"darkorange","Liquid endpoint")]:
            d=cases[f"{model}_{temp}_001"]
            ax.hist(d["qbar6"][-10:].ravel(),bins=np.linspace(0,.55,70),density=True,
                    histtype="step",color=color,label=f"{label}, {temp} K")
        ax.axvline(.25,color="black",lw=1)
        ax.axvspan(.20,.30,color="gray",alpha=.15)
        ax.set(title=LABELS[model],xlabel=r"$\bar q_6$",xlim=(0,.55))
        ax.legend(fontsize=7)
    axs[0].set_ylabel("Probability density")
    fig.tight_layout()
    fig.savefig(HERE/"order_parameter_calibration.png",bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    main()
