import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

mesh_file = "mesh.yaml"
out_dat = "total_dos.dat"
out_pdf = "total_dos_IMM_from_mesh.pdf"
out_png = "total_dos_IMM_from_mesh.png"

fmin = -1.0
fmax = 10.0
df = 0.01
sigma = 0.05  # THz

with open(mesh_file, "r") as f:
    data = yaml.safe_load(f)

freqs = []
weights = []

for q in data["phonon"]:
    w = float(q.get("weight", 1.0))
    for band in q["band"]:
        freqs.append(float(band["frequency"]))
        weights.append(w)

freqs = np.array(freqs)
weights = np.array(weights)

# total q-point weight, not mode weight
q_weight_total = sum(float(q.get("weight", 1.0)) for q in data["phonon"])

grid = np.arange(fmin, fmax + 0.5 * df, df)
dos = np.zeros_like(grid)

norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi))

for f, w in zip(freqs, weights):
    dos += (w / q_weight_total) * norm * np.exp(-0.5 * ((grid - f) / sigma) ** 2)

area = np.trapz(dos, grid)
peak_freq = grid[np.argmax(dos)]

nonzero = np.where(dos > np.max(dos) * 1e-4)[0]
fmax_nonzero = grid[nonzero[-1]] if len(nonzero) else grid.max()

np.savetxt(
    out_dat,
    np.column_stack([grid, dos]),
    header=(
        "Frequency(THz) Total_DOS\n"
        f"Generated from {mesh_file}; sigma={sigma} THz; integral={area:.8f}"
    )
)

dos_norm = dos / area if area > 0 else dos

plt.figure(figsize=(7, 5))
plt.plot(grid, dos_norm, linewidth=2, label=f"DFT Ti IMM, max ≈ {fmax_nonzero:.1f} THz")
plt.axvline(0, linestyle="--", linewidth=1, alpha=0.5)
plt.xlabel("Frequency, THz")
plt.ylabel("Normalized phonon DOS, area = 1")
plt.xlim(fmin, 9.5)
plt.grid(True, alpha=0.3)
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(out_pdf, bbox_inches="tight")
plt.savefig(out_png, dpi=300, bbox_inches="tight")

print("Read:", mesh_file)
print("Number of frequencies:", len(freqs))
print("Frequency min:", freqs.min())
print("Frequency max:", freqs.max())
print("DOS integral:", area)
print("DOS peak frequency:", peak_freq)
print("DOS max nonzero:", fmax_nonzero)
print("Wrote:", out_dat)
print("Wrote:", out_pdf)
print("Wrote:", out_png)
