---
name: hcp-thermodynamic-ce
description: Build a binary HCP cluster expansion from CPU MTP static energies and quantum harmonic phonons, including volume-dependent fits and the Ti–Al D019 reproduction protocol.
---

# HCP cluster expansion from atomistic labels

Keep the model file, chemical site occupations, orbit representatives, fit split and volume grid together. A published potential identifies the atomistic energy function. Reproducing a fitted lattice Hamiltonian also requires the original occupation vectors and the fit convention. Assign fresh designs their own identifier.

1. At each volume, keep cell shape and atomic species fixed and relax internal coordinates. Save the ideal occupations as the input to the correlation calculation. Use the relaxed geometry for energies and phonons. Check the maximum force, species map and actual volume.
2. Enumerate clusters on the infinite parent lattice, then translate and wrap into the chemical cell. Distinct periodic images can map to the same site in a small chemical cell. Preserve their multiplicity when labelling structures. For Ti–Al v16, the two equilateral basal 222 triangles have different orbits. Identify the triangle whose centroid projects onto a site in the next layer.
3. With Ti spin −1 and Al spin +1, use features per atom `[1/2, sum(s_i)/N, sum(products)/N, ...]`. Each translated physical cluster contributes once. Check pure Ti, Al and periodic D019 analytically, and check mixed configurations before fitting. The enumeration cell has axial ratio 1.6. The physical Ti–Al cell uses the specified reference volume and physical axial ratio.
4. For quantum HA, subtract the undisturbed supercell forces from displaced forces, impose the acoustic sum rule and inspect the phonon mesh. Use a Γ-centred mesh, three translations at Γ, normalized q weights and energy per atom. Frequencies in THz use Planck's constant h. Imaginary internal modes invalidate the harmonic label for that structure and volume. Record the affected rows and apply the same fit mask to both static and phonon labels.
5. Fit static energy and harmonic free energy in the same basis, with equal weight per chemical structure if reproducing v16. Check rank, conditioning, leave-one-out error, held-out errors and differences between ordered and disordered structures at equal composition. Propagate the full constant and point coefficients across volume in QHA. They contribute to the minimizing volume.
6. For the v16 controls, calculate the ordered Ti3Al free-energy minimum over volume at 600 K, including endpoints and stationary points. Compare its position and lowering relative to the fixed reference volume with the supplementary text. Then compare finite-temperature boundaries using the companion `vcsgc-binodal` workflow.

The worked implementation lives in the TiAl project at `phase_reproduction/atomistic`. Its `hcp_orbits.py`, `label_cell.py` and fit driver form a reference for porting these conventions. The project report records the tested MTP hash, data lineage and actual UMT runs. For other alloys, supply the target potential, parent-lattice orbits and validation states.
