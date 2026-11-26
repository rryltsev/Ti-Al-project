---
name: atat-eci
description: Enumerate binary alloy structures with ATAT genstr, calculate orbit correlations with corrdump, select independent training cells and fit effective cluster interactions to atomistic energies.
---

# Prepare a native ATAT data set and fit its ECI

Start with the parent lattice, active species in their intended order, concentration interval, energy unit and the geometric cluster basis. Record the actual ATAT release and command lines.

1. Write `lat.in` and run `genstr -l=lat.in -n=<max_atoms>`. Save complete native `str.out` records separated by the `end` sentinel. Inspect each candidate's stoichiometry, determinant and number of sites. A finite enumeration gives symmetry-distinct orderings within its permitted cell sizes. Supply larger ordered or disordered controls explicitly when needed and label their generator.
2. Use `corrdump -l=lat.in -clus -2=<pair_diameter> -3=<triplet_diameter>` to generate native cluster geometry. A diameter includes all shorter clusters of the chosen order. If reproducing a selected basis, match sites, orbit multiplicities and parent symmetry before extracting the requested blocks into `clusters.out`.
3. Fix the cluster file before processing the pool. Use `corrdump -l=lat.in -c -cf=clusters.out -s=<str.out> -sig=14` for each ideal decoration. Preserve the raw rows. Check sign and multiplicities using pure endmembers, an ordered structure and mixed structures. In a two-site Ti–Al parent cell the S1 constant term has weight 1/2 per atom and a pair or triangle has weight equal to half its ATAT multiplicity.
4. Reserve structurally distinct controls before using energies. Deduplicate correlation vectors, enforce rank for all target columns, and inspect singular values, condition number, composition coverage and local-order coverage. Use an overdetermined training set. Retain the candidate IDs and selection seed or deterministic tie rule.
5. Convert the native structure to the physical cell for the energy calculation. Pass its assigned occupations intact, verifying composition, per-atom volume, shape and PBC. Evaluate energy per atom after the declared relaxation procedure. Keep a checksum connecting each energy to the input. At fixed strain remove atomic forces without relaxing the cell.
6. Fit on training rows only. If `E/N = ∑ b_i ρ_i` was fitted to raw corrdump correlations and native cluster multiplicities are `m_i`, write `b_i/m_i` in the actual `clusters.out` order to obtain native predictions in eV/atom. In the two-site S1 basis `X_i=ρ_i m_i/2`, the fitted coefficients are `J_i` and the corresponding native eV/atom file contains `J_i/2`. With ATAT 3.52, `-mi` instead takes multiplicity-weighted `b_i` directly. Test both endmembers and a mixed cell with `corrdump -l=lat.in -c -cf=clusters.out -s=str.out -eci=eci.out -sig=14`. Archive predictions, LOO and excluded-cell residuals.

For a native ATAT 3.52 fit, put the training features in `x.dat` with columns in the agreed orbit order, and put the per-atom energies in `y.dat` in matching row order. Run `lsfit -x=x.dat -y=y.dat`. Multiple columns of energies give one coefficient vector per target in the output. This release prints five fixed decimal places. To preserve numerical precision, scale all energies by a recorded common factor before fitting (the Ti–Al workflow uses 10⁸) and divide all returned coefficients by that factor. Check the fit against an independent linear solve and use native `corrdump` to read back the exported Hamiltonian for held-out structures.

In the Ti-rich Ti–Al v16 example the `genstr -n=10` pool gave rank 19 because its two 222 triplet correlations were identical. The `-n=12` pool separates them. Native corrdump lists the 222 class with no adjacent-layer site projected at its centroid before the class with a site there. Map by geometry when reading and writing ECI.

For Ti–Al use the tested pipeline in the TiAl project's `phase_reproduction/atomistic/atat`. The `hcp-thermodynamic-ce` and `vcsgc-binodal` skills cover the phonon labels and phase-boundary calculations that consume this fit.
