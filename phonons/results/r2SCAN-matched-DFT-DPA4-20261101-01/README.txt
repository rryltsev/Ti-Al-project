Matched DFT / DPA4 central-pair phonon protocol. DFT reference force not yet calculated; cancels exactly from FC.
Reproduce with Python3.12 and requirements.lock.txt: python reproduce.py --xc PBE_or_r2SCAN --inputs inputs --output ../reproduced
Three DPA4 predictions were made under Slurm. No relaxation; exact DFT POSCARs. Checkpoint identity, inputs, forces and outputs SHA verified.
DFT convergence in cutoff/electronic kmesh/smearing/supercell size remains pending. This comparison does not establish those convergences.
Sorted bands are compared without eigenvector branch tracking.

DFT force000 has NOT been calculated: its POSCAR is included for the geometric identity check only. No zero-force placeholder is used. For the central pair, [(Fplus-F0)-(Fminus-F0)]/2 = (Fplus-Fminus)/2. This does not validate reference residual forces or stress.
