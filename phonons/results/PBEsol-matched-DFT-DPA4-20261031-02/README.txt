Exact matched DFT-PBEsol / DPA4-PBEsol phonon protocol.
Reproduce with Python3.12 and requirements.lock.txt: python reproduce.py --inputs inputs --output ../reproduced
Three DPA4 predictions were made under Slurm. No relaxation; exact DFT POSCARs. Checkpoint identity, inputs, forces and outputs SHA verified.
DFT convergence in cutoff/electronic kmesh/smearing/supercell size remains pending. This comparison does not establish those convergences.
Sorted bands are compared without eigenvector branch tracking.
