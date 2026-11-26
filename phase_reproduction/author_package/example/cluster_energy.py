"""Project-wide raw-cell inclusion-exclusion, occupation basis Ti=0 / Al=1.

Physical compatibility and periodic multiplicity checks belong to the caller.
No atom-count, coordination-number, orbit or factorial normalization is applied.
"""
from itertools import combinations
from math import isfinite, fsum
from numbers import Real


def all_subsets(cluster):
    """Each actual site subset once, including the pure Ti matrix ()."""
    parent = tuple(sorted(cluster))
    if len(set(parent)) != len(parent):
        raise ValueError('cluster contains repeated site indices')
    return [subset for n in range(len(parent) + 1)
            for subset in combinations(parent, n)]


def connected_energy(cluster, energies):
    """Sum (-1)^(n-|S|) E_tot(S); energies must be full compatible cell energies.

    For four sites, counts are +1 quartet, -4 triples, +6 pairs,
    -4 singles, +1 pure Ti. Missing energies are errors, never zeros.
    Decimal inputs retain Decimal arithmetic; real-valued floats use fsum.
    """
    parent = tuple(sorted(cluster))
    subsets = all_subsets(parent)
    missing = [s for s in subsets if s not in energies]
    if missing:
        raise ValueError('missing subgroup energies: ' + repr(missing))
    terms = []
    for subset in subsets:
        energy = energies[subset]
        if not isfinite(energy):
            raise ValueError('nonfinite subgroup energy for ' + repr(subset))
        terms.append((-1 if (len(parent) - len(subset)) % 2 else 1) * energy)
    if all(isinstance(e, Real) for e in terms):
        return fsum(terms)
    return sum(terms, terms[0] * 0)


def binding_energy(cluster, energies):
    """Full binding against isolated substitutions; negative means favorable."""
    parent = tuple(sorted(cluster))
    if len(set(parent)) != len(parent):
        raise ValueError('cluster contains repeated site indices')
    needed = [(), parent] + [(i,) for i in parent]
    missing = [s for s in needed if s not in energies]
    if missing:
        raise ValueError('missing binding references: ' + repr(missing))
    return energies[parent] + (len(parent) - 1) * energies[()] - sum(
        (energies[(i,)] for i in parent), energies[()] * 0)
