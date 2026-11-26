"""Ti-Al HCP clusters reconstructed from v16, Table S1 and Fig. S1."""
from functools import lru_cache
from itertools import combinations, product
import math

import numpy as np


ORBIT_NAMES = ("constant", "point", "p1", "p2", "p3", "p4", "p5", "p6", "p7",
               "t112", "t114", "t117", "t123", "t125", "t136", "t222_1", "t222_2",
               "t226", "t247", "t334")
MULTIPLICITIES = np.array([0.5, 1, 3, 3, 3, 1, 6, 3, 6, 6, 3, 6, 12, 12, 6, 1, 1, 6, 12, 3])
STATIC_S1 = np.array([-12.10, 2.08, .049, .036, -.00867, .023, .00247, .00839,
                      -.0048, -.000308, .00429, -.00436, .00157, .000741, .00131,
                      .00411, -.00883, .00177, -.0000181, -.000726])
# Printed a,c have fewer digits than the supplied reference volume.
VREF = 17.288329847774033
VOLUME_RATIOS = (.94, .96, .98, 1., 1.02, 1.04, 1.06, 1.09, 1.12)
AXIAL_RATIO = 4.6046 / 2.9446


def _frac(site):
    i, j, k, b = site
    return np.array([i + b/3, j + b/3, k + b/2])


def _metric(a, b):
    # Dimensionless ATAT enumeration cell: lengths (1,1,1.6), basal angle 60°.
    x, y, z = a - b
    return round(x*x + x*y + y*y + (1.6*z)**2, 8)


@lru_cache(maxsize=1)
def templates():
    """All distinct clusters per primitive translation before periodic wrapping."""
    nodes = list(product(range(-3, 4), range(-3, 4), range(-2, 3), range(2)))
    coords = {n: _frac(n) for n in nodes}
    zero = coords[0, 0, 0, 0]
    radii = sorted({_metric(p, zero) for p in coords.values()} - {0.})[:7]
    shells = {r: i+1 for i, r in enumerate(radii)}
    groups = {name: [] for name in ORBIT_NAMES}
    for b in (0, 1):
        base = (0, 0, 0, b)
        p = coords[base]
        groups["point"].append((base,))
        # Base is the first site in infinite-lattice lexicographic order. Count
        # every lattice cluster once, including distinct periodic images in tiny cells.
        neighbors = {n: shells[_metric(coords[n], p)] for n in nodes
                     if n > base and _metric(coords[n], p) in shells}
        for n, shell in neighbors.items():
            groups[f"p{shell}"].append((base, n))
        for u, v in combinations(neighbors, 2):
            shell = shells.get(_metric(coords[u], coords[v]))
            if shell is None:
                continue
            edges = sorted([neighbors[u], neighbors[v], shell])
            name = "t" + "".join(map(str, edges))
            if name == "t222":
                center = (p + coords[u] + coords[v])[:2] / 3
                adjacent = center - (1-b)/3
                name += "_1" if np.allclose(adjacent, np.rint(adjacent)) else "_2"
            if name in groups:
                groups[name].append((base, u, v))
    for name, m in zip(ORBIT_NAMES[1:], MULTIPLICITIES[1:]):
        if len(groups[name]) != int(2*m):
            raise RuntimeError(f"{name} has {len(groups[name])} clusters per cell, expected {2*m}")
    return groups


def _cell_indices(shape):
    if len(shape) != 3 or any(int(n) != n or n < 1 for n in shape):
        raise ValueError("shape must specify three positive integers")
    return np.indices(shape, dtype=np.int64).reshape(3, -1).T


def build_orbits(shape):
    """Return translated site arrays in S1 order; apply the constant separately."""
    cells = _cell_indices(shape)
    groups = templates()
    indexed = [np.empty((0, 0), dtype=np.int32)]
    for name in ORBIT_NAMES[1:]:
        motif = np.asarray(groups[name], dtype=int)
        # (primitive translations, motif index, cluster slot, xyz)
        sites = (cells[:, None, None, :] + motif[None, :, :, :3]) % shape
        ids = np.ravel_multi_index(np.moveaxis(sites, -1, 0), shape)*2 + motif[:, :, 3]
        indexed.append(ids.reshape(-1, motif.shape[1]).astype(np.int32))
    return tuple(indexed)


def features(spins, orbits):
    """Return multiplicity-weighted correlations per atom."""
    s = np.asarray(spins)
    if s.ndim != 1 or len(s) != len(orbits[1]) or not np.isin(s, [-1, 1]).all():
        raise ValueError("one spin (-1 for Ti, +1 for Al) is required per lattice site")
    return np.array([.5] + [np.prod(s[sites], axis=1).sum() / len(s) for sites in orbits[1:]])


def features_for_parent_sites(matrix, sites, spins):
    """Handle a general integer supercell of HCP (row matrix, unreconstructed parent sites).

    sites[n] = (i,j,k,b) in the same ideal primitive basis as hcp_geometry.
    Fixed-cell internal relaxation does not change this CE site map.
    """
    matrix, sites, spins = np.asarray(matrix), np.asarray(sites), np.asarray(spins)
    if matrix.shape != (3,3) or not np.array_equal(matrix, np.rint(matrix)):
        raise ValueError("expected a 3x3 integer supercell matrix")
    determinant = int(round(abs(np.linalg.det(matrix))))
    if (determinant == 0 or sites.shape != (2*determinant, 4) or
        not np.array_equal(sites, np.rint(sites)) or not np.isin(sites[:, 3], [0, 1]).all() or
        spins.shape != (len(sites),) or not np.isin(spins, [-1, 1]).all()):
        raise ValueError("invalid HCP site or spin list")
    # Two translations are equivalent when their difference is an integer
    # combination of the rows of S. This integer key needs no Cartesian tolerance.
    adj = np.rint(np.linalg.inv(matrix)*determinant).astype(np.int64)
    def key(xyz):
        return tuple((np.asarray(xyz, dtype=int) @ adj) % determinant)
    mapping = {(key(s[:3]), s[3]): spin for s, spin in zip(sites, spins)}
    origins = [s[:3].astype(int) for s in sites if s[3] == 0]
    if len(mapping) != len(sites) or len(origins) != determinant:
        raise ValueError("duplicate or missing periodic sites")
    x = [.5]
    for name in ORBIT_NAMES[1:]:
        value = 0
        for origin in origins:
            for motif in templates()[name]:
                value += math.prod(mapping[key(origin+site[:3]), site[3]] for site in motif)
        x.append(value / len(sites))
    return np.array(x)


def sublattices(shape):
    """Four equal D019 site colors; periodic D019 requires even basal repeat counts."""
    if shape[0] % 2 or shape[1] % 2:
        raise ValueError("D019 coloring requires even nx, ny")
    i, j, _ = _cell_indices(shape).T
    basis = np.arange(2)
    return (2*((i[:, None]+basis) % 2) + (j[:, None]+basis) % 2).ravel()


def hcp_geometry(shape, volume_ratio=1.0):
    """Return row lattice vectors and Cartesian sites in the orbit enumeration order."""
    if volume_ratio <= 0:
        raise ValueError("volume_ratio must be positive")
    a = (4*VREF*volume_ratio / (math.sqrt(3)*AXIAL_RATIO))**(1/3)
    primitive = a*np.array([[1, 0, 0], [.5, math.sqrt(3)/2, 0], [0, 0, AXIAL_RATIO]])
    coords = _cell_indices(shape)[:, None, :] + np.array([[0, 0, 0], [1/3, 1/3, .5]])
    return np.diag(shape) @ primitive, coords.reshape(-1, 3) @ primitive
