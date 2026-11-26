"""Verify the CSV and show the per-atom energy of an hcp Ti cell at 1000 K."""
import csv
import json
from pathlib import Path
import numpy as np


def main():
    root = Path(__file__).resolve().parent
    geometry = json.loads((root / 'models/geometry.json').read_text())
    with np.load(root / 'models/static/volume_temperature_coefficients.npz') as s, np.load(
        root / 'models/HA_QHA/volume_temperature_coefficients.npz'
    ) as h:
        volumes = h['volumes_A3_atom'].tolist()
        temperatures = h['temperatures_K'].tolist()
        seen = set()
        with (root / 'ECI_TiAl.csv').open(newline='') as f:
            for r in csv.DictReader(f):
                iv = volumes.index(float(r['V_A3_per_atom']))
                it = temperatures.index(float(r['T_K']))
                k = int(r['index'])
                key = (iv, it, k)
                assert key not in seen
                seen.add(key)
                cl = geometry['orbits'][k]
                assert r['cluster'] == cl['name'] and int(r['order']) == cl['order']
                assert float(r['multiplicity_per_atom']) == cl['multiplicity_per_atom']
                j0, jt = s['eci'][key], h['eci'][key]
                assert float(r['J_static_eV']) == j0 and float(r['J_effective_eV']) == jt
                assert float(r['delta_J_vib_eV']) == jt - j0
        assert len(seen) == s['eci'].size == h['eci'].size
        iv = volumes.index(geometry['reference_volume_A3_atom'])
        it = temperatures.index(1000)
        x = np.array([c['multiplicity_per_atom'] * (-1)**c['order'] for c in geometry['orbits']])
        np.testing.assert_allclose(x, s['X'][0], rtol=0, atol=1e-12)
        np.testing.assert_allclose(x, h['X'][0], rtol=0, atol=1e-12)
        with (root / 'ECI_Vref_1000K.csv').open(newline='') as f:
            short = list(csv.DictReader(f))
        assert len(short) == len(x)
        for k, r in enumerate(short):
            assert int(r['index']) == k and r['name'] == geometry['orbits'][k]['name']
            assert abs(float(r['J_static_eV']) - s['eci'][iv,it,k]) < 1e-12
            assert abs(float(r['delta_J_vibrational_1000K_eV']) - (h['eci'][iv,it,k]-s['eci'][iv,it,k])) < 1e-12
        es = x @ s['eci'][iv,it]
        fh = x @ h['eci'][iv,it]
        print(f'CSV verified: {len(seen)} rows. Pure hcp Ti at Vref, 1000 K (eV/atom):')
        print(f'  static CE {es:.3f}   target {s["E_static_eV_atom"][iv,0]:.3f}')
        print(f'  harmonic CE {fh:.3f}   target {h["Y_eV_atom"][iv,0,it]:.3f}')


if __name__ == '__main__':
    main()
