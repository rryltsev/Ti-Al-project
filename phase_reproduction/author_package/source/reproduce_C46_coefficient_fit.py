"""Replay the C46 least-squares fits from bundled full-precision observations.

Usage: python reproduce_C46_coefficient_fit.py MODEL_PACKAGE_DIRECTORY
Requires NumPy and SciPy. Does not run MTP, change coefficients, or test their
accuracy against independent electronic-structure calculations.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
from scipy.interpolate import CubicSpline


def reproduce(package):
    paths = {
        label: package / 'models' / label / 'volume_temperature_coefficients.npz'
        for label in ['static', 'HA_QHA']
    }
    with np.load(paths['static']) as data:
        static = {name: data[name] for name in data.files}
    with np.load(paths['HA_QHA']) as data:
        harmonic = {name: data[name] for name in data.files}
    for name in ['X', 'training', 'stable', 'volumes_A3_atom', 'temperatures_K']:
        assert np.array_equal(static[name], harmonic[name]), name
    X, training, stable = (static[k] for k in ['X', 'training', 'stable'])
    assert X.shape == (46, 20)
    assert int(training.sum()) == 39
    assert stable.shape == (9, 46)
    assert np.isfinite(static['E_static_eV_atom']).all()
    # Excluded unstable configuration/volume points may store NaN by design.
    assert np.isfinite(harmonic['Y_eV_atom'][stable]).all()
    rows = []
    for iv, volume in enumerate(static['volumes_A3_atom']):
        selected = training & stable[iv]
        A = X[selected]
        rank = int(np.linalg.matrix_rank(A))
        assert rank == 20, (iv, rank)
        static_fit = np.linalg.lstsq(
            A, static['E_static_eV_atom'][iv, selected], rcond=None
        )[0]
        harmonic_fit = np.linalg.lstsq(
            A, harmonic['Y_eV_atom'][iv, selected, :], rcond=None
        )[0].T
        static_difference = float(np.max(np.abs(
            static['eci'][iv] - static_fit[None, :]
        )))
        harmonic_difference = float(np.max(np.abs(
            harmonic['eci'][iv] - harmonic_fit
        )))
        # This tolerance concerns numerical replay, not physical accuracy.
        assert static_difference < 1e-10, (iv, static_difference)
        assert harmonic_difference < 1e-10, (iv, harmonic_difference)
        prediction = np.einsum('tk,ck->tc', harmonic_fit, X)
        assert np.isfinite(prediction).all()
        errors = prediction - harmonic['Y_eV_atom'][iv].T
        control = ~training & stable[iv]
        rms = np.sqrt(np.mean(errors[:, control] ** 2, axis=1))
        assert np.isfinite(rms).all()
        rows.append({
            'volume_A3_atom': float(volume),
            'fitting_configurations': int(selected.sum()),
            'control_configurations': int(control.sum()),
            'rank': rank,
            'static_coefficient_replay_max_abs_eV': static_difference,
            'harmonic_coefficient_replay_max_abs_eV': harmonic_difference,
            'control_total_free_energy_RMSE_eV_atom_by_temperature': rms.tolist(),
        })
    interpolation = {}
    temperatures = static['temperatures_K']
    volumes = static['volumes_A3_atom']
    for label, data in [('static', static), ('HA_QHA', harmonic)]:
        derivative = CubicSpline(temperatures, data['eci'], axis=1)(temperatures, 1)
        derivative_difference = float(np.max(np.abs(
            derivative - data['d_eci_dT_K']
        )))
        assert derivative_difference < 1e-10
        # The temperature-independent static native table stores endpoints only.
        indices = [0, len(temperatures) - 1] if label == 'static' else list(range(len(temperatures)))
        table_temperatures = temperatures[indices]
        vs = CubicSpline(volumes, data['eci'][:, indices], axis=0)
        ds = CubicSpline(volumes, derivative[:, indices], axis=0)
        expected = np.concatenate([vs.c[::-1], ds.c[::-1]], axis=0)
        expected = expected.transpose(2, 1, 3, 0).reshape(-1, 8)
        table = package / 'models' / label / 'qha_ce.dat'
        with table.open() as stream:
            assert stream.readline().strip() == 'QHA_CE_V1'
            assert list(map(int, stream.readline().split())) == [9, len(indices), 20]
            kb = float(stream.readline())
            assert abs(kb - 8.617333262145e-5) < 1e-18
            assert np.array_equal(np.fromstring(stream.readline(), sep=' '), volumes)
            assert np.array_equal(np.fromstring(stream.readline(), sep=' '), table_temperatures)
            actual = np.loadtxt(stream)
        assert actual.shape == expected.shape
        table_difference = float(np.max(np.abs(actual - expected)))
        assert table_difference < 1e-10
        interpolation[label] = {
            'temperature_derivative_max_abs_eV_K': derivative_difference,
            'volume_spline_table_max_abs_difference': table_difference,
            'native_table_rows': int(len(actual)),
            'native_table_sha256': hashlib.sha256(table.read_bytes()).hexdigest(),
        }
    return {
        'all_fits_reproduced': True,
        'configurations': 46,
        'coefficients': 20,
        'temperatures_K': static['temperatures_K'].tolist(),
        'volume_results': rows,
        'interpolation_replay': interpolation,
        'input_sha256': {
            label: hashlib.sha256(path.read_bytes()).hexdigest()
            for label, path in paths.items()
        },
        'qualification': (
            'Replay of saved static and total harmonic least-squares fits. '
            'No new MTP training or calculation. The exact temperature '
            'derivative coefficients are reproduced from the coefficient '
            'temperature spline used by C46; they are not direct entropy '
            'fits. Volume spline tables are reproduced as well. '
            'This replay does not validate phonon convergence, '
            'chemical sampling, phase boundaries, or MTP accuracy.'
        ),
    }


if __name__ == '__main__':
    print(json.dumps(reproduce(Path(sys.argv[1])), indent=2, allow_nan=False))
