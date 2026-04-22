import numpy as np
from scipy.ndimage import zoom as nd_zoom

import pyMMF
from pyMMF.tools.autofit import Autofit


def _resample_modes(M_hr, target_N, N_hr):
    factor = target_N / N_hr
    out = np.empty((target_N * target_N, M_hr.shape[1]), dtype=M_hr.dtype)
    for k in range(M_hr.shape[1]):
        img = M_hr[:, k].reshape(N_hr, N_hr)
        real = nd_zoom(img.real, factor, order=3)
        imag = nd_zoom(img.imag, factor, order=3)
        out[:, k] = (real + 1j * imag).ravel()
    return out


def test_autofit_realign_TM():
    # Fiber parameters (mirrors docs/examples/Autofit.ipynb)
    NA = 0.2
    radius = 25.0
    areaSize = 3.0 * radius
    n_points_modes = 48
    n1 = 1.45
    wl = 1.55

    r_max = 3.2 * radius
    npoints_search = 2**8
    dh = 2 * radius / npoints_search

    solver_options = {
        "degenerate_mode": "exp",
        "min_radius_bc": 1.5,
        "N_beta_coarse": 1_000,
        "change_bc_radius_step": 0.95,
        "dh": dh,
        "r_max": r_max,
    }

    profile = pyMMF.IndexProfile(npoints=n_points_modes, areaSize=areaSize)
    profile.initParabolicGRIN(n1=n1, a=radius, NA=NA)

    solver = pyMMF.propagationModeSolver()
    solver.setIndexProfile(profile)
    solver.setWL(wl)
    modes = solver.solve(
        solver="radial", curvature=None, options=solver_options
    )

    # Build a non-square clean TM using distinct output and input grids
    n_out = 48
    n_in = 32
    M0_hr = modes.getModeMatrix()
    N_hr = modes.indexProfile.npoints

    M0_out = _resample_modes(M0_hr, n_out, N_hr)
    M0_in = _resample_modes(M0_hr, n_in, N_hr)

    af = Autofit(modes)
    TM_mode_basis = modes.getPropagationMatrix(1e4)
    TM_0 = M0_out @ TM_mode_basis @ M0_in.conj().T
    assert TM_0.shape == (n_out * n_out, n_in * n_in)

    # Asymmetric misalignment
    params = [(1.15, 0.9), (2.0, 3.0), (-4.0, 3.0)]
    TM_misaligned = af.transform(TM_0, params=params)
    assert TM_misaligned.shape == TM_0.shape

    # Realign
    TM_realigned, new_modes_out_af, new_modes_in_af = af.realign_TM(
        TM_misaligned, params={"threshold": 0.5}
    )
    assert TM_realigned.shape == TM_0.shape
    assert new_modes_out_af.shape == (n_out * n_out, modes.number)
    assert new_modes_in_af.shape == (n_in * n_in, modes.number)

    # Projection onto the resampled mode basis: conversion efficiency.
    # realign_TM returns modes normalized by their global Frobenius norm,
    # so we re-normalize per column to get an orthonormal basis before
    # measuring the energy preserved by the projection.
    new_modes_out_col = new_modes_out_af / np.linalg.norm(
        new_modes_out_af, axis=0, keepdims=True
    )
    new_modes_in_col = new_modes_in_af / np.linalg.norm(
        new_modes_in_af, axis=0, keepdims=True
    )
    new_TM_modes_af = (
        new_modes_out_col.conj().T @ TM_realigned @ new_modes_in_col
    )
    conversion_efficiency = np.linalg.norm(new_TM_modes_af) / np.linalg.norm(
        TM_realigned
    )
    assert conversion_efficiency > 0.998, (
        f"Conversion efficiency too low: {conversion_efficiency:.6f}"
    )
