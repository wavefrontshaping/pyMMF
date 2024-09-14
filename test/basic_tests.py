import pytest
import numpy as np

from pyMMF import IndexProfile, propagationModeSolver
import pyMMF


def test_GRIN():

    # parameters
    NA = 0.2
    radius = 25.0
    areaSize = 2.4 * radius
    npoints = 2**8
    n1 = 1.45
    wl = 1.55

    k0 = 2 * np.pi / wl

    # expected
    n_modes = 55
    n2 = np.sqrt(n1**2 - NA**2)
    solver_type = "radial"
    V = k0 * radius * NA
    estimated_mode_number_with_V = np.ceil(V**2 / 8.0).astype(int)

    beta_min = k0 * n2
    beta_max = k0 * n1

    # tests for index profile

    profile = IndexProfile(npoints=npoints, areaSize=areaSize)
    profile.initParabolicGRIN(n1=n1, a=radius, NA=NA)

    assert np.min(profile.n) == pytest.approx(n2)
    assert np.max(profile.n) == pytest.approx(n1)

    solver_type = profile.getOptimalSolver()
    assert solver_type == solver_type

    # test for solver/modes

    solver = propagationModeSolver()
    solver.setIndexProfile(profile)
    solver.setWL(wl)

    assert (
        pyMMF.estimateNumModesGRIN(wl, radius, NA, pola=1)
        == estimated_mode_number_with_V
    )
    modes = solver.solve(solver="radial")

    assert modes.number == n_modes

    assert np.min(modes.betas) >= beta_min
    assert np.max(modes.betas) <= beta_max


def test_SI():

    # parameters
    NA = 0.15
    radius = 10.0
    areaSize = 3.5 * radius
    npoints = 2**7
    n1 = 1.45
    wl = 0.6328

    k0 = 2 * np.pi / wl

    # expected
    n_modes = 59
    n2 = np.sqrt(n1**2 - NA**2)
    solver_type = "SI"
    V = k0 * radius * NA
    estimated_mode_number_with_V = np.ceil(V**2 / 4.0).astype(int)

    beta_min = k0 * n2
    beta_max = k0 * n1

    # tests for index profile

    profile = IndexProfile(npoints=npoints, areaSize=areaSize)
    profile.initStepIndex(n1=n1, a=radius, NA=NA)

    assert np.min(profile.n) == pytest.approx(n2)
    assert np.max(profile.n) == pytest.approx(n1)

    solver_type = profile.getOptimalSolver()
    assert solver_type == solver_type

    # test for solver/modes

    solver = propagationModeSolver()
    solver.setIndexProfile(profile)
    solver.setWL(wl)

    assert (
        pyMMF.estimateNumModesSI(wl, radius, NA, pola=1) == estimated_mode_number_with_V
    )

    modes_semianalytical = solver.solve(solver="SI")

    modes_semianalytical.sort()

    assert modes_semianalytical.number == n_modes

    assert np.min(modes_semianalytical.betas) >= beta_min
    assert np.max(modes_semianalytical.betas) <= beta_max

    solver_options = {
        "boundary": "close",
        "nmodesMax": estimated_mode_number_with_V + 10,
        "propag_only": True,
    }

    modes_eig = solver.solve(
        solver="eig",
        curvature=None,
        options=solver_options,
    )
    modes_eig.sort()

    betas_eig = modes_eig.betas
    betas_semianalytical = modes_semianalytical.betas

    assert len(betas_eig) == len(betas_semianalytical)
    assert np.allclose(betas_eig, betas_semianalytical, rtol=1e-4)


def test_custom():

    # parameters
    NA = 0.2
    # square index profile of size 2*radius
    radius = 12.5
    areaSize = 3.0 * radius
    npoints = 2**8
    n1 = 1.45
    n2 = np.sqrt(n1**2 - NA**2)
    wl = 1.55

    k0 = 2 * np.pi / wl

    # expected
    n_modes = 36
    solver_type = "eig"
    beta_min = k0 * n2
    beta_max = k0 * n1

    V = k0 * radius * NA
    estimated_mode_number_with_V = np.ceil(V**2 / 2.0 * 4 / np.pi).astype(int) // 2

    # tests for index profile

    profile = IndexProfile(npoints=npoints, areaSize=areaSize)

    index_array = n2 * np.ones((npoints, npoints))
    mask_core = (np.abs(profile.X) < radius) & (np.abs(profile.Y) < radius)
    index_array[mask_core] = n1
    profile.initFromArray(index_array)

    assert np.all(profile.n == index_array)
    assert np.min(profile.n) == pytest.approx(n2)
    assert np.max(profile.n) == pytest.approx(n1)

    solver_type = profile.getOptimalSolver()
    assert solver_type == solver_type

    # test for solver/modes

    solver = propagationModeSolver()
    solver.setIndexProfile(profile)
    solver.setWL(wl)

    solver_options = {
        "boundary": "close",
        "nmodesMax": estimated_mode_number_with_V + 10,
        "propag_only": True,
    }

    modes = solver.solve(
        solver=solver_type,
        options=solver_options,
    )

    assert modes.number == n_modes

    assert np.min(modes.betas) >= beta_min
    assert np.max(modes.betas) <= beta_max
