"""
Solver for axisymmetric index profile defined by a radial function,
e.g. graded index fibers.
Solve the 1D problem using the finite difference recursive scheme for 
Riccati's equations.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq as bisect
from functools import partial
from numba import jit
import time
import copy

from ..modes import Modes
from ..logger import get_logger


logger = get_logger(__name__)

MIN_RADIUS_BC_COEFF_DEFAULT = 4
CHANGE_BC_RADIUS_STEP_DEFAULT = 0.9
N_BETA_COARSE_DEFAULT = int(1e3)
DEFAULT_DEGENERATE_MODE = "sin"


# choice for degenerate subspaces
def _expi(x, s=1):
    return np.exp(s * 1j * x)


EXP_PHASE_FUNCS = [partial(_expi, s=1), partial(_expi, s=-1)]
SIN_PHASE_FUNCS = [np.sin, np.cos]


class PrecisionError(Exception):
    def __init__(self, min_val, max_val):
        self.msg = f"Stagnation due to floating point precision in the interval ({min_val}, {max_val})"
        logger.error(self.msg)
        super().__init__(self.msg)


class BisectRootValueError(Exception):
    def __init__(self, root, fval, ftol):
        self.msg = f"Field limit {fval} at the founded beta={root} is greater than field_limit_tol={ftol}"
        logger.error(self.msg)
        super().__init__(self.msg)


class BisectNotConvergedError(Exception):
    def __init__(self, beta, fval, binfo):
        self.msg = (
            f"Binary search was not converged with beta={beta}, f={fval}\n{binfo}"
        )
        logger.error(self.msg)
        super().__init__(self.msg)


class SmallRmaxError(Exception):
    def __init__(self, r_max, min_radius_bc):
        self.msg = (
            f"Boundary condition could not be met for r_max < {min_radius_bc}a\n"
            + "\tTry lower your tolerence or change the resolution."
        )
        logger.error(self.msg)
        super().__init__(self.msg)


class CalculationStopException(Exception):
    def __init__(self):
        self.msg = "Calculation stops"
        logger.error(self.msg)
        super().__init__(self.msg)


@jit(
    "float64[:](int8,float32,float32[:],float32[:],float64, float64, float64)",
    nopython=True,
    fastmath=True,
)
def _get_field_fast(m, dh, r, nr, beta_min, delta_beta, k0):
    beta_sq = beta_min**2 + delta_beta**2 + 2 * beta_min * delta_beta

    def Q(r):
        return 1 / r

    def P(r, n):
        return (k0 * n) ** 2 - beta_sq - (m / r) ** 2

    def iter_g(gn, r, n):
        Qh = dh * Q(r) / 2
        return (-(dh**2 * P(r, n) - 2 + (1 - Qh) / (1 + dh * gn)) / (1 + Qh) - 1) / dh

    # Boundary conditions at r~0
    if m == 0:
        g0 = [0.0]
        f0 = [1.0]
    else:
        g0 = [0.0, (1 - dh**2 * P(r[1], nr[1])) / dh]
        f0 = [0.0, dh]

    bc_length = len(f0)

    f_vec = np.empty(shape=r.shape, dtype=np.float64)
    f_vec[:bc_length] = f0

    g_vec = np.empty(shape=r.shape, dtype=np.float64)
    g_vec[:bc_length] = g0

    for ir in range(bc_length, len(r)):
        g_vec[ir] = iter_g(g_vec[ir - 1], r[ir], nr[ir])
        f_vec[ir] = f_vec[ir - 1] * (1 + dh * g_vec[ir - 1])
    return f_vec


def get_field_fast(m, dh, r, nr, beta_min, delta_beta, k0):
    """
    Get the field calulcated using the recursive scheme for the quadratic Ricatti
    equation as [#]_.

    Be careful that numba does not allow float128, which limits the precision
    and can lead to stagnation of the recursive iterations.

    .. [#] Lakshman S. Tamil, S. S. Mitra, R. Dutta, and J. M. T. Pereira,
       "Finite difference solution for graded-index cylindrical dielectric waveguides:
       a scalar wave approximation" Applied Optics, vol. 30,
       pp. 1113-1116, 1991.
    """
    return _get_field_fast(
        np.int8(m),
        np.float32(dh),
        r.astype(np.float32),
        nr.astype(np.float32),
        np.float64(beta_min),
        np.float64(delta_beta),
        np.float64(k0),
    )


def scan_betas(m, dh, r, nr, beta_min, delta_betas, k0):
    """
    rough scan of betas values between beta_min and beta_max
    returns the sign of the value of the function for the
    farthest radial point for each beta value.
    (sign changes indicate the presence of an eigenvector)
    """
    return [
        np.sign(get_field_fast(m, dh, r, nr, beta_min, delta_beta, k0)[-1])
        for delta_beta in delta_betas
    ]


def binary_search(func, min_val, max_val, sign, beta_tol=1e-12, field_limit_tol=1e-3):
    max_val_incr_factor = 1.1

    if max_val - min_val < beta_tol:
        raise PrecisionError(min_val, max_val)

    converged = 0
    while not converged:
        try:
            beta, binfo = bisect(
                func, min_val, max_val, xtol=beta_tol, full_output=True
            )
        except ValueError:
            max_val *= max_val_incr_factor
            logger.error(
                f"Field at min_val={min_val} and max_val={max_val} have the same sign."
            )
            raise CalculationStopException(
                "Field at min_val and max_val have the same sign."
            )
        converged = binfo.converged

    fval = func(beta)
    if np.abs(fval) > field_limit_tol:
        raise BisectRootValueError(beta, fval, field_limit_tol)

    binfo.fval = fval
    return beta, binfo


def solve_radial(indexProfile, wl, **options):
    t0 = time.time()

    degenerate_mode = options.get("degenerate_mode", DEFAULT_DEGENERATE_MODE)
    phi_funcs = EXP_PHASE_FUNCS if degenerate_mode == "exp" else SIN_PHASE_FUNCS
    min_radius_bc = options.get("min_radius_bc", MIN_RADIUS_BC_COEFF_DEFAULT)
    change_bc_radius_step = options.get(
        "change_bc_radius_step", CHANGE_BC_RADIUS_STEP_DEFAULT
    )
    N_beta_coarse = options.get("N_beta_coarse", N_BETA_COARSE_DEFAULT)
    r_max0 = options.get("r_max", np.max(indexProfile.R))
    dh = options.get("dh", indexProfile.areaSize / indexProfile.npoints)
    beta_tol = options.get("beta_tol", np.finfo(np.float64).eps)
    field_limit_tol = options.get("field_limit_tol", 1e-3)
    save_func = options.get("save_func", False)

    k0 = 2.0 * np.pi / wl

    n_func = indexProfile.radialFunc
    radius = indexProfile.a

    r = np.arange(0, r_max0 + dh, dh).astype(np.float64)

    beta_min = k0 * n_func(r_max0)
    beta_max = k0 * n_func(0)
    delta_betas = np.linspace(0, beta_max - beta_min, N_beta_coarse)

    modes = Modes()
    modes.wl = wl
    modes.indexProfile = indexProfile

    m = 0

    while True:
        if m == 0:
            r = r - dh / 2

        nr = np.array([n_func(rn) for rn in r], dtype=np.float64)

        # rough scan of beta values to find the changes of the sign of
        # the farthest radial point
        sign_f = scan_betas(
            m=m, dh=dh, r=r, nr=nr, beta_min=beta_min, delta_betas=delta_betas, k0=k0
        )

        # finds where fN change sign
        zero_crossings = np.where(np.diff(sign_f))[0][::-1]
        logger.info(f"Found {len(zero_crossings)} radial mode(s) for m={m}")
        if len(zero_crossings) == 0:
            break

        for l, iz in enumerate(zero_crossings):
            logger.info(f"Searching propagation constant for |l| = {l+1}")
            # find the beta value that satisfies the best the boundary condition
            r_max = r_max0
            while True:
                if r_max < min_radius_bc * wl:
                    raise SmallRmaxError(r_max, min_radius_bc)

                r_search = r[r <= r_max]
                n_search = nr[r <= r_max]

                try:
                    #'Searching for beta value that satisfies the zero condition
                    # at r={r_max/radius:.3f}a'
                    def func_fast(delta_beta):
                        f = get_field_fast(
                            m, dh, r_search, n_search, beta_min, delta_beta, k0
                        )
                        return f[-1] / np.max(np.abs(f))

                    delta_beta, binfo = binary_search(
                        func_fast,
                        min_val=delta_betas[iz],
                        max_val=delta_betas[iz + 1],
                        sign=sign_f[iz],
                        beta_tol=beta_tol,
                        field_limit_tol=field_limit_tol,
                    )
                    if not binfo.converged:
                        raise BisectNotConvergedError(
                            delta_beta, func_fast(delta_beta), binfo
                        )
                    # get the discretized radial function
                    f_vec = get_field_fast(
                        m, dh, r_search, n_search, beta_min, delta_beta, k0
                    )
                    # get the radial function by interpolation
                    # assume f = 0 for r>r_max
                    f_interp = np.pad(
                        f_vec, (0, len(r) - len(f_vec)), "constant", constant_values=0
                    )
                    # assume f = 0 for r>radius * min_radius_bc
                    f_r = interp1d(
                        r, f_interp, kind="cubic", bounds_error=False, fill_value=0
                    )

                    if save_func:
                        # normalize the radial function
                        def radial_norm(t, r_vec, d):
                            return np.sqrt(
                                2 * np.pi * np.sum(np.abs(t) ** 2 * r_vec) * d
                            )

                        dr = dh / 2
                        r_vec = np.arange(0, r_max, dr)
                        norm_fr = radial_norm(f_r(r_vec), r_vec, dr)

                    break
                except (BisectNotConvergedError, PrecisionError, BisectRootValueError):
                    logger.warning("Boundary condition could not be met.")
                    r_max *= change_bc_radius_step
                    logger.warning(f"Retrying by changing r_max to {r_max/radius:.2f}a")
                except Exception as E:
                    logger.error(f"Unknown exception: {E}")
                    raise CalculationStopException(f"Unknown exception: {E}")

            mode_profile = f_r(indexProfile.R)

            # add mode
            if m == 0:
                modes.betas.append(delta_beta + beta_min)
                modes.m.append(m)
                modes.l.append(l + 1)
                modes.number += 1
                modes.profiles.append(mode_profile.ravel())
                modes.profiles[-1] = modes.profiles[-1] / np.sqrt(
                    np.sum(np.abs(modes.profiles[-1]) ** 2)
                )
                # is the mode a propagative one?
                modes.propag.append(True)

                if save_func:
                    modes.data.append(
                        {
                            "radial_func": f_r,
                            "r_max": r_max,
                            "norm": norm_fr,
                            "azimuthal_func": lambda x: 0,
                        }
                    )
            else:
                for s, phi_func in zip([-1, 1], phi_funcs):
                    modes.betas.append(delta_beta + beta_min)
                    modes.m.append(s * m if degenerate_mode == "exp" else m)
                    modes.l.append(l + 1)
                    modes.number += 1
                    modes.profiles.append(
                        mode_profile.ravel() * phi_func(m * indexProfile.TH.ravel())
                    )
                    modes.profiles[-1] = modes.profiles[-1] / np.sqrt(
                        np.sum(np.abs(modes.profiles[-1]) ** 2)
                    )
                    # is the mode a propagative one?
                    modes.propag.append(True)

                    if save_func:
                        modes.data.append(
                            {
                                "radial_func": f_r,
                                "r_max": r_max,
                                "norm": norm_fr,
                                "azimuthal_func": phi_func,
                            }
                        )

        m += 1
        r_max = np.max(indexProfile.R)
    # sort the modes
    modes.sort()
    logger.info(
        "Solver found %g modes is %0.2f seconds." % (modes.number, time.time() - t0)
    )
    return modes
