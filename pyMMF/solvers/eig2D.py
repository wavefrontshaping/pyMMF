"""
Solver based on finite difference solution of the eigenvalue problen of the Helmholtz scalar equation.
"""

import numpy as np
import time
import scipy.sparse as sparse
from scipy.sparse.linalg import eigs

from ..modes import Modes
from ..logger import get_logger

logger = get_logger(__name__)


def solve_eig(
    indexProfile,
    wl,
    nmodesMax=6,
    curvature=None,
    propag_only=True,
    poisson=0.5,
    boundary="close",
):
    """
    Find the first modes of a multimode fiber. The index profile has to be set.
    Returns a Modes structure containing the mode information.
    This method is slow and requires a high resolution (so very slow)
    to converge to the correct modes.
    It is thus suitable for a low number of modes.
    However, it is the more general method and can be used for any index profile,
    even non-rotationally symmetric ones.

    Options
    -------
        nmodesMax: int, optional
            Maximum number of modes the solver will try to find.
            This value should be higher than the estimated maximum number of modes if one want to be sure
            to find all the modes.
            One can use the helper function :ref:`pyMMF.estimateNumModesSI` to estimate the number of modes
            and add a few more modes for safety.
            Default is 6
        curvature: float or List[float] or None, optional
            Curvature of the fiber in meters.
            If a list is provided, the first element is the curvature in the x
            direction and the second element is the curvature in the y direction.
            x and y directions are the ones defined in the index profile.
            If a single value is provided, the curvature is assumed to be in the x direction.
            If None, the curvature is not taken into account.
            Default is None
        propag_only: bool, optional
            If True, only propagating modes are returned, the others are rejected.
            It thus can return a number of modes lower than `nmodesMax`.
            If False, all the modes are returned,
            potentially including non-propagating modes with an non-zero imaginary part
            of the propagation constant.
            Default is True
        poisson: float, optional
            Poisson coefficient of the material.
            It is used to take into account the effect of compression/dilatation
            when the fiber is bent (curvature is not None).
            Default is 0.5
        boundary: string, optional
            boundary type, 'close' or 'periodic'
            EXPERIMENTAL.
            It should not make any difference for propagating modes.
            Default is 'close'



    """
    # curvature = options.get("curvature", None)
    # nmodesMax = options.get("nmodesMax", 6)
    # boundary = options.get("boundary", "close")
    # propag_only = options.get("propag_only", True)
    # poisson = options.get("poisson", 0.5)

    t0 = time.time()

    k0 = 2.0 * np.pi / wl
    npoints = indexProfile.npoints
    diags = []
    logger.info("Solving the spatial eigenvalue problem for mode finding.")

    ## Construction of the operator
    dh = indexProfile.dh
    diags.append(-4.0 / dh**2 + k0**2 * indexProfile.n.flatten() ** 2)

    if boundary == "periodic":
        logger.info("Use periodic boundary condition.")
        diags.append(
            ([1.0 / dh**2] * (npoints - 1) + [0.0]) * (npoints - 1)
            + [1.0 / dh**2] * (npoints - 1)
        )
        diags.append(
            ([1.0 / dh**2] * (npoints - 1) + [0.0]) * (npoints - 1)
            + [1.0 / dh**2] * (npoints - 1)
        )
        diags.append([1.0 / dh**2] * npoints * (npoints - 1))
        diags.append([1.0 / dh**2] * npoints * (npoints - 1))

        diags.append(
            ([1.0 / dh**2] + [0] * (npoints - 1)) * (npoints - 1) + [1.0 / dh**2]
        )
        diags.append(
            ([1.0 / dh**2] + [0] * (npoints - 1)) * (npoints - 1) + [1.0 / dh**2]
        )

        diags.append([1.0 / dh**2] * npoints)
        diags.append([1.0 / dh**2] * npoints)

        offsets = [
            0,
            -1,
            1,
            -npoints,
            npoints,
            -npoints + 1,
            npoints - 1,
            -npoints * (npoints - 1),
            npoints * (npoints - 1),
        ]
    elif boundary == "close":
        logger.info("Use close boundary condition.")

        # x parts of the Laplacian
        diags.append(
            ([1.0 / dh**2] * (npoints - 1) + [0.0]) * (npoints - 1)
            + [1.0 / dh**2] * (npoints - 1)
        )
        diags.append(
            ([1.0 / dh**2] * (npoints - 1) + [0.0]) * (npoints - 1)
            + [1.0 / dh**2] * (npoints - 1)
        )
        # y parts of the Laplacian
        diags.append([1.0 / dh**2] * npoints * (npoints - 1))
        diags.append([1.0 / dh**2] * npoints * (npoints - 1))

        offsets = [0, -1, 1, -npoints, npoints]

    if curvature is not None:
        # xi term,
        # - the 1. term represent the geometrical effect
        # - the term in (1-2*poisson_coeff) represent the effect of compression/dilatation
        # see http://wavefrontshaping.net/index.php/68-community/tutorials/multimode-fibers/149-multimode-fiber-modes-part-2
        xi = 1.0 - (indexProfile.n.flatten() - 1.0) / indexProfile.n.flatten() * (
            1.0 - 2.0 * poisson
        )

        #            curv_mat = sparse.diags(1.-2*xi*self.indexProfile.X.flatten()/curvature, dtype = np.complex128)
        curv_inv_diag = 1.0
        if curvature[0] is not None:
            curv_inv_diag += 2 * xi * indexProfile.X.flatten() / curvature[0]
        if curvature[1] is not None:
            curv_inv_diag += 2 * xi * indexProfile.Y.flatten() / curvature[1]
        curv_mat = sparse.diags(1.0 / curv_inv_diag, dtype=np.complex128)
    #            curv_mat = sparse.diags(1./(1.+2*xi*self.indexProfile.X.flatten()/curvature), dtype = np.complex128)

    #        logger.info('Note that boundary conditions should not matter too much for guided modes.')

    H = sparse.diags(diags, offsets, dtype=np.complex128)

    if curvature:
        H = curv_mat.dot(H)

    beta_min = k0 * np.min(indexProfile.n)
    beta_max = k0 * np.max(indexProfile.n)

    # Finds the eigenvalues of the operator with the greatest real part
    res = eigs(H, k=nmodesMax, which="LR")

    modes = Modes()
    modes.wl = wl
    modes.indexProfile = indexProfile
    # select only the propagating modes
    for i, betasq in enumerate(res[0]):
        if (betasq > beta_min**2 and betasq < beta_max**2) or not propag_only:
            modes.betas.append(np.sqrt(betasq))
            modes.number += 1
            modes.profiles.append(res[1][:, i])
            modes.profiles[-1] = modes.profiles[-1] / np.sqrt(
                np.sum(np.abs(modes.profiles[-1]) ** 2)
            )
            # is the mode a propagative one?
            modes.propag.append((betasq > beta_min**2 and betasq < beta_max**2))

    logger.info(
        "Solver found %g modes is %0.2f seconds." % (modes.number, time.time() - t0)
    )

    if nmodesMax == modes.number:
        logger.warning("The solver reached the maximum number of modes set.")
        logger.warning("Some propagating modes may be missing.")

    return modes
