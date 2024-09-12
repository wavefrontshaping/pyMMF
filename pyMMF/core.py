# -*- coding: utf-8 -*-
"""
Simple module to find numerically the propagation modes and their corresponding propagation constants
of multimode fibers of arbitrary index profiles.

Based on 
- http://wavefrontshaping.net/index.php/tutorials
and
- M. Plöschner, T. Tyc, and T. Čižmár, Nature Photonics Seeing through chaos in multimode fibres 9, 529–535 (2015).
https://doi.org/10.1038/nphoton.2015.112

written by Sebastien M. Popoff
"""

import numpy as np
import sys, time
from .modes import Modes
from .logger import get_logger, handleException
from .solvers import (
    solve_eig,
    solve_SI,
    solve_radial_legacy,
    solve_radial,
    solve_WKB,
)


class AssertionError(Exception):
    def __init__(self):
        self.msg = "Invalid combination of index profile type and solver."
        logger.error(self.msg)
        super().__init__(self.msg)


logger = get_logger(__name__)

sys.excepthook = lambda excType, excValue, traceback: handleException(
    excType, excValue, traceback, logger=logger
)
# %%


def estimateNumModesGRIN(wl, a, NA, pola=1):
    """
    Returns a rough estimation of the number of propagating modes of a GRIN fiber.
    See https://www.rp-photonics.com/v_number.html for more details.

    Parameters
    ----------

    wl : float
        Wavelength (in microns)
    a :  float
        Radius of the fiber (in microns)
    NA : float
        Numerical aperture of the fiber
    pola : int (1 or 2)
        Number of polarizations

    Returns
    -------

    N : integer
        Estimation of the number of propagating modes

    """
    k0 = 2.0 * np.pi / wl
    V = k0 * a * NA
    return np.ceil(V**2 / 4.0 * pola / 2.0).astype(int)


def estimateNumModesSI(wl, a, NA, pola=1):
    """
    Returns a rough estimation of the number of propagating modes of a step index fiber.
    See https://www.rp-photonics.com/v_number.html for more details.

    Parameters
    ----------
        wl : float
                 Wavelength (in microns)
        a :  float
                 Radius of the fiber (in microns)
        NA : float
                    Numerical aperture of the fiber
        pola : int (1 or 2)
            Number of polarizations

    Returns
    -------
        N : integer
            Estimation of the number of propagating modes

    """
    k0 = 2.0 * np.pi / wl
    V = k0 * a * NA
    return np.ceil(V**2 / 2.0 * pola / 2.0).astype(int)


# %%


class propagationModeSolver:
    """
    Class for solving the scalar wave equation in multimode fiber.
    """

    def __init__(self):
        self.betas = []
        self.modes = None
        self.modesList = []
        self.number = 0
        self.m = []
        self.l = []
        self.indexProfile = None
        self.wl = None
        self.last_res = None
        self.poisson = 0.5

        logger.debug("Debug mode ON.")

    def setIndexProfile(self, indexProfile):
        """
        Set the index profile for the MMF.

        Parameters
        ----------
            indexProfile : IndexProfile
                The index profile object representing the fiber section geometry.

        Returns:
        None
        """
        self.indexProfile = indexProfile

    def setWL(self, wl):
        """
        Set the wavelength (in microns).

        Parameters
        ----------
            wl : float
             Wavelength in microns.

        """
        self.wl = wl

    def setPoisson(self, poisson):
        """
        Set the poisson coefficient. The default value is 0.5 (no effect of compression/dilatation)

        Parameters
        ----------
            poisson : float
                  Poisson coefficient of the fiber material.

        """
        self.poisson = poisson

    def solve(
        self,
        solver: str = "default",
        curvature: bool = None,
        storeData: bool = True,
        options: dict = {},
    ):
        """
        Find the propagation constants and mode profile of a multimode fiber.
        For an arbitrary index profile, it finds the solution of the eigenvalue problem of the scalar wave equation in a discretized space [1].

        Parameters
        ----------

            storeData: bool
                Stores data in the propagationModeSolver object is set to True
                defaults to True

            curvature: float
                Curvature of the fiber in meters
                defaults to None

            solver: string ('default', 'radial', 'eig', 'SI', or 'WKB')
                solver to be used.Type of solver.  Should be one of

                    - 'radial' solves the 1D radial wave equation for axisymmetric profiles.
                    It requires the profile to be defined by a radial function
                    with the :meth:`initFromRadialFunction<pyMMF.IndexProfile.initFromRadialFunction>` method.

                    :ref:`(more info  and additional options here) <pyMMF.solve-radial>`

                    - 'eig' solves the eigenvalue problem in the discretized space.
                    Slower and less precise, but allows for non-axisymmetric profiles and bend curvature.

                    :ref:`(more info  and additional options here) <pyMMF.solve-eig>`

                    - 'SI' solves numerically the analytical dispersion relation for step index fibers
                    and approximate modes to LP modes.

                    :ref:`(more info  and additional options here) <pyMMF.solve-SI>`

                    - 'WKB' uses the WKB approximation for GRIN profiles.

                    :ref:`(more info  and additional options here) <pyMMF.solve-WKB>`

                    - 'default' tries to find the optimal solver for the given index profile.
                Default is 'default' ;)

            options: dict
                A dictionary of solver options.
                They are specific to the solver used,
                read the documentation of the solver for more information.

        Returns
        -------

            modes : Modes
                Modes object containing all the mode information.


        """
        assert self.indexProfile
        assert self.wl
        # check if cuvature is a list or array of length 2 or None
        if curvature == None:
            pass
        elif hasattr(curvature, "__len__") and len(curvature) == 2:
            if 0 in curvature:
                logger.error("curvature = 0 not allowed!")
                raise (ValueError("curvature = 0 not allowed!"))
        elif isinstance(curvature, float) or isinstance(curvature, int):
            # if only one value for curvature, use the curvatrue for the X axis and add curvature = None for the Y axis
            curvature = [curvature, None]
        else:
            logger.error("Wrong type of data for curvature.")
            raise (ValueError("Wrong type of data for curvature."))

        if solver == "default":
            solver = self.indexProfile.getOptimalSolver(curvature)

        if solver == "SI":
            if not (self.indexProfile.type == "SI"):
                logger.error("SI solver only available for step-index profiles")
                raise AssertionError
            if curvature is not None:
                logger.error(
                    "Semi-analytical solution of step-index fiber is not compatible with curvature."
                )
                raise AssertionError
            modes = solve_SI(self.indexProfile, self.wl, **options)
        elif solver == "radial":
            if self.indexProfile.radialFunc is None:
                logger.error(
                    "radial solver only available for axisymmetric profiles defined by a radial function"
                )
                raise AssertionError
            modes = solve_radial(self.indexProfile, self.wl, **options)

        elif solver == "radial_legacy":
            if self.indexProfile.radialFunc is None:
                logger.error(
                    "radial solver only available for axisymmetric profiles defined by a radial function (legacy)"
                )
                raise AssertionError
            modes = solve_radial_test(self.indexProfile, self.wl, **options)

        elif solver == "eig":
            modes = solve_eig(
                indexProfile=self.indexProfile,
                wl=self.wl,
                curvature=curvature,
                **options,
            )
        elif solver == "WKB":
            if self.indexProfile.type != "GRIN":
                logger.error("WKB solver only available for parabolic GRIN profiles")
                raise AssertionError
            modes = solve_WKB(self.indexProfile, self.wl, **options)
        else:
            raise ValueError("Invalid mode")

        if storeData:
            self.modes = modes
            logger.debug("Mode data stored in memory.")

        modes.poison = self.poisson
        modes.indexProfile = self.indexProfile
        modes.wl = self.wl
        modes.curvature = curvature

        return modes
