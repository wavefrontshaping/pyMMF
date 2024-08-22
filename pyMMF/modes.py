#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 12:02:57 2019

@author: Sébastien M. Popoff
"""

import numpy as np
import pickle
from scipy.linalg import expm
from scipy.ndimage import rotate as scipy_rotate
from scipy.ndimage import shift as scipy_shift
from .logger import get_logger, handleException
import itertools

logger = get_logger(__name__)


class Modes:
    """
    Class to store the modes of a multimode fiber,
    typically returned by the solver

    See Also
    --------
    pyMMF.propagationModeSolver
    """

    def __init__(self):
        self.betas = []
        self.propag = []
        self.u = []
        self.w = []
        self.modesList = []
        self.number = 0
        self.m = []
        self.l = []
        self.indexProfile = None
        self.profiles = []
        self.modeMatrix = None
        self.wl = None
        self.curvature = None
        self.poisson = 0.5
        self.data = []

    def getModeMatrix(self, npola=1, shift=None, angle=None):
        """
        Returns the matrix containing the mode profiles.
        Note that while you can set two polarizations, the modes profiles are obtained under a scalar apporoximation.

        Parameters
        ----------
        npola : int
            Number of polarizations considered. For npola = 2, the mode matrix will be a block diagonal matrix.
        shift : list or None
            (slow) Value of a coordinate offset, allows to return the mode matrix for a fiber with the center shifted with regard
            to the center of the observation window. Defaults to None.
        angle : float or None
            (slow) Angle in radians, allows to rotate the mode matrix with an arbitrary angle.
            Note that the rotation is applied BEFORE the transverse shift. Defaults to None.

        Returns
        -------
        M : numpy.ndarray
            The matrix representing the basis of the propagating modes.
        """
        assert self.profiles
        if shift is not None:
            assert len(shift) == 2

        N = self.profiles[0].shape[0]
        M = np.zeros((N * npola, npola * self.number), dtype=np.complex128)
        angle = angle / np.pi * 180.0 if (angle is not None) else None

        for pol in range(npola):

            for ind, modeProfile in enumerate(self.profiles):

                if shift is None and angle is None:
                    M[pol * N : (pol + 1) * N, pol * self.number + ind] = (
                        modeProfile  # .reshape(1,self._npoints**2)
                    )
                else:
                    mode2D = modeProfile.reshape([self.indexProfile.npoints] * 2)

                    if angle is not None:
                        mode2D = scipy_rotate(
                            mode2D.real, angle, reshape=False
                        ) + complex(0, 1) * scipy_rotate(
                            mode2D.imag, angle, reshape=False
                        )

                    if shift is not None:
                        mode2D = scipy_shift(input=mode2D.real, shift=shift) + complex(
                            0, 1
                        ) * scipy_shift(input=mode2D.imag, shift=shift)

                    M[pol * N : (pol + 1) * N, pol * self.number + ind] = (
                        mode2D.flatten()
                    )

        self.modeMatrix = M

        return M

    def sort(self, fn=None):
        """
        Sorts the modes based on the provided function `fn`.
        If none provided, sort by the values of `self.betas` in descending order.

        This method rearranges the elements of `self.betas` and other associated lists
        (such as `self.u`, `self.w`, `self.m`, `self.l`, `self.profiles`, `self.modesList`, and `self.data`)
        based on the sorted order of `self.betas` in descending order.

        Parameters
        ----------
        fn : function or None, optional
            The function used to sort the modes. Default is None.

        Returns
        -------
        None

        """

        class mode:
            m: int
            l: int
            beta: float
            profile: np.array
            index: int

            def __init__(self, index, m, l, beta, profile):
                self.m = m
                self.l = l
                self.beta = beta
                self.profile = profile
                self.index = index

        class modes_iterable:
            def __init__(self, modes):
                self.modes = modes
                self.idx = 0
                self.M0 = modes.getModeMatrix()

            def __iter__(self):
                return self

            def __next__(self):
                if self.idx < self.modes.number:
                    self.idx += 1
                    return mode(
                        self.idx - 1,
                        self.modes.m[self.idx - 1],
                        self.modes.l[self.idx - 1],
                        self.modes.betas[self.idx - 1],
                        self.M0[:, self.idx - 1],
                    )

                else:
                    raise StopIteration

        if fn is not None:
            it = modes_iterable(self)

            idx = [m.index for m in sorted(it, key=fn)]
        else:
            idx = np.flip(np.argsort(self.betas), axis=0)

        self.betas = [self.betas[i] for i in idx]
        if self.u:
            self.u = [self.u[i] for i in idx]
        if self.w:
            self.w = [self.w[i] for i in idx]
        if self.m:
            self.m = [self.m[i] for i in idx]
        if self.l:
            self.l = [self.l[i] for i in idx]
        if self.profiles:
            self.profiles = [self.profiles[i] for i in idx]
        if self.modesList:
            self.modesList = [self.modesList[i] for i in idx]
        if self.data:
            self.data = [self.data[i] for i in idx]

    def getNearDegenerate(self, tol=1e-2):
        """
        Find the groups of near degenerate modes with a given tolerence.
        Optionnaly sort the modes (not implemented).
        """
        copy_betas = {i: b for i, b in enumerate(self.betas)}
        groups = []

        while not (len(copy_betas) == 0):
            next_ind = np.min(list(copy_betas.keys()))
            beta0 = copy_betas.pop(next_ind)
            current_group = [next_ind]
            to_remove = []
            for ind in copy_betas.keys():
                if np.abs(copy_betas[ind] - beta0) <= tol:
                    to_remove.append(ind)
                    current_group.append(ind)
            [copy_betas.pop(x) for x in to_remove]
            groups.append(current_group)

        return groups

    def getNearDegenerateMask(self, tol=1e-2):
        """
        Generates a mask for near-degenerate modes.

        Parameters:
        - tol (float): Tolerance value for determining near-degeneracy.

        Returns:
        - mask_near_degenerate (ndarray): Mask for near-degenerate modes.

        Description:
        This method generates a mask for near-degenerate modes based on a given tolerance value.
        It first identifies groups of near-degenerate modes using the `getNearDegenerate` method.
        Then, it creates a square matrix of size `nmodes x nmodes`, where `nmodes` is the number of modes.
        The elements of the matrix are initialized to zero.
        For each group of near-degenerate modes, the corresponding submatrix in the mask is set to one.
        The resulting mask is returned as a complex-valued ndarray.

        Example usage:
        ```
        mask = getNearDegenerateMask(tol=1e-5)
        ```
        """
        degenerate_groups = self.getNearDegenerate(tol=tol)
        nmodes = self.number
        mask_near_degenerate = np.zeros([nmodes, nmodes], dtype=complex)

        for g in degenerate_groups:
            for combination in itertools.product(g, g):
                print(combination)
                mask_near_degenerate[combination] = 1
        return mask_near_degenerate

    def getEvolutionOperator(self, npola=1, curvature=None):
        """
        Returns the evolution operator B of the fiber.
        The diagonal of the evolution operator correspond to the propagation constants.
        The off-diagonal terms account for the mode coupling.
        The transmission matrix of the fiber reads exp(iBL) with L the propagation distance.
        For a straight fiber, B is a diagonal matrix.

        One can add the effect of curvature to a system solved for a straight fiber.
        It returns then the evolution operator in the basis of the straight fiber modes.
        The calculation is different from directly solving the system for a bent fiber [#]_ [#]_.



        Parameters
        ----------
        npola : int, optional
            Number of polarizations. Default is 1.
        curvature : float or None, optional
            The curvature of the fiber. Default is None.

        Returns
        -------
        B : numpy.ndarray
            The propagation operator.


        See Also
        --------
        getPropagationMatrix()

        References
        ----------

        .. _PEP 484:
           https://www.python.org/dev/peps/pep-0484/

        .. [#]  M. Plöschner, T. Tyc and T. Čižmár, "Seeing through chaos in multimode fibres"
                Nature Photonics, vol. 9,
                pp. 529–535, 2015.
                https://doi.org/10.1038/nphoton.2015.112

        .. [#]  S. M. Popoff, "Numerical Estimation of Multimode Fiber Modes and Propagation Constants: Part 2, Bent Fibers"
                http://wavefrontshaping.net/index.php/component/content/article/68-community/tutorials/multimode-fibers/149-multimode-fiber-modes-part-2

        """
        betas_vec = self.betas * npola
        B = np.diag(betas_vec).astype(np.complex128)

        # check if cuvature is a list or array of length 2 or None
        if hasattr(curvature, "__len__") and len(curvature) == 2:
            if 0 in curvature:
                logger.error("curvature = 0 not allowed!")
                raise (ValueError("curvature = 0 not allowed!"))
        elif curvature == None:
            pass
        elif isinstance(curvature, float) or isinstance(curvature, int):
            # if only one value for curvature, use the curvatrue for the X axis and add curvature = None for the Y axis
            curvature = [curvature, None]
        else:
            logger.error("Wrong type of data for curvature.")
            raise (ValueError("Wrong type of data for curvature."))

        if curvature is not None:
            assert self.wl
            assert self.indexProfile
            try:
                assert self.curvature == None
            except:
                logger.error(
                    "Adding curvature to the propagation operator requires the system to be solved for a straight fiber!"
                )
                return

            if self.modeMatrix is None:
                self.getModeMatrix(npola=npola)
            M = self.getModeMatrix()
            x = np.diag(self.indexProfile.X.flatten())
            Gamma_x = M.transpose().conjugate().dot(x).dot(M)
            y = np.diag(self.indexProfile.Y.flatten())
            Gamma_y = M.transpose().conjugate().dot(y).dot(M)
            k0 = 2 * np.pi / self.wl
            n_min = np.min(self.indexProfile.n)
            if curvature[0]:
                B = B - n_min * k0 * 1.0 / curvature[0] * Gamma_x
            if curvature[1]:
                B = B - n_min * k0 * 1.0 / curvature[1] * Gamma_y

        return B

    def getCurvedModes(self, curvature, npola=1):
        """
        Calculates the curved modes of the fiber.

        Parameters
        ----------
        curvature : float
            The curvature of the fiber.
        npola : int, optional
            Number of polarizations. Default is 1.

        Returns
        -------
        tuple
            A tuple containing the eigenvalues (new_betas) and the transposed modes (new_modes).
        """

        assert self.wl
        assert self.indexProfile
        try:
            assert self.curvature == None
        except:
            logger.error(
                "Adding curvature to the propagation operator requires the system to be first solved for a straight fiber!"
            )
            return
        if self.modeMatrix is None:
            self.getModeMatrix(npola=npola)
        M = self.getModeMatrix()

        B = self.getEvolutionOperator(npola=npola, curvature=curvature)

        new_betas, U = np.linalg.eig(B)

        new_modes = U.transpose().conjugate().dot(M.transpose().conjugate())

        # sort the modes
        new_modes = np.stack(
            [
                m
                for _, m in sorted(
                    zip(new_betas, new_modes),
                    key=lambda pair: pair[0].real,
                    reverse=True,
                )
            ]
        )
        new_betas = sorted(new_betas, key=lambda val: val.real, reverse=True)

        return new_betas, new_modes.transpose()

    def getPropagationMatrix(self, distance, npola=1, curvature=None):
        """
        Returns the transmission matrix T for a given fiber length in the basis of the fiber modes.
        Note that while you can set two polarizations, the modes profiles are obtained under a scalar apporoximation.
        For a straight fiber, T is a diagonal matrix.

        One can add the effect of curvature to a system solved for a straight fiber.
        It returns then the evolution operator in the basis of the straight fiber modes.
        The calculation is different from directly solving the system for a bent fiber [#]_.

        Parameters
        ----------
        distance : float
            Size of the fiber segment (in microns).
        npola : int
            Number of polarizations considered. For npola = 2, the mode matrix will be a block diagonal matrix.
        curvature : float
            Curvature of the fiber segment (in microns).

        Returns
        -------
        T : numpy.ndarray
            The transmission matrix of the fiber.

        See Also
        --------
        getEvolutionOperator

        Notes
        -----
        .. [#]  M. Plöschner, T. Tyc and T. Čižmár, "Seeing through chaos in multimode fibres"
            Nature Photonics, vol. 9, pp. 529–535, 2015.

        """
        B = self.getEvolutionOperator(npola, curvature)

        return expm(complex(0, 1) * B * distance)

    def save(self, filename: str, save_indes_profile=False):
        """
        Save the object to a file using pickle.

        Parameters
        ----------
        filename : str
            The name of the file to save the object to.
        save_indes_profile : bool, optional
            If True, the index profile will be saved. Default is False.

        Returns
        -------
        None

        Examples
        --------
        >>> modes = Modes()
        >>> ...
        >>> modes.save("modes.pkl")
        """
        dict_to_save = self.__dict__.copy()
        if not save_indes_profile:
            dict_to_save.pop("indexProfile")
        else:
            dict_to_save["indexProfile"] = self.indexProfile.to_dict()
        # dict_to_save.pop("indexProfile")
        with open(filename, "wb") as f:
            pickle.dump(dict_to_save, f)

    def load(self, filename: str):
        """
        Load the object from a file using pickle.

        Parameters
        ----------
        filename : str
            The name of the file to load the object from.

        Returns
        -------
        None

        Examples
        --------
        >>> modes = Modes()
        >>> modes.load("modes.pkl")
        """
        with open(filename, "rb") as f:
            data = pickle.load(f)

        if data.get("indexProfile"):
            from .index_profile import IndexProfile

            self.indexProfile = IndexProfile(npoints=0, areaSize=0)
            self.indexProfile.load(data["indexProfile"])
        self.__dict__.update(data)

    @classmethod
    def fromFile(cls, filename: str):
        """
        Load the object from a file using pickle.

        Parameters
        ----------
        filename : str
            The name of the file to load the object from.

        Returns
        -------
        Modes
            The object loaded from the file.

        Examples
        --------
        >>> modes = Modes.fromFile("modes.pkl")
        """
        obj = cls()
        obj.load(filename)
        return obj
