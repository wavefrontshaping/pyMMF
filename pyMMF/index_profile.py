#!/usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np
from matplotlib import pyplot as plt
from typing import List
from .functions import cart2pol
from typing import List, Callable, Union
import pickle
from scipy.interpolate import interp1d

NB_POINTS_TO_SAVE_RADIAL_FUNC = 10_000


class IndexProfile:
    """
    Class representing the refractive index profile of the fiber.

    Parameters
    ----------
    npoints : int
        The number of points in each dimension of the grid.
    areaSize : float
        The size in um of the area (let it be larger that the core size!)

    """

    npoints: int
    """
    Number of points in each dimension of the grid
    """

    n: np.ndarray
    """
    Index profile
    """

    areaSize: float
    """
    Size in um of the area
    """

    X: np.ndarray
    """
    X coordinate of the grid (2d array)
    """

    Y: np.ndarray
    """
    Y coordinate of the grid (2d array)
    """

    TH: np.ndarray
    """
    Azimuthal coordinate of the grid (2d array)
    """

    R: np.ndarray
    """
    Radial coordinate of the grid (2d array)
    """

    dh: float
    """
    Spatial resolution of the grid
    """

    radialFunc: Callable[[float], float]
    """
    Radial function (if initialized from a function)
    """

    type: str
    """
    Type of index profile (custom, GRIN, SI)
    """

    def __init__(
        self,
        npoints: int,
        areaSize: float,
    ):
        self.npoints = npoints
        self.n = np.zeros([npoints] * 2)
        self.areaSize = areaSize
        x = np.linspace(-areaSize / 2, areaSize / 2, npoints)  # +1./(npoints-1)
        self.X, self.Y = np.meshgrid(x, x)
        self.TH, self.R = cart2pol(self.X, self.Y)
        self.dh = 1.0 * self.areaSize / (self.npoints - 1.0)
        self.radialFunc: Callable[[float], float] = None
        self.type: str = None

    def initFromArray(self, n_array: np.ndarray) -> None:
        """
        Initializes the index profile from a numpy array.
        Use this function to define a custom index profile.

        Parameters
        ----------
        n_array : np.ndarray
            2d array containing the index values.
            Each value correspond to a pixel with coordinates given by
            (self.X, self.Y) in the cartesian coordinate system,
            or (self.R, self.TH) in the polar coordinate system.

        Returns
        -------
        None

        Example
        -------
        Square fiber index profile::

            import numpy as np
            import pyMMF
            n1 = 1.45
            n2 = 1.44
            npoints = 64
            core_size = 10
            areaSize = 20
            profile = pyMMF.IndexProfile(npoints=npoints, areaSize=areaSize)
            index_array = n2*np.ones((npoints,npoints))
            mask_core = (np.abs(profile.X) < core_size/2) & (np.abs(profile.Y) < core_size/2)
            index_array[mask_core] = n1
            profile.initFromArray(index_array)

        """
        assert n_array.shape == self.n.shape
        self.n = n_array
        self.NA = None
        self.radialFunc = None
        self.type = "custom"

    def initFromRadialFunction(self, nr: Callable[[float], float]) -> None:
        """
        Initializes the index profile from a radial function.
        Use this function to define a custom axisymmetric index profile.

        Parameters
        ----------
        nr : Callable[[float], float]
            A callable function that takes a float argument and returns the refractive index.

        Returns
        -------
        None

        Example
        -------
        Ring core fiber index profile::

            import numpy as np
            import pyMMF
            n1 = 1.445
            n2 = 1.45
            n3 = 1.44
            a = 5
            b = 10
            npoints = 256
            areaSize = 25
            profile = pyMMF.IndexProfile(npoints=npoints, areaSize=areaSize)
            def radialFunc(r):
                if r < a:
                    return n1
                elif r < b:
                    return n2
                else:
                    return n3
            profile.initFromRadialFunction(radialFunc)

        """
        self.radialFunc = nr
        print(self.radialFunc)
        self.n = np.fromiter((nr(i) for i in self.R.reshape([-1])), np.float32)
        self.n = self.n.reshape(self.R.shape)

    def initParabolicGRIN(
        self, n1: float, a: float, NA: float, alpha: float = 2.0
    ) -> None:
        r"""
        Initializes the refractive index profile for a parabolic GRIN fiber.

        \[
        \begin{eqnarray} 
            n(r) &=& \sqrt{n_1^2  \left[1 - 2  (r / a)^\alpha  \Delta n \right]} \quad  \forall \, r \leq a \\
            n(r) &=& n_2 \quad \forall \, r > a
        \end{eqnarray}
        \]

        with 

        \[
            \Delta n = \frac{NA^2}{2 n_1^2}
        \]

        Parameters
        ----------
        n1 : float
            The refractive index at the core center.
        a : float
            The core radius.
        NA : float
            The numerical aperture.
        alpha : float, optional
            The exponent of the parabolic profile. Default is 2.0.

        Returns
        -------
        None

        Example
        -------
        Parabolic GRIN fiber::
        
            import pyMMF
            n1 = 1.45; a = 10; NA = 0.2
            npoints = 64
            areaSize = 20
            profile = pyMMF.IndexProfile(npoints=npoints, areaSize=areaSize)
          
            profile.initParabolicGRIN(n1=n1, a=a, NA=NA)
            

        """
        self.NA = NA
        self.a = a
        self.type = "GRIN"
        n2 = np.sqrt(n1**2 - NA**2)
        Delta = NA**2 / (2.0 * n1**2)

        def radialFunc(r):
            return (
                np.sqrt(n1**2.0 * (1.0 - 2.0 * (r / a) ** alpha * Delta))
                if r < a
                else n2
            )

        print("*" * 80)
        print(radialFunc)

        self.initFromRadialFunction(radialFunc)

    def initStepIndex(self, n1: float, a: float, NA: float) -> None:
        """
        Initializes the step index profile.

        Parameters
        ----------
        n1 : float
            The refractive index inside the core region.
        a : float
            The core radius.
        NA : float
            The numerical aperture.

        Returns
        -------
        None

        Example
        -------
        ::

            import pyMMF
            n1 = 1.45; a = 10; NA = 0.2
            npoints = 256
            areaSize = 20
            profile = pyMMF.IndexProfile(npoints=npoints, areaSize=areaSize)
            profile.initStepIndex(n1=n1, a=a, NA=NA)

        """
        self.NA = NA
        self.a = a
        self.type = "SI"
        self.n1 = n1
        n2 = np.sqrt(n1**2 - NA**2)

        def radialFunc(r):
            return n1 if r < a else n2

        self.initFromRadialFunction(radialFunc)

    def getOptimalSolver(self, curvature: bool = False):
        """
        Returns the optimal solver based on the index profile and curvature flag.

        Parameters:
            curvature (bool): Flag indicating whether curvature is considered.

        Returns:
            str: The optimal solver based on the index profile and curvature flag.
                 Possible values are "SI" (for self.type == "SI" and not curvature),
                 "radial" (if self.indexProfile.radialFunc is not None), or "eig" (default).
        """
        if self.type == "SI" and not curvature:
            return "SI"
        elif self.radialFunc is not None:
            return "radial"
        else:
            return "eig"

    def plot(self) -> None:
        """
        Plot the index profile.

        Returns
        -------
        None
        """
        plt.figure()
        plt.imshow(
            self.n,
            extent=[self.X.min(), self.X.max(), self.Y.min(), self.Y.max()],
            origin="lower",
        )
        plt.colorbar()
        plt.xlabel("x (um)")
        plt.ylabel("y (um)")
        plt.title("Index profile")
        plt.show()

    def save(self, filename: str) -> None:
        """
        Save the index profile to a file.

        Parameters
        ----------
        filename : str
            The name of the file where to save the index profile.

        Returns
        -------
        None

        Example
        -------
        ::

            import pyMMF
            npoints = 256
            areaSize = 20
            profile = pyMMF.IndexProfile(npoints=npoints, areaSize=areaSize)
            ...
            profile.save("index_profile.pkl")

        """
        dict_to_save = self.to_dict()
        # Save the dictionary to a file
        with open(filename, "wb") as f:
            pickle.dump(dict_to_save, f)
        # np.savez(filename, **dict_to_save)

    def load(self, filename: [str, dict]) -> None:
        """
        Load the index profile from a file.

        Parameters
        ----------
        filename : str or dict
            The name of the file where the index profile is saved,
            or a dictionary containing the index profile data.

        Returns
        -------
        None
        """
        if isinstance(filename, dict):
            data_dict = filename
        elif isinstance(filename, str):
            data_dict = np.load(filename, allow_pickle=True)
        data = self.from_dict(data_dict)
        for key, value in data.items():
            setattr(self, key, value)

    def to_dict(self):
        data_dict = self.__dict__.copy()
        data_dict.pop("radialFunc")

        radial_func_r_vec = np.linspace(
            0, self.areaSize / 2, NB_POINTS_TO_SAVE_RADIAL_FUNC
        )
        data_dict["radial_func_r_vec"] = radial_func_r_vec
        data_dict["radial_func_discretized"] = np.fromiter(
            (self.radialFunc(i) for i in radial_func_r_vec),
            np.float32,
        )
        return data_dict

    def from_dict(self, data_dict):

        radialFunc = interp1d(
            data_dict["radial_func_r_vec"],
            data_dict["radial_func_discretized"],
            fill_value="extrapolate",
        )
        data_dict.pop("radial_func_r_vec")
        data_dict.pop("radial_func_discretized")
        data_dict["radialFunc"] = radialFunc
        return data_dict

    @classmethod
    def fromFile(cls, filename: str):
        """
        Load the index profile from a file.

        Parameters
        ----------
        filename : str
            The name of the file where the index profile is saved.

        Returns
        -------
        IndexProfile
            The index profile loaded from the file.

        Example
        -------
        ::

            import pyMMF
            filename = "index_profile.pkl"
            profile = pyMMF.IndexProfile.fromFile(filename)

        """
        profile = IndexProfile(npoints=0, areaSize=0)
        profile.load(filename)
        return profile
