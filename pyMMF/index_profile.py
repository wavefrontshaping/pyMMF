#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Sebastien M. Popoff
"""

import numpy as np
from matplotlib import pyplot as plt
from typing import List
from .functions import cart2pol
from typing import List, Callable


class IndexProfile:
    def __init__(
        self,
        npoints: int,
        areaSize: float,
    ):
        """
        Initialize the IndexProfile object.

        Args:
            npoints (int): The number of points in each dimension of the grid.
            areaSize (float): The size in um of the area (let it be larger that the core size!)
        """
        self.npoints: int = npoints
        self.n: np.ndarray = np.zeros([npoints] * 2)
        self.areaSize: float = areaSize
        x: np.ndarray = np.linspace(
            -areaSize / 2, areaSize / 2, npoints
        )  # +1./(npoints-1)
        self.X: np.ndarray
        self.Y: np.ndarray
        self.X, self.Y = np.meshgrid(x, x)
        self.TH: np.ndarray
        self.R: np.ndarray
        self.TH, self.R = cart2pol(self.X, self.Y)
        self.dh: float = 1.0 * self.areaSize / (self.npoints - 1.0)
        self.radialFunc: Callable[[float], float] = None
        self.type: str = None

    def initFromArray(self, n_array: np.ndarray) -> None:
        """
        Initializes the index profile from an array.

        Args:
            n_array (np.ndarray):
                2d array containing the index values.
                Each value correspond to a pixel with coordinates given by
                (self.X, self.Y) in the cartesian coordinate system,
                or (self.R, self.TH) in the polar coordinate system.

        Returns:
            None
        """
        assert n_array.shape == self.n.shape
        self.n = n_array
        self.NA = None
        self.radialFunc = None
        self.type = "custom"

    def initFromRadialFunction(self, nr: Callable[[float], float]) -> None:
        self.radialFunc = nr
        self.n = np.fromiter((nr(i) for i in self.R.reshape([-1])), np.float32)

    def initParabolicGRIN(self, n1: float, a: float, NA: float) -> None:
        self.NA = NA
        self.a = a
        self.type = "GRIN"
        n2 = np.sqrt(n1**2 - NA**2)
        Delta = NA**2 / (2.0 * n1**2)

        radialFunc = lambda r: (
            np.sqrt(n1**2.0 * (1.0 - 2.0 * (r / a) ** 2 * Delta)) if r < a else n2
        )

        self.initFromRadialFunction(radialFunc)

    def initStepIndex(self, n1: float, a: float, NA: float) -> None:
        self.NA = NA
        self.a = a
        self.type = "SI"
        self.n1 = n1
        n2 = np.sqrt(n1**2 - NA**2)
        # Delta = NA**2/(2.*n1**2)

        radialFunc = lambda r: n1 if r < a else n2

        self.initFromRadialFunction(radialFunc)

    def plot(self) -> None:
        """
        Plot the index profile.

        Returns:
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
