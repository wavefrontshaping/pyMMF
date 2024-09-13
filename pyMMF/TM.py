import numpy as np
import itertools
from scipy.stats import unitary_group
from .logger import get_logger


# https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html
def get_random_unitary_matrix(size):
    # Generate a random unitary matrix of the given size
    return unitary_group.rvs(size)


logger = get_logger(__name__)


class TransmissionMatrix(np.ndarray):
    def __new__(cls, input_array, npola=1, mode_repr="sin"):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.npola = npola
        # add the mode representation, `sin` or `exp`
        obj.mode_repr = mode_repr
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.npola = getattr(obj, "npola", 1)

    def rotatePolarization(self, angle):
        """
        Perform a polarization rotation operation on the transmission matrix based on the given angle.

        Args:
            angle (float): The angle in radians by which the polarization should be rotated.

        Returns:
            TransmissionMatrix: The updated transmission matrix after performing the polarization rotation operation.
        """
        if self.npola == 1:
            return self
        N = self.shape[0]
        Pola1 = self.view()[:, : N // 2]
        Pola2 = self.view()[:, N // 2 : N]

        self.view()[:, : N // 2] = Pola1 * np.cos(angle) + Pola2 * np.sin(angle)
        self.view()[:, N // 2 : N] = Pola2 * np.cos(angle) - Pola1 * np.sin(angle)
        return self

    def randomGroupCoupling(self, groups, strength=0.1):
        """
        Perform a random group coupling operation on the transmission matrix.

        Args:
            groups (list): A list of lists, where each sublist contains the indices of the modes in a group.
            strength (float): The strength of the random group coupling operation.

        Returns:
            TransmissionMatrix: The updated transmission matrix after performing the random group coupling operation.
        """
        # assert self.shape[0] == self.shape[1], "The transmission matrix must be square."
        N = self.shape[0] if self.npola == 1 else self.shape[0] // 2
        groups_size = list(itertools.chain(*groups))
        assert (
            len(groups_size) == N
        ), "The number of modes in the groups must be equal to the number of modes in the transmission matrix."

        for group in groups:
            indices = np.ix_(group, group)
            n_g = len(group)
            rnd_mat = (
                get_random_unitary_matrix(len(group))
                if n_g > 1
                else np.exp(1j * np.random.uniform(0, 2 * np.pi))
            )
            self[indices] = self[indices] * np.sqrt((1 - strength)) + rnd_mat * np.sqrt(
                strength
            )
        return self
