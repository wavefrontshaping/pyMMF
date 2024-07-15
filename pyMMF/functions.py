import numpy as np
from scipy.special import jv, kn
from .logger import get_logger

logger = get_logger(__name__)
from typing import Tuple
from colorsys import hls_to_rgb


logger = logging.getLogger(__name__)


def colorize(
    z,
    theme="dark",
    saturation=1.0,
    beta=1.4,
    transparent=False,
    alpha=1.0,
    max_threshold=1,
):
    r = np.abs(z)
    r /= max_threshold * np.max(np.abs(r))
    arg = np.angle(z)

    h = (arg + np.pi) / (2 * np.pi) + 0.5
    l = 1.0 / (1.0 + r**beta) if theme == "white" else 1.0 - 1.0 / (1.0 + r**beta)
    s = saturation

    c = np.vectorize(hls_to_rgb)(h, l, s)  # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0, 2)
    if transparent:
        a = 1.0 - np.sum(c**2, axis=-1) / 3
        alpha_channel = a[..., None] ** alpha
        return np.concatenate([c, alpha_channel], axis=-1)
    else:
        return c


def cart2pol(X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns the angular and radial matrices (polar coordinates) correspondng to the input cartesian matrices X and Y.

    Parameters
    ----------

    X : numpy array
                matrix corresponding to the first Cartesian coordinate

    Y :  numpy array
                matrix corresponding to the second Cartesian coordinate

    Returns
    -------

    TH : numpy array
                matrix corresponding to the theta coordinate

    R : numpy array
                matrix corresponding to the radial coordinate
    """

    TH = np.arctan2(Y, X)
    R = np.sqrt(X**2 + Y**2)
    return (TH, R)
