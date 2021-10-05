'''
Solver for parabolic graded index fibers using analytical expression of the mode profile and dispersion relation
under the WKB approximation.
'''

import time
import numpy as np
from scipy.special import genlaguerre
from joblib import Parallel, delayed

from ..modes import Modes
from ..logger import get_logger
logger = get_logger(__name__)


def solve_WKB(indexProfile, wl, **options):
    degenerate_mode = options.get('degenerate_mode','sin')
    n_jobs = options.get('n_jobs', -2)
    modes = findPropagationConstants(wl,indexProfile)
    modes = associateLPModeProfiles(modes,
                                    wl,
                                    indexProfile,
                                    degenerate_mode=degenerate_mode,
                                    n_jobs=n_jobs)
    return modes


def findPropagationConstants(wl, indexProfile, tol = 1e-9):
    '''
    Find the propagation constants of parabolic GRIN multimode fibers under the WKB approximation [1]_.
    This approximation leads to inaccurate results for groups of modes close to the cutoff, 
    hence is not suitable when a limited number of mode is considered.
    It is provided only for comparison. 
    
    Parameters
    ----------
    wl : float
    		wavelength in microns.
        
    indexProfile: IndexProfile object
        object that contains data about the transverse index profile.
        
    Returns
    -------
    modes : Modes object
        Object containing data about the modes. 
        Note that it does not fill the transverse profiles, only the data about the propagation constants 
        and the mode numbers.
    
    See Also
    --------
        associateLPModeProfiles()
    
    Notes
    -----

    Propagation constants under the WKB approximation:

    .. math:: \beta_{l,m} = \sqrt{k_o^2 n_1^2-2\alpha \left( |l|+2m+1\right)}

    .. math:: \alpha = k_o n_1/b

    with 

    .. math:: b = \frac{radius \times n_1}{NA}

    Mode profiles under the WKB approximation

    .. math:: \psi_{l,m}(r, \phi) = A e^{- \frac{\alpha r^2}{2}}(\alpha r^2)^{|m|/2} L_l^{|m|}(\alpha r^2)e^{im\phi}

    with

    .. math:: L_l^{|m|}
    
    the Laguerre polynomials
         
    .. [1]  K. Okamoto, "Fundamentals of optical waveguides" 
            Academic Press,
            2006
            
    '''
    NA = indexProfile.NA
    a = indexProfile.a
    k0 = 2.*np.pi/wl

    n1 = np.max(indexProfile.n)
    n2 = np.min(indexProfile.n)

    b = a*n1/NA
    
    alpha = k0*n1/b

    beta_min = k0*n2
    beta_max = k0*n1

    modes = Modes()

    l = 1
    while True:
        m = 0
        while True:
            beta_current = np.sqrt(k0**2*n1**2-2*alpha*(np.abs(m)+2*(l-1)+1))
            if beta_current < beta_min or beta_current > beta_max:
                break
            degeneracy = 1 if m == 0 else 2
            modes.betas.extend([beta_current]*degeneracy)
            modes.number += degeneracy
            modes.m.extend([m]*degeneracy)
            modes.l.extend([l]*degeneracy)
            m += 1
        if m == 0:
            # if we stopped with m = 0, there is no more modes
            break
        l += 1

    return modes


def calc_mode(modes, idx, degenerate_mode, R, TH, a, alpha):
    m = modes.m[idx]
    l = modes.l[idx]

    aR2 = alpha*R.ravel()**2

    phase = m * TH.ravel()
    psi = 0

    # Non-zero transverse component
    if degenerate_mode == 'sin':
        # two pi/2 rotated degenerate modes for m < 0
        psi = np.pi/2 if m[idx] < 0 else 0 
        phase_mult = np.cos(phase + psi)

    elif degenerate_mode == 'exp':
        # noticably faster than writing exp(1j*phase)
        phase_mult = np.cos(phase) + 1j * np.sin(phase)
    
    amplitude = np.exp(-aR2/2)*aR2**(np.abs(m)/2)*genlaguerre(l-1,np.abs(m))(aR2)

    Et = phase_mult * amplitude
    mode = Et.astype(np.complex64)
    mode /= np.sqrt(np.sum(np.abs(mode)**2))
    return mode

def associateLPModeProfiles(
    modes, 
    wl,
    indexProfile, 
    degenerate_mode='sin',
    n_jobs=-2
    ):
    '''
    Associate the linearly polarized mode profile to the corresponding constants.
    '''
    
    assert(not modes.profiles)
    assert(degenerate_mode in ['sin', 'exp'])
    R = indexProfile.R
    TH = indexProfile.TH
    a = indexProfile.a
    NA = indexProfile.NA
    n1 = np.max(indexProfile.n)

    b = a*n1/NA
    k0 = 2.*np.pi/wl
    alpha = k0*n1/b
    
    logger.info('Finding analytical LP mode profiles associated to the propagation constants.')
    t0 = time.time()

    # for exp modes, we have +m and -m for degenerate modes
    if degenerate_mode == 'exp':
        modes.m = \
            [m if not (m, l) in zip(modes.m[:idx], modes.l[:idx]) else -m for idx, (m,l) in enumerate(zip(modes.m, modes.l)) ]

    R[R<np.finfo(np.float32).eps] = np.finfo(np.float32).eps

    
    modes.profiles = Parallel(n_jobs = n_jobs)(
        delayed(calc_mode)(modes, idx, degenerate_mode, R, TH, a, alpha) for idx in range(modes.number)
        )
    modes.sort() 

    logger.info("Found %g LP mode profiles in %0.1f minutes." % (
        modes.number, (time.time() - t0) / 60))

    return modes