'''
Solver for step index fiber using analytical expression of the mode profile 
and solving numerically the analytical dispersion equation.
'''

import time
import numpy as np
from scipy.optimize import root
from scipy.special import jv, kn

from ..modes import Modes
from ..logger import get_logger
logger = get_logger(__name__)

from joblib import Parallel, delayed

def solve_SI(indexProfile, wl, **options):
    degenerate_mode = options.get('degenerate_mode','sin')
    n_jobs = options.get('n_jobs', -2)
    modes = findPropagationConstants(wl,indexProfile)
    modes = associateLPModeProfiles(modes,indexProfile,
                                    degenerate_mode=degenerate_mode,
                                    n_jobs=n_jobs)
    return modes


def findPropagationConstants(wl,indexProfile, tol=1e-9):
    '''
    Find the propagation constants of a step index fiber by numerically finding the solution of the
    scalar dispersion relation [1]_ [2]_.
    

   
    
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
         
    .. [1]  K. Okamoto, "Fundamentals of optical waveguides" 
            Academic Press,
            2006
            
    .. [2]  S. M. Popoff, "Modes of step index multimode fibers" 
            http://wavefrontshaping.net/index.php/component/content/article/68-community/tutorials/multimode-fibers/118-modes-of-step-index-multimode-fibers
            
    '''
    lbda = wl
    NA = indexProfile.NA
    a = indexProfile.a
    n1 = indexProfile.n1
    
    logger.info('Finding the propagation constant of step index fiber by numerically solving the dispersion relation.')
    t0 = time.time()
    # Based on the dispersion relation,l eq. 3.74

    # The normalized frequency, cf formula 3.20 of the book
    v=(2*np.pi/lbda*NA*a)   
   
    roots = [0]
    m = 0
    modes = Modes()


    interval = np.arange(np.spacing(10),v-np.spacing(10),v*1e-4)
    while len(roots):
        
        def root_func(u):
            w=np.sqrt(v**2-u**2)
            return jv(m,u)/(u*jv(m-1,u))+kn(m,w)/(w*kn(m-1,w))
               
        guesses = np.argwhere(np.abs(np.diff(np.sign(root_func(interval)))))
        froot = lambda x0: root(root_func,x0,tol = tol)
        sols = map(froot, interval[guesses])
        roots = [s.x for s in sols if s.success]
      
        # remove solution outside the valid interval, round the solutions and remove duplicates
        roots = np.unique([np.round(r/tol)*tol for r in roots if (r > 0 and r<v)]).tolist()
        roots_num = len(roots)

        if roots_num:
            degeneracy = 1 if m == 0 else 2
            modes.betas.extend([np.sqrt((2*np.pi/lbda*n1)**2-(r/a)**2) for r in roots]*degeneracy)
            modes.u.extend(roots*degeneracy)
            modes.w.extend([np.sqrt(v**2-r**2) for r in roots]*degeneracy)
            modes.number += roots_num*degeneracy
            modes.m.extend([m]*roots_num*degeneracy)
            modes.l.extend([x+1 for x in range(roots_num)]*degeneracy)

        m += 1
    
    logger.info("Found %g modes in %0.2f seconds." % (modes.number,time.time()-t0))
    return modes


def calc_mode(modes, idx, degenerate_mode, R, a, TH,
              Rlessa, Rgreatera):
    m = modes.m[idx]
    l = modes.l[idx]
    u = modes.u[idx]
    w = modes.w[idx]

    degenerated = False
    if (m, l) in zip(modes.m[:idx], modes.l[:idx]):
        degenerated = True
        m = -m

    phase = m * TH
    psi = 0

    # Non-zero transverse component
    if degenerate_mode == 'sin':
        # two pi/2 rotated degenerate modes for m < 0
        psi = np.pi/2 if degenerated else 0 
        phase_mult = np.cos(phase + psi)

    elif degenerate_mode == 'exp':
        # noticably faster than writing exp(1j*phase)
        phase_mult = np.cos(phase) + 1j * np.sin(phase)

    Et = phase_mult * (jv(m, u/a*R)/jv(m, u)*Rlessa +
                       kn(m, w/a*R)/kn(m, w)*Rgreatera)
    mode = Et.ravel().astype(np.complex64)
    mode /= np.sqrt(np.sum(np.abs(mode)**2))
    return mode


def associateLPModeProfiles(modes, indexProfile, degenerate_mode='sin',
                            n_jobs=-2):
    '''
    Associate the linearly polarized mode profile to the corresponding constants found solving the analytical dispersion relation.
    see: "Weakly Guiding Fibers" by D. Golge in Applied Optics, 1971
    '''
    
    assert(not modes.profiles)
    assert(degenerate_mode in ['sin', 'exp'])
    R = indexProfile.R
    TH = indexProfile.TH
    a = indexProfile.a
    
    logger.info('Finding analytical LP mode profiles associated to the propagation constants.')
    t0 = time.time()

    # for exp modes, we have +m and -m for degenerate modes
    if degenerate_mode == 'exp':
        modes.m = \
            [m if not (m, l) in zip(modes.m[:idx], modes.l[:idx]) else -m for idx, (m,l) in enumerate(zip(modes.m, modes.l))]

    # Avoid division bt zero in the Bessel function
    R[R<np.finfo(np.float32).eps] = np.finfo(np.float32).eps
    Rlessa = (R <= a)
    Rgreatera = (R > a)
    
    modes.profiles = Parallel(n_jobs=n_jobs)(
        delayed(calc_mode)(modes, idx, degenerate_mode, R, a, TH,
                           Rlessa, Rgreatera) for idx in range(modes.number))
    logger.info("Found %g LP mode profiles in %0.1f minutes." % (
        modes.number, (time.time() - t0) / 60))

    return modes
