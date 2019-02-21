#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 12:02:57 2019

@author: Sébastien M. Popoff
"""

from scipy.special import jv,kv,iv
import matplotlib.pyplot as plt

#from scipy.signal import find_peaks_cwt
#import peakutils

#try:
#    from peakutils import indexes 
#    find_peaks = lambda x: indexes(x,thres=0.02/max(Function), min_dist=10)
#except ImportError:
#    from scipy.signal import find_peaks_cwt
#    find_peaks = lambda x: find_peaks_cwt(x, np.arange(0.1,3,0.1))



import time
import numpy as np
from .modes import Modes
from .logger import get_logger
from scipy.optimize import root

logger = get_logger(__name__)

def _root_guesses(f,a,b,dx):
    x1 = a; f1 = f(a)
    x2 = a + dx; f2 = f(x2)
    roots = []
    while x1 < b:
        if f1*f2 < 0 and f1*f2 > -1.:
            roots.append(.5*(x1+x2))
        x1 = x2; f1 = f2
        x2 = x1 + dx; f2 = f(x2)
    return roots



def findPropagationConstants(wl,indexProfile, tol=1e-9):
    '''
    Find the propagation constants of a step index fiber by numerically finding the sollution of the
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
    m=0
    modes = Modes()
    
    
    
    

    while(len(roots)>0):
        
        def root_func(u):
            w=np.sqrt(v**2-u**2)
            return jv(m,u)/(u*jv(m-1,u))+kv(m,w)/(w*kv(m-1,w))
               
        guesses = _root_guesses(root_func,np.spacing(10),v-np.spacing(10),v*1e-4)
        froot = lambda x0: root(root_func,x0,tol = tol)
        sols = list(map(froot,guesses))
        roots = [s.x for s in sols if s.success]
      
        # remove solution outside the valid interval, round the solutions and remove duplicates
        roots = np.unique([np.round(r/tol)*tol for r in roots if (r > 0 and r<v)]).tolist()

        
        
        if(len(roots) > 0):
           
            degeneracy = 1 if m == 0 else 2
            modes.betas = np.concatenate((modes.betas, [np.sqrt((2*np.pi/lbda*n1)**2-(r/a)**2) for r in roots]*degeneracy))
            modes.u = np.concatenate((modes.u,roots*degeneracy))
            modes.w = np.concatenate((modes.w,[np.sqrt(v**2-r**2) for r in roots]*degeneracy))
            modes.number += len(roots)*degeneracy
            modes.m.extend([m]*len(roots)*degeneracy)
            modes.l.extend([x+1 for x in range(len(roots))]*degeneracy)
            for l in range(len(roots)*degeneracy):
                modes.modesList.append(str(m)+','+str(l+1))
            
        
        m+=1
    
    logger.info("Found %g modes is %0.2f seconds." % (modes.number,time.time()-t0))
    return modes


