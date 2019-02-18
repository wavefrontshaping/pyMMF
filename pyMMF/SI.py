#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 12:02:57 2019

@author: Sébastien M. Popoff
"""

from scipy.special import jv,kv,iv
import matplotlib.pyplot as plt

from scipy.signal import find_peaks_cwt

#try:
#    from peakutils import indexes 
#    find_peaks = lambda x: indexes(x,thres=0.02/max(Function), min_dist=10)
#except ImportError:
#    from scipy.signal import find_peaks_cwt
#    find_peaks = lambda x: find_peaks_cwt(x, np.arange(0.1,3,0.1))

#import logging
import time
#from scipy.linalg import expm
import numpy as np
from .modes import Modes
#from .core import logger
from .logger import get_logger, handleException 

logger = get_logger(__name__)


def findPropagationConstants(wl,indexProfile):
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
    
    # u and w are the normalized transverse wave numbers (3.16a and 3.16b)
    u=np.array(np.linspace(np.spacing(1),v-np.spacing(10),2e4))

    w=np.sqrt(v**2-u**2)
    
    peaks = [0]
    m=0
    modes = Modes()
    
    

    while(len(peaks)>0):
        
        # Function that is 1/abs(dispersion relation), based on the dispersion relation,l eq. 3.74 of "Fundamentals of optical wavegudies" by H. Okamoto
        # the number and positions of the maximum gives the modes of the fiber
#        if m == 0:
#            Function = 1/np.abs(jv(0,u)/(u*jv(1,u))-kv(0,w)/(w*kv(1,w)))
#        print(u)
        
#        else:
#        Function = (u*jv(m-1,u))+kv(m,w)/(w*kv(m-1,w))
        Function = 1./(np.spacing(1)+np.abs(jv(m,u)/(u*jv(m-1,u))+kv(m,w)/(w*kv(m-1,w))))
  
#        peaks = peakutils.indexes(Function, thres=0.02/max(Function), min_dist=10)

        peaks = find_peaks_cwt(Function, np.arange(0.1,3,0.1))
        
#        plt.figure()
#        plt.plot(u,Function)
##        plt.hold(True)
#        plt.plot(u[peaks],Function[peaks],'ko')
        
        
        if(len(peaks) > 0):
           
            degeneracy = 1 if m == 0 else 2
            
            modes.betas = np.concatenate((modes.betas, [np.sqrt((2*np.pi/lbda*n1)**2-(u[p]/a)**2) for p in peaks]*degeneracy))
            modes.u = np.concatenate((modes.u,[u[p] for p in peaks]*degeneracy))
            modes.w = np.concatenate((modes.w,[np.sqrt(v**2-u[p]**2) for p in peaks]*degeneracy))
            modes.number += len(peaks)*degeneracy
            modes.m.extend([m]*len(peaks)*degeneracy)
            modes.l.extend([x+1 for x in range(len(peaks))]*degeneracy)
#            modes.l.extend(np.arange(len(peaks))+1)
            for l in range(len(peaks)*degeneracy):
                modes.modesList.append(str(m)+','+str(l+1))
            
        
        m+=1
    
    logger.info("Found %g modes is %0.2f seconds." % (modes.number,time.time()-t0))
#    print(modes.modesList)
    return modes


