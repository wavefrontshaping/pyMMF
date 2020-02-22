#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 12:02:57 2019

@author: Sébastien M. Popoff
"""

import numpy as np
from scipy.special import jv, kn
from scipy.ndimage.interpolation import geometric_transform
import logging


logger = logging.getLogger(__name__)

def cart2pol(X, Y):
    '''
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
    '''
    TH = np.arctan2(Y, X)
    R = np.sqrt(X**2 + Y**2)
    return (TH, R)   



def associateLPModeProfiles(modes, indexProfile, degenerate_mode = 'sin'):
    '''
    Associate the linearly polarized mode profile to the corresponding constants found solving the analytical dispersion relation.
    see: "Weakly Guiding Fibers" by D. Golge in Applied Optics, 1971
    '''
    
    assert(not modes.profiles)
    assert(degenerate_mode in ['sin','exp'])
    R = indexProfile.R
    TH = indexProfile.TH
    a = indexProfile.a
    
    logger.info('Finding analytical LP mode profiles associated to the propagation constants.')
    
    # Avoid division bt zero in the Bessel function
    R[R<np.finfo(np.float32).eps] = np.finfo(np.float32).eps
  
    for idx in range(modes.number):
        m = modes.m[idx]
        l = modes.l[idx]
        u = modes.u[idx]
        w = modes.w[idx]
        

        
        psi = 0
        # Non-zero transverse component
        if degenerate_mode == 'sin':
            # two pi/2 rotated degenerate modes for m > 0
            if  (m,l) in zip(modes.m[:idx],modes.l[:idx]):
                psi = np.pi/2
            Et = ( jv(m,u/a*R)/jv(m,u)*np.cos(m*TH+psi)*(R <= a)+ \
                   kn(m,w/a*R)/kn(m,w)*np.cos(m*TH+psi)*(R > a))
                
        elif degenerate_mode == 'exp':
            if  (m,l) in zip(modes.m[:idx],modes.l[:idx]):
                modes.m[idx] = -m
                m = modes.m[idx]
            Et = ( jv(m,u/a*R)/jv(m,u)*np.exp(1j*m*TH)*(R <= a)+ \
                   kn(m,w/a*R)/kn(m,w)*np.exp(1j*m*TH)*(R > a))
                
      
        
        

        modes.profiles.append(Et.ravel().astype(np.complex64))
        modes.profiles[-1] = modes.profiles[-1]/np.sqrt(np.sum(np.abs(modes.profiles[-1])**2))
        
    return modes
