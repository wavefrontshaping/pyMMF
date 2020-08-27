#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 12:02:57 2019

@author: Sébastien M. Popoff
"""

import numpy as np
from scipy.special import jv, kn
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



