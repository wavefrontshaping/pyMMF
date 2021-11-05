#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Sebastien M. Popoff
"""

import numpy as np
from .functions import cart2pol

class IndexProfile():
    def __init__(self,npoints,areaSize, n_r = None, n_theta = None):
        '''
		Parameters
		----------
		
		npoints : int
			size in pixels of the area
			
		areaSize : float
			 size in um of the area (let it be larger that the core size!)
        '''
        self.npoints = npoints
        self.n = np.zeros([npoints]*2)
        self.areaSize = areaSize
        x = np.linspace(-areaSize/2,areaSize/2,npoints)#+1./(npoints-1)
        self.X,self.Y = np.meshgrid(x,x)
        self.TH, self.R = cart2pol(self.X, self.Y)
        self.dh = 1.*self.areaSize/(self.npoints-1.)
        self.radialFunc = None
        self.type =  None
	
    def initFromArray(self,n_array):
        assert(n_array.shape == self.n.shape)
        self.n = n_array
        self.NA = None
        self.radialFunc = None
        self.type = 'custom'
		
    def initFromRadialFunction(self, nr):
        self.radialFunc = nr
        self.n = np.fromiter((nr(i) for i in self.R.reshape([-1])), np.float32)
        
    def initParabolicGRIN(self,n1,a,NA):
        self.NA = NA
        self.a = a
        self.type = 'GRIN'
        n2 = np.sqrt(n1**2-NA**2)
        Delta = NA**2/(2.*n1**2)
		
        radialFunc = lambda r: np.sqrt(n1**2.*(1.-2.*(r/a)**2*Delta)) if r<a else n2
        
        self.initFromRadialFunction(radialFunc)
        
    def initStepIndex(self,n1,a,NA):
        self.NA = NA
        self.a = a
        self.type = 'SI'
        self.n1 = n1
        n2 = np.sqrt(n1**2-NA**2)
        #Delta = NA**2/(2.*n1**2)

        radialFunc = lambda r: n1 if r < a else n2

        self.initFromRadialFunction(radialFunc)

    def initStepIndexMultiCore(
            self, n1: float = 1.45, a: float = 1., delta: float = 0.039,
            dims: int = 4, layers: int = 1,
            NA: float = 0.4, core_pitch: float = 5,
            central_core: bool = True):
        n2 = n1 * (1 - delta)
        self.NA = NA
        self.a = a
        self.type = 'SIMC'
        self.n1 = n1
        core_pitch = int(core_pitch // self.dh)
        lgrid = np.arange(0 if central_core else 1, layers + 1)
        angles = [
            np.arange(0, 2 * np.pi, 2 * np.pi / dims / lnum) for lnum in lgrid]
        radiuses = lgrid * core_pitch
        cores_coords = [[
            [self.npoints // 2 + int(r * np.sin(t)),
             self.npoints // 2 + int(r * np.cos(t))]
            for t in _a] for r, _a in zip(radiuses, angles)]
        cores_coords = sum(cores_coords, [])
        self.n = np.ones_like(self.R) * n2
        for indxs in cores_coords:
            i, j = indxs
            self.n[np.sqrt((self.X - self.X[i, j]) ** 2 +
                   (self.Y - self.Y[i, j]) ** 2) < a] = n1
        self.n.flatten()

    def initStepIndexConcentric(
            self, n1: float = 1.45, a: float = 1., delta: float = 0.039,
            NA: float = 0.4, core_pitch: float = 5, layers: int = 1):
        n2 = n1 * (1 - delta)
        self.NA = NA
        self.a = a
        self.type = 'SIC'
        self.n1 = n1
        radiuses = np.arange(layers + 1) * core_pitch

        def radialFunc(r):
            for r0 in radiuses:
                if np.abs(r - r0) < a:
                    return n1
            return n2

        self.initFromRadialFunction(radialFunc)

    def initStepIndexMicrostructured(
            self, n1: float = 1.45, a: float = 1.,
            dims: int = 4, layers: int = 1,
            NA: float = 0.4, core_pitch: float = 5,
            cladding_radius: float = 25):
        delta = 1 - n1
        n2 = np.sqrt(n1**2-NA**2)
        self.initStepIndexMultiCore(
            1, a, delta, dims, layers, NA, core_pitch,
            central_core=0)
        self.type = 'MSF'
        self._crop_cladding(n2, cladding_radius)

    def _crop_cladding(self, n2, cladding_radius):
        self.n[self.R > cladding_radius] = n2
