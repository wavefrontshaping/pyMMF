	# -*- coding: utf-8 -*-
"""
Simple module to find numerically the propagation modes and their corresponding propagation constants
of multimode fibers of arbitrary index profiles.

Based on 
- http://wavefrontshaping.net/index.php/tutorials
and
- M. Plöschner, T. Tyc, and T. Čižmár, Nature Photonics Seeing through chaos in multimode fibres 9, 529–535 (2015).
https://doi.org/10.1038/nphoton.2015.112

written by Sebastien M. Popoff
"""

import numpy as np
import sys, time
from .modes import Modes
from .logger import get_logger, handleException 
from .solvers import solve_eig, solve_SI, solve_radial

class AssertionError(Exception):
    pass

logger = get_logger(__name__)

sys.excepthook = lambda excType, excValue, traceback: handleException(excType, excValue, traceback, logger=logger)
#%%

#https://docs.scipy.org/doc/numpy-1.13.0/user/basics.subclassing.html
class TransmissionMatrix(np.ndarray):

    def __new__(cls, input_array, npola=1):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.npola = npola
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.npola = getattr(obj, 'npola', 1)
        
    def polarization_rotation(self,angle):
        if self.npola == 1: return self
        N=self.shape[0]
        Pola1 = self.view()[:,:N//2]
        Pola2 = self.view()[:,N//2:N]
        
        self.view()[:,:N//2] = Pola1*np.cos(angle)+Pola2*np.sin(angle)
        self.view()[:,N//2:N] = Pola2*np.cos(angle)-Pola1*np.sin(angle)
        return self


        

def randomGroupCoupling(groups):
    '''
    Create a unitary matrix accounting for random mode coupling only into given groups of modes.
    '''
    size = np.max([np.max(g) for g in groups])+1
    H = np.zeros([size]*2,dtype=np.complex128)
    
    for g in groups:
        # generate random unitary matrix
        g_size = len(g)
        u,_,__ = np.linalg.svd(np.random.randn(g_size,g_size)+complex(0,1)*np.random.randn(g_size,g_size))
        H[np.ix_(g,g)] = u
        
    return H

def estimateNumModesGRIN(wl,a,NA,pola=1):
    '''
	Returns a rough estimation of the number of propagating modes of a GRIN fiber.
    See https://www.rp-photonics.com/v_number.html for more details.
	
	Parameters
	----------
	
	wl : float
		Wavelength (in microns)
	a :  float
		Radius of the fiber (in microns)
    NA : float
		Numerical aperture of the fiber
    pola : int (1 or 2)
        Number of polarizations
		
	Returns
	-------
	
	N : integer
		Estimation of the number of propagating modes
    '''
    k0 = 2.*np.pi/wl
    V = k0*a*NA
    return np.ceil(V**2/4.*pola/2.).astype(int)

def estimateNumModesSI(wl,a,NA,pola=1):
    '''
	Returns a rough estimation of the number of propagating modes of a step index fiber.
    See https://www.rp-photonics.com/v_number.html for more details.
	
    Parameters
    ----------
	wl : float
		 Wavelength (in microns)
	a :  float
		 Radius of the fiber (in microns)
    NA : float
		 Numerical aperture of the fiber
    pola : int (1 or 2)
        Number of polarizations
         
    Returns
    -------
	N : integer
		Estimation of the number of propagating modes
    '''
    k0 = 2.*np.pi/wl
    V = k0*a*NA
    return np.ceil(V**2/2.*pola/2.).astype(int)

   


#%%

class propagationModeSolver():
    
    def __init__(self):
        self.betas = []
        self.modes = None
        self.modesList = []
        self.number = 0
        self.m = []
        self.l = []
        self.indexProfile = None
        self.wl = None
        self.last_res = None
        self.poisson = 0.5
        
        logger.debug('Debug mode ON.')
        
    
        
    def setIndexProfile(self,indexProfile):
        self.indexProfile = indexProfile
        
    def setWL(self,wl):
        '''
        Set the wavelength (in microns).
        
        Parameters
        ----------
	    wl : float
             Wavelength in microns.
        '''
        self.wl = wl
        
    def setPoisson(self,poisson):
        '''
        Set the poisson coefficient. The default value is 0.5 (no effect of compression/dilatation)
        
        Parameters
        ----------
	    poisson : float
                  Poisson coefficient of the fiber material.
        '''
        self.poisson = poisson
        
    def solve(self, mode = 'default', curvature = None, storeData = True, **options):
        '''
        Find the propagation constants and mode profile of a multimode fiber.
        For an arbitrary index profile, it finds the solution of the eigenvalue problem of the scalar wave equation in a discretized space [1].
        
        Parameters
        ----------
        storeData: bool
            Stores data in the propagationModeSolver object is set to True
            defaults to True
        curvature: float
            Curvature of the fiber in meters
            defaults to None
        mode: string ('default','eig' or 'SI')
            solver to be used. 
            'eig' solves the eigenvalue problem in the discretized space.
            'SI' solves numerically the analytical dispersion relation and approximate modes to LP modes.
            'default' use the best appropriate solver.
            detauls to 'default'
        **options: dict
            specific options for the solver
		    
        Returns
        -------
        modes : Modes
		    Modes object containing all the mode information.
                     
        See Also
        --------
            solve_eig()
        
        '''
        assert(self.indexProfile)
        assert(self.wl)

        # check if cuvature is a list or array of length 2 or None
        if curvature == None:
            pass
        elif hasattr(curvature, "__len__") and len(curvature) == 2:
            if 0 in curvature:
                logger.error('curvature = 0 not allowed!')
                raise(ValueError('curvature = 0 not allowed!'))
        elif isinstance(curvature, float) or isinstance(curvature, int):
            # if only one value for curvature, use the curvatrue for the X axis and add curvature = None for the Y axis
            curvature = [curvature,None]
        else:
            logger.error('Wrong type of data for curvature.')
            raise(ValueError('Wrong type of data for curvature.'))
            
        if mode == 'default':
            mode = self.get_optimal_solver(curvature)   
        
        if mode == 'SI':
            if not (self.indexProfile.type == 'SI'):
                logger.error('SI solver only available for step-index profiles')
                raise AssertionError
            if curvature is not None:
                logger.error('Semi-analytical solution of step-index fiber is not compatible with curvature.')
                raise AssertionError
            modes = solve_SI(
                self.indexProfile,
                self.wl,
                **options
            )
        elif mode == 'radial':
            if self.indexProfile.radialFunc is None:
                logger.error('radial solver only available for axisymmetric profiles defined by a radial function')
                raise AssertionError
            modes = solve_radial(
                self.indexProfile,
                self.wl,
                **options
            )
            
            
        elif mode == 'eig':
            modes = solve_eig(
                indexProfile = self.indexProfile,
                wl = self.wl,
                curvature=curvature,
                poisson = self.poisson,
                **options
            )
        else:
            raise ValueError('Invalid mode')
            
        if storeData:
            self.modes = modes
            logger.debug('Mode data stored in memory.')
            
        modes.poison = self.poisson
        modes.indexProfile = self.indexProfile
        modes.wl = self.wl
        modes.curvature = curvature

        return modes
        
    def get_optimal_solver(self, curvature):
        if self.indexProfile.type == 'SI' and not curvature:
            logger.info('Selectinf step-index solver')
            return 'SI'
        elif self.indexProfile.radialFunc is not None:
            logger.info('Selectinf axisymmetric radial solver')
            return 'radial'
        else:
            return 'eig'
    

    
    def saveData(self,outfile,saveArea = None):
        assert(self.modes)
        if not saveArea:
            pass
        betas = self.modes.betas
        mode_profiles = self.modes.profiles[1]
        index_profile = self.indexProfile.n
        X = self.indexProfile.X
        Y = self.indexProfile.Y
        np.savez(outfile,
                 betas=betas,
                 mode_profiles=mode_profiles,
                 index_profile=index_profile,
                 X=X,
                 Y=Y)
        logger.info('Data saved to %s.' % outfile)
        


        
        

