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
import scipy.sparse as sparse
from scipy.sparse.linalg import eigs
#from scipy.linalg import expm
#from scipy.special import jv,kv,iv
#from scipy.ndimage.interpolation import shift as scipy_shift
#from scipy.ndimage.interpolation import rotate as scipy_rotate
import sys, time
from . import SI
#from .index_profile import IndexProfile 
from .modes import Modes
from .functions import associateLPModeProfiles
from .logger import get_logger, handleException 



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
        Pola1 = self.view()[:,:N/2]
        Pola2 = self.view()[:,N/2:N]
        
        self.view()[:,:N/2] = Pola1*np.cos(angle)+Pola2*np.sin(angle)
        self.view()[:,N/2:N] = Pola2*np.cos(angle)-Pola1*np.sin(angle)
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
        #self.u = []
        #self.w = []
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
        if hasattr(curvature, "__len__") and len(curvature) == 2:
            if 0 in curvature:
                logger.error('curvature = 0 not allowed!')
                raise(ValueError('curvature = 0 not allowed!'))
        elif curvature == None:
            pass
        elif isinstance(curvature, float) or isinstance(curvature, int):
            # if only one value for curvature, use the curvatrue for the X axis and add curvature = None for the Y axis
            curvature = [curvature,None]
        else:
            logger.error('Wrong type of data for curvature.')
            raise(ValueError('Wrong type of data for curvature.'))
            
        
            
        
        
        if self.indexProfile.type == 'SI' and (mode == 'SI'  or (mode == 'default' and not curvature)):
            if curvature is not None:
                logger.info('Semi-analytical solution of step-index fiber is not compatible with curvature.')
                return
            modes = SI.findPropagationConstants(self.wl,self.indexProfile)
            degenerate_mode = options.get('degenerate_mode','sin')
            modes = associateLPModeProfiles(modes,self.indexProfile,degenerate_mode=degenerate_mode)
            
            
        elif mode == 'default' or mode == 'eig':
            modes = self.solve_eig(curvature=curvature,**options)
        
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
        
    def solve_eig(self,curvature,nmodesMax=6,boundary = 'close', propag_only = True):
        '''
	    Find the first modes of a multimode fiber. The index profile has to be set.
        Returns a Modes structure containing the mode information.
	    
        Parameters
        ----------
	    nmodesMax : int 
		    Maximum number of modes the solver will try to find. 
            This value should be higher than the estimated maximum number of modes if one want to be sure 
            to find all the modes.
            defaults to 6
	    boundary : string
		    boundary type, 'close' or 'periodic'
            EXPERIMENTAL.
            It should not make any difference for propagating modes.
        storeData: bool
            Stores data in the propagationModeSolver object is set to True
            defaults to True
        curvature: float
            Curvature of the fiber in meters
            defaults to None
        mode: string
            detauls to 'default'
		    
        Returns
        -------
	    modes : Modes
		    Modes object containing all the mode information.
            
        See Also
        --------
            solve()
        '''

        
        t0 = time.time()
        
        
        k0 = 2.*np.pi/self.wl
        npoints = self.indexProfile.npoints
        diags = []
        
        logger.info('Solving the spatial eigenvalue problem for mode finding.')   
        
        ## Construction of the operator 
        dh = self.indexProfile.dh
        diags.append(-4./dh**2+k0**2*self.indexProfile.n.flatten()**2)
        
        if boundary == 'periodic':
            logger.info('Use periodic boundary condition.')
            diags.append(([1./dh**2]*(npoints-1)+[0.])*(npoints-1)+[1./dh**2]*(npoints-1))
            diags.append(([1./dh**2]*(npoints-1)+[0.])*(npoints-1)+[1./dh**2]*(npoints-1))
            diags.append([1./dh**2]*npoints*(npoints-1))
            diags.append([1./dh**2]*npoints*(npoints-1))
            
#            diags.append(([0]*(npoints-1)+[1./dh**2])*(npoints-1)+[0]*(npoints-1))
#            diags.append(([0]*(npoints-1)+[1./dh**2])*(npoints-1)+[0]*(npoints-1))
            diags.append(([1./dh**2]+[0]*(npoints-1))*(npoints-1)+[1./dh**2])
            diags.append(([1./dh**2]+[0]*(npoints-1))*(npoints-1)+[1./dh**2])
            
            
            diags.append([1./dh**2]*npoints)
            diags.append([1./dh**2]*npoints)
            
            offsets = [0,-1,1,-npoints,npoints,-npoints+1,npoints-1,-npoints*(npoints-1),npoints*(npoints-1)]
        elif boundary == 'close':
            logger.info('Use close boundary condition.')
            
            # x parts of the Laplacian
            diags.append(([1./dh**2]*(npoints-1)+[0.])*(npoints-1)+[1./dh**2]*(npoints-1))
            diags.append(([1./dh**2]*(npoints-1)+[0.])*(npoints-1)+[1./dh**2]*(npoints-1))
            # y parts of the Laplacian
            diags.append([1./dh**2]*npoints*(npoints-1))
            diags.append([1./dh**2]*npoints*(npoints-1))
            
            offsets = [0,-1,1,-npoints,npoints]
            
        if curvature is not None:
            # xi term, 
            # - the 1. term represent the geometrical effect
            # - the term in (1-2*poisson_coeff) represent the effect of compression/dilatation
            # see http://wavefrontshaping.net/index.php/68-community/tutorials/multimode-fibers/149-multimode-fiber-modes-part-2
            xi = 1.-(self.indexProfile.n.flatten()-1.)/self.indexProfile.n.flatten()*(1.-2.*self.poisson)
           
            
#            curv_mat = sparse.diags(1.-2*xi*self.indexProfile.X.flatten()/curvature, dtype = np.complex128)
            curv_inv_diag = 1.
            if curvature[0] is not None:
                curv_inv_diag+=2*xi*self.indexProfile.X.flatten()/curvature[0]
            if curvature[1] is not None:
                curv_inv_diag+=2*xi*self.indexProfile.Y.flatten()/curvature[1]   
            curv_mat = sparse.diags(1./curv_inv_diag, dtype = np.complex128)
#            curv_mat = sparse.diags(1./(1.+2*xi*self.indexProfile.X.flatten()/curvature), dtype = np.complex128)


#        logger.info('Note that boundary conditions should not matter too much for guided modes.')   
   

            
        
        H = sparse.diags(diags,offsets, dtype = np.complex128)

        if curvature:
            H = curv_mat.dot(H)
        
        self.H = H
        beta_min = k0*np.min(self.indexProfile.n)
        beta_max =  k0*np.max(self.indexProfile.n)

        # Finds the eigenvalues of the operator with the greatest real part
        res = eigs(H,k=nmodesMax,which = 'LR')
                    
        modes = Modes()
        modes.wl = self.wl
        modes.indexProfile = self.indexProfile
        # select only the propagating modes
        for i,betasq in enumerate(res[0]):
             if (betasq > beta_min**2 and betasq < beta_max**2) or not propag_only:
                modes.betas.append(np.sqrt(betasq))
                modes.number+=1
                modes.profiles.append(res[1][:,i])
                modes.profiles[-1] = modes.profiles[-1]/np.sqrt(np.sum(np.abs(modes.profiles[-1])**2))
                # is the mode a propagative one?
                modes.propag.append((betasq > beta_min**2 and betasq < beta_max**2))
                
        logger.info("Solver found %g modes is %0.2f seconds." % (modes.number,time.time()-t0))
        
        if (nmodesMax == modes.number):
            logger.warning('The solver reached the maximum number of modes set.')
            logger.warning('Some propagating modes may be missing.')
        


        return modes
    

    
    def saveData(self,outfile,saveArea = None):
        assert(self.modes)
        if not saveArea:
            pass
        betas = self.modes.betas
        mode_profiles = self.modes.profiles[1]
        index_profile = self.indexProfile.n
        X = self.indexProfile.X
        Y = self.indexProfile.X
        np.savez(outfile,
                 betas=betas,
                 mode_profiles=mode_profiles,
                 index_profile=index_profile,
                 X=X,
                 Y=Y)
        logger.info('Data saved to %s.' % outfile)
        


        
        

