'''
Solver based on finite difference solution of the eigenvalue problen of the Helmholtz scalar equation.
'''
import numpy as np
import time
import scipy.sparse as sparse
from scipy.sparse.linalg import eigs

from ..modes import Modes
from ..logger import get_logger
logger = get_logger(__name__)

def solve_eig(
    indexProfile,
    wl,
    **options):
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
	    boundary : string, optional
		    boundary type, 'close' or 'periodic'
            EXPERIMENTAL.
            It should not make any difference for propagating modes.
        storeData: bool, optional
            Stores data in the propagationModeSolver object is set to True
            defaults to True
        curvature: float, optional
            Curvature of the fiber in meters
            defaults to None
        mode: string, optional
            detauls to 'default'
		    
        Returns
        -------
	    modes : Modes
		    Modes object containing all the mode information.
            
        See Also
        --------
            solve()
        '''
        curvature = options.get('curvature',None)
        nmodesMax= options.get('nmodesMax',6)
        boundary = options.get('boundary','close')
        propag_only = options.get('propag_only',True)
        poisson = options.get("poisson", 0.5)

        t0 = time.time()
        
        k0 = 2.*np.pi/wl
        npoints = indexProfile.npoints
        diags = []
        logger.info('Solving the spatial eigenvalue problem for mode finding.')   
        
        ## Construction of the operator 
        dh = indexProfile.dh
        diags.append(-4./dh**2+k0**2*indexProfile.n.flatten()**2)
        
        if boundary == 'periodic':
            logger.info('Use periodic boundary condition.')
            diags.append(([1./dh**2]*(npoints-1)+[0.])*(npoints-1)+[1./dh**2]*(npoints-1))
            diags.append(([1./dh**2]*(npoints-1)+[0.])*(npoints-1)+[1./dh**2]*(npoints-1))
            diags.append([1./dh**2]*npoints*(npoints-1))
            diags.append([1./dh**2]*npoints*(npoints-1))
            
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
            xi = 1.-(indexProfile.n.flatten()-1.)/indexProfile.n.flatten()*(1.-2.*poisson)
           
#            curv_mat = sparse.diags(1.-2*xi*self.indexProfile.X.flatten()/curvature, dtype = np.complex128)
            curv_inv_diag = 1.
            if curvature[0] is not None:
                curv_inv_diag+=2*xi*indexProfile.X.flatten()/curvature[0]
            if curvature[1] is not None:
                curv_inv_diag+=2*xi*indexProfile.Y.flatten()/curvature[1]   
            curv_mat = sparse.diags(1./curv_inv_diag, dtype = np.complex128)
#            curv_mat = sparse.diags(1./(1.+2*xi*self.indexProfile.X.flatten()/curvature), dtype = np.complex128)


#        logger.info('Note that boundary conditions should not matter too much for guided modes.')   
   

            
        
        H = sparse.diags(diags,offsets, dtype = np.complex128)

        if curvature:
            H = curv_mat.dot(H)
        
        beta_min = k0*np.min(indexProfile.n)
        beta_max =  k0*np.max(indexProfile.n)

        # Finds the eigenvalues of the operator with the greatest real part
        res = eigs(H,k=nmodesMax,which = 'LR')
                    
        modes = Modes()
        modes.wl = wl
        modes.indexProfile = indexProfile
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
