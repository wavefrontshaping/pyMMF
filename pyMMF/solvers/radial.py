'''
Solver for axisymmetric index profile defined by a radial function,
e.g. graded index fibers.
Solve the 1D problem using the finite difference recursive scheme for 
Riccati's equations.
'''

import numpy as np
from scipy.interpolate import interp1d
from numba import jit, double

from ..modes import Modes
from ..logger import get_logger
logger = get_logger(__name__)

MIN_RADIUS_BC_DEFAULT = 1.5
CHANGE_BC_RADIUS_STEP_DEFAULT = 0.9
N_BETA_COARSE_DEFAULT = 1e3

# choice for degenerate subspaces
EXP_PHASE_FUNCS = [lambda x: np.exp(1j*x), lambda x: np.exp(-1j*x)]
SIN_PHASE_FUNCS = [np.sin, np.cos]

class PrecisionError(Exception):
    pass

def get_field_slow(m, dh, rr, nr, beta):
    def Q(r):
        return 1/r
    
    def P(r,n):
        return k0**2*n**2-beta**2-m**2/r**2
    
    def iter_g(gn,r,n):
        return (-1./(1+dh*Q(r)/2)*(dh**2*P(r,n)-2+(1-dh*Q(r)/2)/(1+dh*gn))-1)/dh
    
    # Boundary conditions at r~0
    if m == 0:
        g = [0.]
        f = [1.]
    else:
        g = [0., (1-dh**2*P(r[1],nr[1]))/dh]
        f = [0., 1.]
    i0 = len(f)
    for rn, nn in zip(rr[i0:],nr[i0:]):
        f.append(f[-1]*(1+dh*g[-1]))
        g.append(iter_g(g[-1],rn,nn))
    return np.array(f)

@jit('float64[:](int8,float32,float32[:],float32[:],float64,float64)', nopython=True, fastmath=True)
def _get_field_fast(m, dh, r, nr, beta,k0):
    def Q(r):
        return 1/r
    
    def P(r,n):
        return k0**2*n**2-beta**2-m**2/r**2
    
    def iter_g(gn,r,n):
        return (-1./(1+dh*Q(r)/2)*(dh**2*P(r,n)-2+(1-dh*Q(r)/2)/(1+dh*gn))-1)/dh
    
    # Boundary conditions at r~0
    if m == 0:
        g0 = [0.]
        f0 = [1.]
    else:
        g0 = [0., (1-dh**2*P(r[1],nr[1]))/dh]
        f0 = [0., 1.]
    bc = np.array([f0,g0], dtype = np.float64)
    
    bc_length = len(bc[0])
    f_vec = np.empty(shape = r.shape,
                 dtype=np.float64)
    f_vec[:bc_length] = bc[0]
    
    g_vec = np.empty(shape = r.shape,
                 dtype=np.float64)
    g_vec[:bc_length] = bc[1]
    
    
    for ir in range(bc_length,len(r)):
        f_vec[ir] = f_vec[ir-1]*(1+dh*g_vec[ir-1])
        g_vec[ir] = iter_g(g_vec[ir-1],r[ir],nr[ir])
    return f_vec

def get_field_fast(m, dh, r, nr, beta, k0):
    '''
    Get the field calulcated using the recursive scheme for the quadratic Ricatti
    equation as [1].
    
    Be careful that numba does not allow float128, which limits the precision
    and can lead to stagnation of the recursive iterations.
    
    .. [1] Lakshman S. Tamil, S. S. Mitra, R. Dutta, and J. M. T. Pereira, 
       "Finite difference solution for graded-index cylindrical dielectric waveguides: 
       a scalar wave approximation" Applied Optics, vol. 30,
       pp. 1113-1116, 1991.
    '''
#     func = jit('float64[:](int8,float32,float32[:],float32[:],float64)', nopython=True, fastmath=True)(get_field_)

    return _get_field_fast(np.int8(m),
                      np.float32(dh),
                      r.astype(np.float32), 
                      nr.astype(np.float32), 
                      np.float64(beta),
                       np.float64(k0)
                     )

def scan_betas(m,dh,r,nr,betas,k0):
    '''
    rough scan of betas values between beta_min and beta_max
    returns the sign of the value of the function for the
    farthest radial point for each beta value.
    (sign changes indicate the presence of an eigenvector)
    '''
    return [np.sign(get_field_fast(m,dh,r,nr,beta,k0)[-1]) for beta in betas]

def binary_search(func, min_val, max_val, sign, prev, tol = 1e-3):
    
    mid_val = (np.longdouble(min_val)+np.longdouble(max_val))/2
    res = func(mid_val)
    
    if min_val == max_val:
        logger.error('Stagnation due to floating point precision')
        raise PrecisionError

    if np.abs(res) < tol:
        return mid_val
    
    if sign*res>0:
        return binary_search(func, mid_val, max_val, sign, res, tol)
    else:
        return binary_search(func, min_val, mid_val, sign, res, tol)
    
def solve_radial(
    indexProfile,
    wl,
    **options
):
    degenerate_mode = options.get('degenerate_mode','sin')
    phi_funcs = EXP_PHASE_FUNCS if degenerate_mode == 'exp' else SIN_PHASE_FUNCS
    min_radius_bc = options.get('min_radius_bc',MIN_RADIUS_BC_DEFAULT)
    change_bc_radius_step = options.get('change_bc_radius_step',CHANGE_BC_RADIUS_STEP_DEFAULT)
    N_beta_coarse = options.get('N_beta_coarse',N_BETA_COARSE_DEFAULT)
    r_max = options.get('r_max',indexProfile.areaSize)
    dh = options.get('dh',indexProfile.areaSize/indexProfile.npoints)
    k0 = 2.*np.pi/wl

    n_func = indexProfile.radialFunc
    radius = indexProfile.a

    r_vec = np.arange(0,r_max,dh)
    nr = [n_func(r) for r in r_vec]

    beta_min = k0*np.min(nr)
    beta_max =  k0*np.max(nr)
    betas = np.linspace(beta_min,beta_max,N_beta_coarse)
    tol = 1e-6

    modes = Modes()
    modes.wl = wl
    modes.indexProfile = indexProfile

    m=0


    r = r_vec.astype(np.float64)

    while True:

        #logger.info(f'Searching radial modes for m = {m}')
        if m == 0:
            r = r-dh/2

        nr = np.array([n_func(rn) for rn in r], dtype = np.float64)

        # rough scan of beta values to find the changes of the sign of 
        # the farthest radial point
        sign_f = scan_betas(m = m,
                            dh = dh,
                            r = r,
                            nr = nr,
                            betas = betas,
                            k0 = k0
                           )

        # finds where fN change sign
        zero_crossings = np.where(np.diff(sign_f))[0][::-1]
        logger.info(f'Found {len(zero_crossings)} radial mode(s) for m={m}')
        if len(zero_crossings) == 0:
            break

        for l,iz in enumerate(zero_crossings):

            logger.info(f'Searching propagation constant for |l| = {l+1}')
            # find the beta value that satisfies the best the boundary condition 
            r_max = np.max(r)

            while True:

                if r_max < min_radius_bc*radius:
                    logger.error(f'Error: Boundary condition could not be met for r_max < {min_radius_bc}a')
                    logger.error('Try lower your tolerence or change the resolution.')
                    raise Exception

                r_search = r[r<=r_max]
                n_search = nr[r<=r_max]

                try:
                    #'Searching for beta value that satisfies the zero condition at r={r_max/radius:.3f}a'
                    def func_fast(beta):
                        f = get_field_fast(m,dh,r_search,n_search,beta,k0)
                        return f[-1]/np.max(np.abs(f))


                    beta = binary_search(
                        func_fast, 
                        min_val = betas[iz], 
                        max_val = betas[iz+1], 
                        sign = sign_f[iz],
                        prev = None,
                        tol = tol
                    )
                    # get the discretized radial function
                    f_vec = get_field_fast(m,dh,r_search,n_search,beta,k0)
                    # get the radial function by interpolation
                    # assume f = 0 for r>r_max
                    f_interp = np.pad(f_vec, (0,len(r)-len(f_vec)),'constant')
                    f = interp1d(r, f_interp, kind='cubic')
                    break
                except (RecursionError,PrecisionError) as e:
                    logger.warning(e)
                    logger.warning('Boundary condition could not be met.')
                    r_max *= change_bc_radius_step
                    logger.warning(f'Retrying by changing r_max to {r_max/radius:.2f}a')

            # add mode
            if m == 0:
                modes.betas.append(beta)
                modes.m.append(m)
                modes.l.append(l+1)
                modes.number+=1
                modes.profiles.append(f(indexProfile.R).ravel())
                modes.profiles[-1] = modes.profiles[-1]/np.sqrt(np.sum(np.abs(modes.profiles[-1])**2))
                # is the mode a propagative one?
                modes.propag.append(True)
            else:
                for s, phi_func in zip([-1,1],phi_funcs):
                    modes.betas.append(beta)
                    modes.m.append(s*m if degenerate_mode == 'exp' else m)
                    modes.l.append(l+1)
                    modes.number+=1
                    modes.profiles.append(f(indexProfile.R).ravel()*phi_func(m*indexProfile.TH.ravel()))
                    modes.profiles[-1] = modes.profiles[-1]/np.sqrt(np.sum(np.abs(modes.profiles[-1])**2))
                    # is the mode a propagative one?
                    modes.propag.append(True)

        m = m+1
    # sort the modes
    modes.sort()  
    return modes