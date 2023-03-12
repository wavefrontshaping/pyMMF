# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path, BaseModel
import numpy as np
import matplotlib.pyplot as plt
import pyMMF
from colorsys import hls_to_rgb
import tempfile
from typing import List

PROFILE_TYPE_OPTIONS = ['GRIN', 'SI']
# SOLVER_OPTIONS = ['Radial', 'Eig', 'Radial test']
SOLVER = 'Radial'

AREA_SIZE_COEFF = 1.2
CURVATURE = None

SOLVER_N_POINTS_SEARCH = 2**8
SOLVER_N_POINTS_MODE = 2**8
SOLVER_R_MAX_COEFF = 1.8
SOLVER_BC_RADIUS_STEP = 0.95
SOLVER_N_BETA_COARSE = 1000
SOLVER_DEGENERATE_MODE = 'exp'
SOLVER_MIN_RADIUS_BC = 1.5

def colorize(z, theme = 'dark', saturation = 1., beta = 1.4, transparent = False, alpha = 1., max_threshold = 1):
    r = np.abs(z)
    r /= max_threshold*np.max(np.abs(r))
    arg = np.angle(z) 

    h = (arg + np.pi)  / (2 * np.pi) + 0.5
    l = 1./(1. + r**beta) if theme == 'white' else 1.- 1./(1. + r**beta)
    s = saturation

    c = np.vectorize(hls_to_rgb) (h,l,s) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0,2) 
    if transparent:
        a = 1.-np.sum(c**2, axis = -1)/3
        alpha_channel = a[...,None]**alpha
        return np.concatenate([c,alpha_channel], axis = -1)
    else:
        return c
 

def compute_modes(profile_type, solver, diameter, NA, wl, n1):
    profile = pyMMF.IndexProfile(
        npoints = SOLVER_N_POINTS_MODE, 
        areaSize = AREA_SIZE_COEFF*diameter
    )
    # build profile
    if profile_type == 'GRIN':
        profile.initParabolicGRIN(n1=n1, a=diameter/2, NA=NA)
    else:
        profile.initStepIndex(n1=n1, a=diameter/2, NA=NA)
    if solver == 'Radial':
        solver_type = 'radial'
    elif solver == 'Radial test':
        solver_type = 'radial_test'
    elif solver == 'Eig':
        solver_type = 'eig'

    # init solver
    solver = pyMMF.propagationModeSolver()
    solver.setIndexProfile(profile)
    solver.setWL(wl)

    r_max = SOLVER_R_MAX_COEFF*diameter
    k0 = 2.*np.pi/wl
    dh = diameter/SOLVER_N_POINTS_SEARCH

    modes = solver.solve(mode=solver_type,
                        curvature = CURVATURE,
                        r_max = r_max, # max radius to calculate (and first try for large radial boundary condition)
                        dh = dh, # radial resolution during the computation
                        min_radius_bc = SOLVER_MIN_RADIUS_BC, # min large radial boundary condition
                        change_bc_radius_step = SOLVER_BC_RADIUS_STEP, #change of the large radial boundary condition if fails 
                        N_beta_coarse = SOLVER_N_BETA_COARSE, # number of steps of the initial coarse scan
                        degenerate_mode = SOLVER_DEGENERATE_MODE
                        )

    return modes

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")

    def predict(
        self,
        profile_type: str = Input(
            description="Index profile (Graded index or step-index)",
            default="GRIN",
            choices=PROFILE_TYPE_OPTIONS,
        ),
        # solver: str = Input(
        #     description="Solver",
        #     default="Radial",
        #     choices=SOLVER_OPTIONS,
        # ),
        wl: float = Input(
            description="Wavelength (in nm)", ge=100, le=2000, default=1550
        ),
        core_diam: float = Input(
            description="Core diameter (in microns)", ge=10, le=80, default=50
        ),
        n_cladding: float = Input(
            description="Core diameter (in microns)", ge=1.3, le=1.6, default=1.45
        ),
        NA: float = Input(
            description="Core diameter (in microns)", ge=0.05, le=.5, default=.2
        ),
    ) -> List[Path]:
        outputs = []

        output_dir = Path(tempfile.mkdtemp())
        

        modes = compute_modes(
            profile_type, 
            SOLVER,
            core_diam, 
            NA, 
            wl/1e3, 
            n_cladding
            )

        M0 = modes.getModeMatrix()
        n_modes = modes.number

        fig_betas_path = output_dir.joinpath(f"betas.png")

        
        plt.figure(figsize = (6,5), constrained_layout=True)
        plt.plot(
            np.sort(np.real(modes.betas))[::-1],
            linewidth=2.
            )
        plt.xticks(fontsize = 16)
        plt.yticks(fontsize = 16)
        plt.title(f'{n_modes} modes found' ,fontsize = 16)
        plt.ylabel(r'Propagation constant $\beta$ (in $\mu$m$^{-1}$)', fontsize = 16)
        plt.xlabel(r'Mode index', fontsize = 16)
        plt.savefig(fig_betas_path)
        outputs.append(fig_betas_path)        

        fig_first_modes_path = output_dir.joinpath(f"first_modes.png")
        plt.plot(figsize = (2, 2))

        for i in range(4):
            # file_path = output_dir.joinpath(f"mode_{i}.png")
            Mi = M0[...,i]
            profile = Mi.reshape([SOLVER_N_POINTS_MODE]*2)
            # plt.figure(figsize = (4,4))
            plt.subplot(2, 2, i+1)
            plt.imshow(colorize(profile,'white'))
            plt.axis('off')
            plt.title(
                f'Mode {i} (l={modes.l[i]}, m={modes.m[i]})',
                fontsize = 16
            )
        plt.suptitle('First modes')
        plt.savefig(fig_first_modes_path)
        outputs.append(fig_first_modes_path)


        fig_last_modes_path = output_dir.joinpath(f"last_modes.png")
        plt.plot(figsize = (2, 2))

        for ind, i in enumerate(range(-4,0,1)):
            # file_path = output_dir.joinpath(f"mode_{i}.png")
            Mi = M0[...,i]
            profile = Mi.reshape([SOLVER_N_POINTS_MODE]*2)
            # plt.figure(figsize = (4,4))
            plt.subplot(2, 2, ind+1)
            plt.imshow(colorize(profile,'white'))
            plt.axis('off')
            plt.title(
                f'Mode {n_modes+i+1} (l={modes.l[i]}, m={modes.m[i]})',
                fontsize = 16
            )
        plt.suptitle('Last modes')

        plt.savefig(fig_last_modes_path)
        outputs.append(fig_last_modes_path)
        
        mode_file  = output_dir.joinpath(f"modes.npz")
        np.savez(
            mode_file, 
            n_points = SOLVER_N_POINTS_MODE,
            n_modes = n_modes,
            profiles = M0, 
            betas = modes.betas)
        outputs.append(mode_file)
        
        return outputs

