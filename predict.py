# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path, BaseModel
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import pyMMF
from colorsys import hls_to_rgb
import tempfile
from typing import List

PROFILE_TYPE_OPTIONS = ["GRIN", "SI"]
CURVATURE_OPTIONS = ["Yes", "No"]
DEGENERATE_MODES_OPTIONS = ["cos", "exp"]

AREA_SIZE_COEFF = 1.0

SOLVER_N_POINTS_SEARCH = 2**8
SOLVER_N_POINTS_MODE = 2**7
SOLVER_R_MAX_COEFF = 1.8
SOLVER_BC_RADIUS_STEP = 0.95
SOLVER_N_BETA_COARSE = 1000


def colorize(
    z,
    theme="dark",
    saturation=1.0,
    beta=1.4,
    transparent=False,
    alpha=1.0,
    max_threshold=1,
):
    r = np.abs(z)
    r /= max_threshold * np.max(np.abs(r))
    arg = np.angle(z)

    h = (arg + np.pi) / (2 * np.pi) + 0.5
    l = 1.0 / (1.0 + r**beta) if theme == "white" else 1.0 - 1.0 / (1.0 + r**beta)
    s = saturation

    c = np.vectorize(hls_to_rgb)(h, l, s)  # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.swapaxes(0, 2)
    if transparent:
        a = 1.0 - np.sum(c**2, axis=-1) / 3
        alpha_channel = a[..., None] ** alpha
        return np.concatenate([c, alpha_channel], axis=-1)
    else:
        return c


def compute_modes(profile_type, diameter, NA, wl, n1, mode_repr):
    profile = pyMMF.IndexProfile(
        npoints=SOLVER_N_POINTS_MODE, areaSize=AREA_SIZE_COEFF * diameter
    )
    # build profile
    if profile_type == "GRIN":
        profile.initParabolicGRIN(n1=n1, a=diameter / 2, NA=NA)
    else:
        profile.initStepIndex(n1=n1, a=diameter / 2, NA=NA)

    # init solver
    solver = pyMMF.propagationModeSolver()
    solver.setIndexProfile(profile)
    solver.setWL(wl)

    r_max = SOLVER_R_MAX_COEFF * diameter
    k0 = 2.0 * np.pi / wl
    dh = diameter / SOLVER_N_POINTS_SEARCH

    modes = solver.solve(
        mode="radial",
        r_max=r_max,  # max radius to calculate (and first try for large radial boundary condition)
        dh=dh,  # radial resolution during the computation
        change_bc_radius_step=SOLVER_BC_RADIUS_STEP,  # change of the large radial boundary condition if fails
        N_beta_coarse=SOLVER_N_BETA_COARSE,  # number of steps of the initial coarse scan
        degenerate_mode=mode_repr,
        field_limit_tol=1e-4,
    )

    return modes


class Predictor(BasePredictor):
    def predict(
        self,
        profile_type: str = Input(
            description="Index profile (Graded index or step-index)",
            default="GRIN",
            choices=PROFILE_TYPE_OPTIONS,
        ),
        wl: float = Input(
            description="Wavelength (in nm).", ge=100, le=2000, default=1550
        ),
        core_diam: float = Input(
            description="Core diameter (in microns).", ge=10, le=80, default=50
        ),
        n_cladding: float = Input(
            description="Refractive index of the cladding.",
            ge=1.3,
            le=1.6,
            default=1.45,
        ),
        NA: float = Input(
            description="Core diameter (in microns).", ge=0.05, le=0.5, default=0.2
        ),
        mode_repr: str = Input(
            description="Mode representation, 'cos' for LP modes, 'exp' for OAM modes (if no curvature).",
            default="cos",
            choices=DEGENERATE_MODES_OPTIONS,
        ),
        is_curvature: str = Input(
            description="Curvature. Select 'No' for a straight fiber. Expect much longer computation time with curvature.",
            default="No",
            choices=CURVATURE_OPTIONS,
        ),
        curvature_x: float = Input(
            description="Curvature (in mm)", ge=1, le=200, default=10
        ),
    ) -> List[Path]:
        curvature = curvature_x * 1e3 if is_curvature == "Yes" else None

        outputs = []
        output_dir = Path(tempfile.mkdtemp())

        modes = compute_modes(
            profile_type, core_diam, NA, wl / 1e3, n_cladding, mode_repr
        )

        M0 = modes.getModeMatrix()
        print(curvature)
        if curvature is not None:
            betas, M0 = modes.getCurvedModes(npola=1, curvature=curvature)
        else:
            betas = modes.betas

        betas = modes.betas

        n_modes = modes.number

        fig_betas_path = output_dir.joinpath(f"betas.png")

        plt.figure(figsize=(6, 5), constrained_layout=True)
        plt.plot(np.sort(np.real(betas))[::-1], linewidth=2.0)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title(f"{n_modes} modes found", fontsize=16)
        plt.ylabel(r"Propagation constant $\beta$ (in $\mu$m$^{-1}$)", fontsize=16)
        plt.xlabel(r"Mode index", fontsize=16)
        plt.savefig(fig_betas_path)
        outputs.append(fig_betas_path)

        fig_first_modes_path = output_dir.joinpath(f"first_modes.png")
        plt.plot(figsize=(2, 2))

        for i in range(4):
            Mi = M0[..., i]
            profile = Mi.reshape([SOLVER_N_POINTS_MODE] * 2)
            # plt.figure(figsize = (4,4))
            plt.subplot(2, 2, i + 1)
            plt.imshow(colorize(profile, "white"))
            plt.axis("off")
            plt.title(f"Mode {i} (l={modes.l[i]}, m={modes.m[i]})", fontsize=16)
        plt.suptitle("First modes", fontsize=18)
        plt.savefig(fig_first_modes_path)
        outputs.append(fig_first_modes_path)

        fig_last_modes_path = output_dir.joinpath(f"last_modes.png")
        plt.plot(figsize=(2, 2))

        for ind, i in enumerate(range(-4, 0, 1)):
            # file_path = output_dir.joinpath(f"mode_{i}.png")
            Mi = M0[..., i]
            profile = Mi.reshape([SOLVER_N_POINTS_MODE] * 2)
            # plt.figure(figsize = (4,4))
            plt.subplot(2, 2, ind + 1)
            plt.imshow(colorize(profile, "white"))
            plt.axis("off")
            plt.title(
                f"Mode {n_modes+i+1} (l={modes.l[i]}, m={modes.m[i]})", fontsize=16
            )
        plt.suptitle("Last modes", fontsize=18)

        plt.savefig(fig_last_modes_path)
        outputs.append(fig_last_modes_path)

        # Save for Python
        python_mode_file = output_dir.joinpath(f"modes.npz")
        np.savez(
            python_mode_file,
            n_points=SOLVER_N_POINTS_MODE,
            n_modes=n_modes,
            profiles=M0,
            betas=betas,
        )
        outputs.append(python_mode_file)

        # Save for Matlab
        matlab_mode_file = output_dir.joinpath(f"modes.mat")
        matlab_dic = {
            "n_points": SOLVER_N_POINTS_MODE,
            "n_modes": n_modes,
            "profiles": M0,
            "betas": betas,
        }
        savemat(matlab_mode_file, matlab_dic)
        outputs.append(matlab_mode_file)

        return outputs
