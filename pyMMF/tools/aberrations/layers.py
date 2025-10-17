# aberration_layers.py
# PyTorch is optional for the parent package. This submodule will:
# - Import PyTorch if available
# - Otherwise, raise a clear ImportError the moment anything here is used,
#   with instructions to install from https://pytorch.org/get-started/locally/

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.nn import Module


if TYPE_CHECKING:
    # For static type checkers only
    import torch as _torch
    from torch import Tensor as _Tensor

################################################################
###################### AUX FUNCTIONS ###########################
################################################################


def zernike_Z(j, X, Y):
    # see https://en.wikipedia.org/wiki/Zernike_polynomials
    if j == 0:
        Fv = torch.ones_like(X)
    elif j == 1:
        Fv = 2 * X
    elif j == 2:
        Fv = 2 * Y
    elif j == 3:
        # Oblique astigmatism
        Fv = 2.0 * (6.0**0.5) * X.mul(Y)
    elif j == 4:
        # Defocus
        Fv = (3.0**0.5) * (2.0 * (X**2 + Y**2) - 1)
    elif j == 5:
        # Vertical astigmatism
        Fv = (6.0**0.5) * (X**2 - Y**2)
    else:
        R = torch.sqrt(X**2 + Y**2)
        THETA = torch.atan2(Y, X)
        if j == 6:
            # Vertical trefoil
            Fv = (8.0**0.5) * (R**3) * torch.sin(3.0 * THETA)
        elif j == 7:
            # Vertical coma
            Fv = (8.0**0.5) * (3.0 * R**3 - 2.0 * R) * torch.sin(3.0 * THETA)
        elif j == 8:
            # Horizontal coma
            Fv = (8.0**0.5) * (3.0 * R**3 - 2.0 * R) * torch.cos(3.0 * THETA)
        elif j == 9:
            # Oblique trefoil
            Fv = (8.0**0.5) * (R**3) * torch.cos(3.0 * THETA)
        elif j == 10:
            # Oblique quadrafoil
            Fv = (10.0**0.5) * (R**4) * torch.sin(4.0 * THETA)
        elif j == 11:
            # Oblique secondary astigmatism
            Fv = (10.0**0.5) * (4.0 * R**4 - 3.0 * R**2) * torch.sin(2.0 * THETA)
        elif j == 12:
            # Primary spherical
            Fv = (5.0**0.5) * (6.0 * R**4 - 6.0 * R**2 + torch.ones_like(R))
        elif j == 13:
            # Vertical secondary astigmatism
            Fv = (10.0**0.5) * (4.0 * R**4 - 3.0 * R**2) * torch.cos(2.0 * THETA)
        elif j == 14:
            # Vertical quadrafoil
            Fv = (10.0**0.5) * (R**4) * torch.cos(4.0 * THETA)
        else:
            raise ValueError(f"Unsupported Zernike index j={j}")

    return Fv


#######################################################
#################### MODULES ##########################
#######################################################


class ComplexZernike(Module):
    """
    Layer that applies a complex Zernike polynomial to the phase of a batch
    of complex images (or matrices). Only one parameter, the strength of the
    polynomial, is learned. Initial value is 0.
    """

    def __init__(self, j: int):
        super().__init__()
        if j not in range(15):
            raise ValueError("j must be in [0, 14]")
        self.j = j
        self.alpha = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, input):
        # input: (..., H, W) complex tensor
        H, W = input.shape[-2], input.shape[-1]

        # Use linspace and explicit indexing for meshgrid.
        nx = torch.linspace(0.0, 2.0, steps=H, dtype=torch.float32, device=input.device)
        ny = torch.linspace(0.0, 2.0, steps=W, dtype=torch.float32, device=input.device)

        X0 = 1.0 + 1.0 / H
        Y0 = 1.0 + 1.0 / W
        X, Y = torch.meshgrid(nx, ny, indexing="ij")
        X = X - X0
        Y = Y - Y0

        # Ensure F is on same device/dtype; exponent needs complex dtype
        Fv = zernike_Z(self.j, X, Y).to(dtype=input.real.dtype, device=input.device)

        return input * torch.exp((1j * self.alpha.to(input.real.dtype)) * Fv)


class ComplexScaling(Module):
    """
    Global scaling for a stack of 2D complex images (or matrices).
    Only one parameter, the scaling factor, is learned. Initial value is 1.
    """

    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, input):
        # Convert complex (N,H,W) to real 2-channel (N,2,H,W) for grid ops
        x = torch.view_as_real(input).permute(0, 3, 1, 2)  # (N,2,H,W)

        # Build affine matrices with consistent dtype/device
        base = torch.tensor([1, 0.0, 0.0, 0.0, 1, 0.0], dtype=x.dtype, device=x.device)
        theta = ((1.0 + self.theta) * base).reshape(2, 3).expand(x.shape[0], 2, 3)

        grid = F.affine_grid(theta, size=x.size(), align_corners=True)
        y = F.grid_sample(x, grid, align_corners=True)

        return torch.view_as_complex(y.permute(0, 2, 3, 1).contiguous())


class ComplexDeformation(Module):
    """
    Global affine transformation (6 learnable params) for 2D complex images.
    """

    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    def forward(self, input):
        # Convert complex (N,H,W) to real 2-channel (N,2,H,W) for grid ops
        x = torch.view_as_real(input).permute(0, 3, 1, 2)  # (N,2,H,W)

        base = torch.tensor([1, 0.0, 0.0, 0.0, 1, 0.0], dtype=x.dtype, device=x.device)
        shift_scale = torch.tensor(
            [0, 0, 0.1, 0, 0, 0.1], dtype=x.dtype, device=x.device
        )

        # Compose a gentle affine around identity + learnable params
        theta_vec = (1.0 + self.theta) * base + self.theta * shift_scale
        theta = theta_vec.reshape(2, 3).expand(x.shape[0], 2, 3)

        grid = F.affine_grid(theta, size=x.size(), align_corners=True)
        y = F.grid_sample(x, grid, align_corners=True)

        return torch.view_as_complex(y.permute(0, 2, 3, 1).contiguous())
