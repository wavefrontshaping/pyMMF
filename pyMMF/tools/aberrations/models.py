import torch
import numpy as np
from torch.nn import Module, Sequential, Identity, ZeroPad2d
from .layers import ComplexDeformation, ComplexZernike, ComplexScaling
import torch.fft as fft


def crop_center(input: torch.Tensor, size: int) -> torch.Tensor:
    x = input.shape[1]
    y = input.shape[2]
    start_x = x // 2 - (size // 2)
    start_y = y // 2 - (size // 2)
    return input[:, start_x : start_x + size, start_y : start_y + size, ...]


class AberrationModes(torch.nn.Module):
    """
    Model for input and output aberrations.
    Apply an `Aberration` model to the input and output mode basis.
    """

    def __init__(
        self,
        inpoints: int,
        onpoints: int,
        padding_coeff: float = 0.0,
        list_zernike_ft=list(range(3)),
        list_zernike_direct=list(range(3)),
        deformation: str = "single",
    ):
        super().__init__()

        if isinstance(list_zernike_ft[0], int):
            list_zernike_ft_in = list_zernike_ft
            list_zernike_ft_out = list_zernike_ft
        elif isinstance(list_zernike_ft[0], list):
            list_zernike_ft_in = list_zernike_ft[0]
            list_zernike_ft_out = list_zernike_ft[1]
        else:
            raise ValueError("Zernike coefficients can only have 1 or 2 dims")

        if isinstance(list_zernike_direct[0], int):
            list_zernike_direct_in = list_zernike_direct
            list_zernike_direct_out = list_zernike_direct
        elif isinstance(list_zernike_direct[0], list):
            list_zernike_direct_in = list_zernike_direct[0]
            list_zernike_direct_out = list_zernike_direct[1]
        else:
            raise ValueError("Zernike coefficients can only have 1 or 2 dims")

        self.abberation_output = Aberration(
            onpoints,
            list_zernike_ft=list_zernike_ft_out,
            list_zernike_direct=list_zernike_direct_out,
            padding_coeff=padding_coeff,
            deformation=deformation,
        )
        self.abberation_input = Aberration(
            inpoints,
            list_zernike_ft=list_zernike_ft_in,
            list_zernike_direct=list_zernike_direct_in,
            padding_coeff=padding_coeff,
            deformation=deformation,
        )
        self.inpoints = inpoints
        self.onpoints = onpoints

    def forward(self, input: torch.Tensor, output: torch.Tensor):
        output_modes = self.abberation_output(output)
        input_modes = self.abberation_input(input)
        return input_modes, output_modes


class Aberration(torch.nn.Module):
    """
    Model that applies aberrations (direct and Fourier plane) and a global scaling
    at the input dimension of a matrix.
    """

    def __init__(
        self,
        shape: int,
        list_zernike_ft,
        list_zernike_direct,
        padding_coeff: float = 0.0,
        deformation: str = "single",
        features=None,
    ):
        super().__init__()

        # Normalize inputs: allow "count" or explicit list/ndarray
        if type(list_zernike_direct) not in [list, np.ndarray]:
            list_zernike_direct = list(range(0, int(list_zernike_direct)))
        if type(list_zernike_ft) not in [list, np.ndarray]:
            list_zernike_ft = list(range(0, int(list_zernike_ft)))

        self.nxy = int(shape)

        # padding layer, to have a good FFT resolution (requires cropping after IFFT)
        padding = int(padding_coeff * self.nxy)
        self.pad = ZeroPad2d(padding)

        # deformation
        if deformation == "single":
            self.deformation = ComplexDeformation()
        elif deformation == "scaling":
            self.deformation = ComplexScaling()
        else:
            self.deformation = Identity()

        # Zernike layers (start from j+1 to skip piston if desired)
        self.zernike_ft = Sequential(
            *(ComplexZernike(j=j + 1) for j in list_zernike_ft)
        )
        self.zernike_direct = Sequential(
            *(ComplexZernike(j=j + 1) for j in list_zernike_direct)
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Expect complex tensor with square spatial dims: (N, H, W) or (N,H,W) complex
        assert (
            input.shape[-1] == input.shape[-2]
        ), "Input must be square in spatial dims"

        # padding
        x = self.pad(input)

        # global deformation / scaling
        x = self.deformation(x)

        # to Fourier domain
        x = fft.ifftshift(x, dim=(-2, -1))
        x = fft.fft2(x)
        x = fft.fftshift(x, dim=(-2, -1))

        # Zernike layers in the Fourier plane
        x = self.zernike_ft(x)

        # back to direct domain
        x = fft.ifftshift(x, dim=(-2, -1))
        x = fft.ifft2(x)
        x = fft.fftshift(x, dim=(-2, -1))

        # Zernike layers in the direct plane
        x = self.zernike_direct(x)

        # Crop at the center (because of padding)
        x = crop_center(x, self.nxy)

        return x
