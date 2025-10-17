from .models import AberrationModes
from .cost_functions import norm_mode_to_norm_pix

import torch
from torch.autograd import Variable
import numpy as np


def _normalize_modes(modes):
    norms = torch.sqrt(torch.sum(modes * modes.conj(), dim=(-2, -1)))
    return modes / norms[:, None, None]


class AberrationOptimization:

    def __init__(
        self,
        TM_pix: np.ndarray,
        modes_in: np.ndarray,
        modes_out: np.ndarray,
        list_zernike_ft: list[int] = list(range(9)),
        list_zernike_direct: list[int] = list(range(9)),
        deformation: str = "scaling",
        padding_coeff: float = 0.05,
        device: None | str = None,
        dtype: torch.dtype = torch.complex64,
        lr: float = 1e-2,
    ):

        if device is None:
            if torch.cuda.is_available():
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
                device = torch.device(f"cuda:0")
            else:
                print("No GPU available, running on CPU. Will be slow.")
                device = torch.device("cpu")
        else:
            device = torch.device(device)

        self.dtype = dtype

        # check that the modes are of the shape (nmodes, N^2)
        if len(modes_out.shape) != 2 or len(modes_in.shape) != 2:
            raise ValueError(
                "modes_out and modes_in must be 2D arrays of shape (nmodes, N^2)"
            )
        if modes_out.shape[1] != modes_in.shape[1]:
            raise ValueError(
                "modes_out and modes_in must have the same number of columns"
            )
        if (
            modes_out.shape[0] != TM_pix.shape[0]
            or modes_in.shape[0] != TM_pix.shape[1]
        ):
            raise ValueError(
                "modes_out and modes_in must have the same number of columns as TM_pix has rows/columns",
                f"modes_out.shape: {modes_out.shape}, modes_in.shape: {modes_in.shape}, TM_pix.shape: {TM_pix.shape}",
            )

        n_out = int(np.sqrt(modes_out.shape[0]))
        n_in = int(np.sqrt(modes_in.shape[0]))
        if n_out**2 != modes_out.shape[0] or n_in**2 != modes_in.shape[0]:
            raise ValueError(
                "modes_out and modes_in must have a number of columns that is a perfect square"
            )

        self.n_out = n_out
        self.n_in = n_in

        self.nmodes = modes_out.shape[1]

        # convert the TM and the modes to pytorch tensors
        self.TM_pix = Variable(torch.from_numpy(TM_pix), requires_grad=False).to(device)

        self.modes_out = Variable(
            torch.from_numpy(modes_out.transpose().reshape((-1, n_out, n_out))),
            requires_grad=False,
        ).to(device)
        self.modes_in = Variable(
            torch.from_numpy(modes_in.transpose().reshape((-1, n_in, n_in))),
            requires_grad=False,
        )

        # define the model for the aberrations
        self.model = AberrationModes(
            n_in,
            n_out,
            padding_coeff=padding_coeff,
            list_zernike_ft=list_zernike_ft,
            list_zernike_direct=list_zernike_direct,
            deformation=deformation,
        ).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.loss_fun = norm_mode_to_norm_pix

        self.TM_modes_before = modes_out.T.conj() @ TM_pix @ modes_in

    def run(self, num_epoch: int = 1000, verbose: bool = True):

        new_modes_in, new_modes_out = self.model(self.modes_in, self.modes_out)

        # compute the cost function
        cost = self.loss_fun(
            self.TM_pix,
            new_modes_out,
            new_modes_in,
            self.n_out,
            self.n_in,
        )

        best_cost = 1e10
        evol_cost = [cost.item()]
        evol_conversion = [1 / cost.item()]

        for epoch in range(num_epoch):
            self.optimizer.zero_grad()

            # apply the aberrations to the modes
            pt_modes_in_var = _normalize_modes(self.modes_in)
            pt_modes_out_var = _normalize_modes(self.modes_out)

            new_modes_in, new_modes_out = self.model(pt_modes_in_var, pt_modes_out_var)

            # compute the cost function
            cost = self.loss_fun(
                self.TM_pix,
                new_modes_out,
                new_modes_in,
                self.n_out,
                self.n_in,
            )

            if cost.item() < best_cost:
                best_cost = cost.item()
                self.pt_best_modes_in = new_modes_in.detach()
                self.pt_best_modes_out = new_modes_out.detach()

            cost.backward()
            self.optimizer.step()

            evol_cost.append(cost.item())
            evol_conversion.append(1 / cost.item())

            if verbose and (epoch % 10 == 0 or epoch == num_epoch - 1):
                print(
                    f"Epoch {epoch+1}/{num_epoch}, energy conservation: {1/cost.item()*100:.3f}%"
                )

        # store the final modes and TM in mode basis
        new_modes_in = (
            self.pt_best_modes_in.detach().cpu().reshape((-1, self.n_in**2)).numpy()
        )

        new_modes_out = (
            self.pt_best_modes_out.detach().cpu().reshape((-1, self.n_out**2)).numpy()
        )

        TM_modes_after = (
            new_modes_out @ self.TM_pix.detach().cpu().numpy()
        ) @ new_modes_in.conj().transpose()

        new_modes_in = new_modes_in.transpose().conj()
        new_modes_out = new_modes_out.transpose().conj()

        evol_data = {
            "cost": evol_cost,
            "conversion": evol_conversion,
        }

        return (new_modes_in, new_modes_out, TM_modes_after, evol_data)
