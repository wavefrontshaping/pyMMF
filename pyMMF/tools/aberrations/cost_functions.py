import torch


def normalize(A: torch.Tensor) -> torch.Tensor:
    b = torch.sqrt(torch.sum(A * A.conj(), dim=(-2, -1)))
    return A / b[:, None, None]


def norm_mode_to_norm_pix(TM_pix, modes_out, modes_in, N_out, N_in):
    """
    Maximizes total conversion energy.

    It is defined as the inverse of the ratio between the norm squared of the TM
    in the mode basis by the norm squared of the TM in the pixel basis.
    """
    # reshape the change of basis matrices
    modes_out = modes_out.reshape((-1, N_out**2))
    modes_in = modes_in.reshape((-1, N_in**2))
    # project the TM in the mode basis with the current aberration parameters
    TM_mode = (modes_out @ TM_pix) @ torch.transpose(modes_in.conj(), 0, 1)
    # use the ratio of energy between the projected matrix and the pixel basis
    energy_ratio = (torch.norm(TM_mode) / torch.norm(TM_pix)) ** 2
    # the cost function to minimize is the inverse of this quantity
    return 1.0 / energy_ratio


def energy_on_diagonal(
    T_pix, degenerate_mask, pt_modes_out, pt_modes_in, onpoints, inpoints
):
    """
    Maximizes energy in the degenerate blocks.

    It is defined as the inverse of the ratio between the norm squared of the
    degenerate part of the TM in the mode basis by the norm squared of the
    TM in the pixel basis.
    """
    # reshape the change of basis matrices
    pt_modes_out = pt_modes_out.reshape((-1, onpoints**2, 2))
    pt_modes_in = pt_modes_in.reshape((-1, inpoints**2, 2))
    # project the TM in the mode basis with the current aberration parameters
    T_mode = (pt_modes_out @ T_pix) @ torch.transpose(pt_modes_in.conj(), 0, 1)
    # use the ratio of energy on block diagonal
    energy_diagonal = (torch.norm(T_mode * degenerate_mask) / torch.norm(T_mode)) ** 2
    # the cost function to minimize is the inverse of this quantity
    return 1.0 / energy_diagonal
