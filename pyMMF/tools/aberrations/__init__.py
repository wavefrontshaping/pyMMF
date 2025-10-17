try:
    import torch
    import torch.nn.functional as F
    from torch.nn import Module

    _TORCH_AVAILABLE = True
except Exception as _e:  # ImportError or other import-time errors
    torch = None  # type: ignore
    F = None  # type: ignore
    Module = object  # graceful fallback for type checking
    _TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = _e


def _require_torch():
    """
    Ensure torch is available before executing any operation in this submodule.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "This submodule requires PyTorch but it is not installed.\n\n"
            "Please install PyTorch following the official instructions for your OS, "
            "package manager, CUDA/ROCm version, and Python environment:\n"
            "  https://pytorch.org/get-started/locally/\n\n"
            "Tip: the page provides a command generator (e.g., pip/conda) tailored "
            "to your system."
        ) from _TORCH_IMPORT_ERROR


_require_torch()

from .models import AberrationModes, Aberration
