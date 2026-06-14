"""Small tensor helpers for NumPy CPU and Torch CUDA UVMS RL paths."""

from __future__ import annotations

from typing import Any

import numpy as np


def torch_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def torch_cuda_tensor_from_ptr(ptr: int, shape: tuple[int, ...], owner: Any):
    """Wrap a CUDA float32 pointer as a Torch tensor without copying."""

    import cupy as cp
    import torch

    if ptr == 0:
        raise ValueError("cannot wrap a null CUDA pointer")
    size = int(np.prod(shape))
    memory = cp.cuda.UnownedMemory(int(ptr), size * np.dtype(np.float32).itemsize, owner)
    pointer = cp.cuda.MemoryPointer(memory, 0)
    cupy_array = cp.ndarray(shape, dtype=cp.float32, memptr=pointer)
    return torch.utils.dlpack.from_dlpack(cupy_array)


def is_torch_tensor(value: Any) -> bool:
    try:
        import torch

        return torch.is_tensor(value)
    except Exception:
        return False


def as_numpy(value: Any) -> np.ndarray:
    if is_torch_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)
