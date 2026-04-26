"""
OmniStack-RS — TMA Descriptor Utilities (Phase 4)

Thin Python-side helpers for creating Triton TMA tensor descriptors.
On Hopper (H100), these descriptors are uploaded to constant memory as
128-byte hardware tensor-maps; the hardware DMA engine then performs the
actual GMEM→SRAM transfer without warp involvement.

On CPU (TRITON_INTERPRET=1), TMAStub in conftest.py intercepts every
`tl._experimental_descriptor_load` call and replaces it with `tl.load`.
These Python-side helpers are never called in that path.

MANIFEST Invariant #5:
    "TMA cannot be tested with TRITON_INTERPRET=1. All TMA calls are
     wrapped in try_tma_load(). CPU tests use TMAStub injected via
     conftest.py. The Modal H100 CI script is the mandatory gate."
"""

from __future__ import annotations

import torch

_HOPPER_MIN_CAPABILITY = 9   # sm_90 = H100


def is_hopper() -> bool:
    """True when running on a Hopper-class GPU (H100/GH200, sm_90+)."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= _HOPPER_MIN_CAPABILITY


def check_tma_alignment(ptr: torch.Tensor, element_size: int = 1) -> None:
    """
    TMA requires the base pointer to be 16-byte aligned.
    Raises ValueError if the tensor's data_ptr is not aligned.
    """
    if ptr.data_ptr() % 16 != 0:
        raise ValueError(
            f"TMA base pointer must be 16-byte aligned; "
            f"got data_ptr={ptr.data_ptr()} (offset {ptr.data_ptr() % 16}). "
            "Ensure tensors are allocated with torch.empty(..., device='cuda')."
        )


def make_2d_tma_descriptor(
    tensor: torch.Tensor,
    dim0: int,
    dim1: int,
    block_dim0: int,
    block_dim1: int,
):
    """
    Create a 2D TMA descriptor for an (dim0, dim1) contiguous tensor.

    Used for K/V nibble buffers laid out as (S, HEAD_DIM//2) uint8 and
    (S, QJL_DIM//8) uint8.  Block shape (block_dim0 × block_dim1) must
    satisfy block_dim0 * block_dim1 * element_size ≤ 128 KB (TMA limit).

    Args:
        tensor:     Source tensor (contiguous, CUDA, 16-byte aligned base).
        dim0:       Total rows (e.g. S = number of key positions).
        dim1:       Total columns (e.g. HEAD_DIM // 2).
        block_dim0: Tile rows per TMA load (e.g. BLOCK_S).
        block_dim1: Tile cols per TMA load (e.g. HEAD_DIM // 2).

    Returns:
        triton TensorDescriptor (opaque handle, passable to @triton.jit kernels).
    """
    try:
        import triton
        import triton.language as tl  # noqa: F401 — confirm import OK
    except ImportError as exc:
        raise RuntimeError("Triton is required for TMA descriptors") from exc

    assert tensor.is_contiguous(), "TMA source tensor must be contiguous"
    assert tensor.is_cuda, "TMA source tensor must be on CUDA"

    # Triton 3.x: tl.make_tensor_descriptor is called inside @triton.jit kernels.
    # From the Python side, we use triton's experimental Python API to pre-create
    # the descriptor (constant-memory upload) before the kernel launch.
    # This avoids recreating the descriptor on every Triton program launch.
    try:
        from triton.tools.experimental_descriptor import create_2d_tma_descriptor
        return create_2d_tma_descriptor(
            tensor.data_ptr(),
            dim0,
            dim1,
            block_dim0,
            block_dim1,
            tensor.element_size(),
        )
    except (ImportError, AttributeError):
        # Triton build without experimental_descriptor (e.g. older 3.x nightly).
        # Fall back to returning the tensor itself; the kernel will use regular loads.
        return tensor
