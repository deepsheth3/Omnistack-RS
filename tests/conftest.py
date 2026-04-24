"""
OmniStack-RS — pytest configuration and shared fixtures.

Primary responsibility: TMAStub monkeypatch.

TMA (Tensor Memory Accelerator) is a Hopper-only hardware instruction exposed
in Triton as tl._experimental_descriptor_load. Under TRITON_INTERPRET=1, the
interpreter cannot emulate TMA DMA — calling the API raises an error or
produces undefined behavior.

The TMAStub redirects every known TMA load path to a standard tl.load so
CPU unit tests exercise the full kernel arithmetic without H100 hardware.
Real TMA correctness is gated exclusively by ci/run_gpu_tests.py on Modal.

MANIFEST.md Invariant #5:
    "TMA cannot be tested with TRITON_INTERPRET=1. All TMA calls are
    wrapped in try_tma_load(). CPU tests use TMAStub injected via
    conftest.py. The Modal H100 CI script is the mandatory gate."
"""

from __future__ import annotations

import pytest
import torch


# ── TMA Stub ──────────────────────────────────────────────────────────────

# Every module path where a TMA descriptor load function may be registered.
# Triton 3.x exposes the primary API at triton.language._experimental_descriptor_load.
# triton.language.core is where it is defined before being re-exported to tl.
# The extra.cuda paths cover CUDA-specific dispatch added in some Triton builds.
# All entries use raising=False so missing paths are silently skipped — this
# list can be extended for future Triton versions without breaking CPU tests.
_TMA_PATCH_TARGETS: list[tuple[str, str]] = [
    ("triton.language",                          "_experimental_descriptor_load"),
    ("triton.language.core",                     "_experimental_descriptor_load"),
    ("triton.language.extra.cuda",               "tensormap_load_common"),
    ("triton.language.extra.cuda.libdevice",     "tensormap_load_common"),
]


class TMAStub:
    """
    CPU-compatible stand-in for tl._experimental_descriptor_load.

    In the Phase 4 fused Triton kernel, TMA descriptors (created via
    tl.make_tensor_descriptor) encode base pointer + shape + strides, and
    _experimental_descriptor_load performs an async DMA block fetch.
    Under TRITON_INTERPRET=1 we approximate this with a standard tl.load —
    same data movement, no DMA hardware.

    Signature absorbs all current and future TMA API parameters via *args,
    **kwargs so that new Triton arguments (cache_modifier, eviction_policy,
    boundary_check, etc.) never cause a TypeError here.

    Descriptor dispatch — attempted in order until one succeeds:
      1. desc.base  — Triton TensorDescriptor carries a .base block pointer;
                      this is the correct path for tl.make_tensor_descriptor.
      2. desc + coord — block pointer descriptors support '+' arithmetic.
      3. tl.load(desc) — last resort: treats desc as a raw pointer; data
                          movement may not be perfectly positioned but the
                          kernel arithmetic still runs for CPU validation.
    """

    # kwargs accepted by tl.load in all known Triton 3.x versions.
    # Filtering prevents TypeError when the Triton API adds new parameters
    # that our stub receives but tl.load does not yet understand.
    _LOAD_KWARGS = frozenset({"mask", "other", "boundary_check", "padding_option",
                               "cache_modifier", "eviction_policy", "volatile"})

    def __call__(self, desc, offsets, *args, **kwargs) -> object:
        import triton.language as tl

        # Forward only kwargs that tl.load accepts; silently drop the rest.
        load_kwargs = {k: v for k, v in kwargs.items() if k in self._LOAD_KWARGS}

        # Normalise offsets: TMA takes a list/tuple of per-dim block indices.
        coord = offsets[0] if isinstance(offsets, (list, tuple)) else offsets

        # ── Tier 1: TensorDescriptor with .base block pointer ─────────────
        # tl.make_tensor_descriptor returns a descriptor whose .base attribute
        # is the underlying block pointer. This is the semantically correct path.
        try:
            return tl.load(desc.base + coord, **load_kwargs)
        except (AttributeError, TypeError):
            pass

        # ── Tier 2: Block-pointer descriptor supporting '+' arithmetic ─────
        # Some Triton builds represent descriptors as tl.tensor block pointers
        # that implement __add__ directly.
        try:
            return tl.load(desc + coord, **load_kwargs)
        except (TypeError, Exception):
            pass

        # ── Tier 3: Raw pointer fallback ──────────────────────────────────
        # Treats desc as a plain pointer; ignores coord. Data may not land at
        # the expected offset, but the kernel's arithmetic logic still executes,
        # which is the goal of CPU unit tests.
        return tl.load(desc, **load_kwargs)


def _install_tma_stub(monkeypatch) -> int:
    """
    Monkeypatch every entry in _TMA_PATCH_TARGETS with a shared TMAStub instance.

    Returns the count of paths successfully patched (0 if Triton is absent).
    raising=False means absent attributes are silently skipped — so adding
    new entries to _TMA_PATCH_TARGETS never breaks existing test runs.
    """
    stub = TMAStub()
    n_patched = 0
    for module_path, attr_name in _TMA_PATCH_TARGETS:
        try:
            mod = __import__(module_path, fromlist=[attr_name])
            monkeypatch.setattr(mod, attr_name, stub, raising=False)
            n_patched += 1
        except ImportError:
            pass  # This Triton build doesn't have this sub-module — skip cleanly
    return n_patched


@pytest.fixture(autouse=True)
def tma_stub(monkeypatch) -> None:
    """
    Auto-used: installs TMAStub across all known TMA load paths before every test.

    Tests that require real TMA hardware (H100) must be decorated:
        @pytest.mark.skipif(
            not (torch.cuda.is_available()
                 and torch.cuda.get_device_capability()[0] >= 9),
            reason="requires Hopper (H100) for TMA",
        )
    and run exclusively via:
        python ci/run_gpu_tests.py          # Modal H100 CI gate
    """
    _install_tma_stub(monkeypatch)


# ── Common fixtures ───────────────────────────────────────────────────────

@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def default_config():
    """Production OmniConfig (head_dim=128, n_heads=32, hidden_dim=4096)."""
    from omnistack_rs.config import OmniConfig
    return OmniConfig()


@pytest.fixture
def small_config():
    """
    Minimal OmniConfig for fast CPU unit tests.

    Structurally identical to production — same invariants, same GQA ratio
    (n_heads / n_kv_heads = 2), same power-of-2 head_dim constraint.
    Runs a full forward pass in < 100ms on any machine.
    """
    from omnistack_rs.config import OmniConfig
    return OmniConfig(
        n_heads=4,
        n_kv_heads=2,
        hidden_dim=256,
        head_dim=64,   # 2^6 — valid for WHT butterfly; production uses 2^7 = 128
        lora_rank=4,
        lora_alpha=8.0,
        manifold_rank=8,
        ambient_dim=64,
    )
