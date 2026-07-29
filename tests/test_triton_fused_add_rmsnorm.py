#!/usr/bin/env python3
"""Offline gate for ATOM's fused add+RMSNorm Triton kernel.

The kernel exists so a row-strided residual can be passed as a view instead of a
contiguous copy, which means the cases worth pinning are the strided ones: a
residual whose row stride differs from x's, and one whose rows are not 16-byte
aligned (the kernel may only promise ``tl.multiple_of`` when that actually
holds). Contiguous shapes are also checked against aiter's HIP kernel, which is
the production fallback, so the two paths cannot silently diverge.

No model or engine needed."""

import pytest
import torch

try:
    from aiter import rmsnorm2d_fwd_with_add

    from atom.model_ops.triton_fused_add_rmsnorm import can_fuse, fused_add_rmsnorm
except Exception as _e:  # pragma: no cover - bare-pytest import env
    pytest.skip(f"requires full atom import env: {_e}", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused add+RMSNorm is a GPU (Triton) kernel"
)

DEV = "cuda"
DTYPE = torch.bfloat16
EPS = 1e-6
DIM = 2880  # gpt-oss-120b hidden size


def _reference(x: torch.Tensor, res_in: torch.Tensor, weight: torch.Tensor):
    """fp32 reference for (out, res_out)."""
    summed = x.float() + res_in.float()
    rms = torch.rsqrt(summed.pow(2).mean(dim=-1, keepdim=True) + EPS)
    return summed * rms * weight.float(), summed


def _assert_matches(got: torch.Tensor, want: torch.Tensor, what: str):
    got_f, want_f = got.float(), want.float()
    cos = torch.nn.functional.cosine_similarity(
        got_f.flatten(), want_f.flatten(), dim=0
    ).item()
    denom = want_f.abs().max().clamp_min(1e-6)
    rel = ((got_f - want_f).abs().max() / denom).item()
    assert cos > 0.9999, f"{what}: cos={cos:.8f} rel={rel:.3e}"
    assert rel < 5e-2, f"{what}: rel={rel:.3e} cos={cos:.8f}"


def _run(x, res_in, weight):
    """Run the kernel into freshly allocated contiguous outputs."""
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    res_out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    fused_add_rmsnorm(out, x, res_in, res_out, weight, EPS)
    return out, res_out


def test_can_fuse_gates_on_width():
    assert can_fuse(DIM)
    assert not can_fuse(1 << 20)
    assert not can_fuse(0)


@pytest.mark.parametrize("n_rows", [1, 7, 1024])
def test_contiguous_matches_reference(n_rows):
    torch.manual_seed(0)
    x = torch.randn(n_rows, DIM, dtype=DTYPE, device=DEV)
    res_in = torch.randn(n_rows, DIM, dtype=DTYPE, device=DEV)
    weight = torch.randn(DIM, dtype=DTYPE, device=DEV)

    out, res_out = _run(x, res_in, weight)
    want_out, want_res = _reference(x, res_in, weight)
    _assert_matches(out, want_out, f"out n_rows={n_rows}")
    _assert_matches(res_out, want_res, f"res_out n_rows={n_rows}")


@pytest.mark.parametrize("n_rows", [1, 1024])
def test_contiguous_matches_aiter_hip_kernel(n_rows):
    """Parity with the fallback path that runs when the lever is off."""
    torch.manual_seed(1)
    x = torch.randn(n_rows, DIM, dtype=DTYPE, device=DEV)
    res_in = torch.randn(n_rows, DIM, dtype=DTYPE, device=DEV)
    weight = torch.randn(DIM, dtype=DTYPE, device=DEV)

    out, res_out = _run(x, res_in, weight)

    hip_out = torch.empty_like(x)
    hip_res = torch.empty_like(x)
    rmsnorm2d_fwd_with_add(hip_out, x, res_in, hip_res, weight, EPS)

    _assert_matches(out, hip_out, f"out vs HIP n_rows={n_rows}")
    _assert_matches(res_out, hip_res, f"res_out vs HIP n_rows={n_rows}")


@pytest.mark.parametrize("n_rows", [1, 33, 1024])
def test_residual_row_stride_independent_of_input(n_rows):
    """The reason this kernel exists: res_in strided differently from x.

    aiter's kernel indexes the residual with x's row stride, so this shape can
    only be served by copying the residual first.
    """
    torch.manual_seed(2)
    x = torch.randn(n_rows, DIM, dtype=DTYPE, device=DEV)
    # A (n_rows, DIM) view with row stride 2*DIM: rows land every other block.
    wide = torch.randn(n_rows, 2 * DIM, dtype=DTYPE, device=DEV)
    res_in = wide[:, :DIM]
    weight = torch.randn(DIM, dtype=DTYPE, device=DEV)

    assert res_in.stride(0) == 2 * DIM and res_in.stride(1) == 1
    assert res_in.stride(0) != x.stride(0)

    out, res_out = _run(x, res_in, weight)
    want_out, want_res = _reference(x, res_in, weight)
    _assert_matches(out, want_out, f"out strided-res n_rows={n_rows}")
    _assert_matches(res_out, want_res, f"res_out strided-res n_rows={n_rows}")


@pytest.mark.parametrize("n_rows", [1, 65])
def test_unaligned_residual_rows(n_rows):
    """Rows not on a 16-byte boundary must disable the alignment promise.

    A row stride of DIM+1 bf16 elements is 5762 bytes, not a multiple of 16, so
    successive rows are misaligned; a column offset also misaligns the base.
    """
    torch.manual_seed(3)
    x = torch.randn(n_rows, DIM, dtype=DTYPE, device=DEV)
    wide = torch.randn(n_rows, DIM + 1, dtype=DTYPE, device=DEV)
    res_in = wide[:, 1:]
    weight = torch.randn(DIM, dtype=DTYPE, device=DEV)

    assert res_in.stride(0) * res_in.element_size() % 16 != 0
    assert res_in.shape == (n_rows, DIM) and res_in.stride(1) == 1

    out, res_out = _run(x, res_in, weight)
    want_out, want_res = _reference(x, res_in, weight)
    _assert_matches(out, want_out, f"out unaligned-res n_rows={n_rows}")
    _assert_matches(res_out, want_res, f"res_out unaligned-res n_rows={n_rows}")


def test_strided_output_rows():
    """out and res_out may also carry their own row strides."""
    torch.manual_seed(4)
    n_rows = 16
    x = torch.randn(n_rows, DIM, dtype=DTYPE, device=DEV)
    res_in = torch.randn(n_rows, DIM, dtype=DTYPE, device=DEV)
    weight = torch.randn(DIM, dtype=DTYPE, device=DEV)

    out = torch.empty(n_rows, 2 * DIM, dtype=DTYPE, device=DEV)[:, :DIM]
    res_out = torch.empty(n_rows, 2 * DIM, dtype=DTYPE, device=DEV)[:, DIM:]
    fused_add_rmsnorm(out, x, res_in, res_out, weight, EPS)

    want_out, want_res = _reference(x, res_in, weight)
    _assert_matches(out, want_out, "out strided-out")
    _assert_matches(res_out, want_res, "res_out strided-out")
