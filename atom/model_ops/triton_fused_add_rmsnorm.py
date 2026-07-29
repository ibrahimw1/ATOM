"""Fused add + RMSNorm Triton kernel with independent residual row strides.

Based on aiter's ``_fused_add_rmsnorm_kernel``. aiter's version indexes res_in
and res_out with ``input_row_stride``, so a residual whose rows are strided
differently from x cannot be passed at all -- callers have to materialise a
contiguous copy first, which on gpt-oss-120b cost a full extra read+write per
layer per step. Here every tensor carries its own row stride, so a row-strided
residual view is passed as-is.

Vendored rather than patched into aiter so ATOM depends only on upstream aiter.

Two differences from aiter's kernel beyond the strides, both safe for an
inference-only path:

* No ``rsigma`` output. aiter stores the per-row normalisation factor for its
  backward pass; nothing in ATOM reads it, so the store and its per-call
  allocation are dropped.
* The weight is loaded once instead of once per row, which matters when one
  program walks many rows (prefill).

Single-tile only: one program handles a whole row, so ``n_cols`` must fit in one
tile. Callers gate on :func:`can_fuse` and fall back to the aiter HIP kernel for
anything wider.
"""

from functools import cache

import torch
import triton
import triton.language as tl

# A row is processed as one tile of next_power_of_2(n_cols) lanes. Past this the
# tile stops being a sensible register budget and the caller should fall back;
# aiter switches to its blocked path at the same width.
_MAX_TILE_COLS = 32768


@cache
def _num_cus(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def can_fuse(n_cols: int) -> bool:
    """Whether this kernel can serve a row of *n_cols* elements."""
    return 0 < n_cols <= _MAX_TILE_COLS


def _row_aligned(t: torch.Tensor) -> bool:
    """Whether *t*'s rows all start on a 16-byte boundary.

    ``tl.multiple_of`` is a promise to the compiler, not a request, so it may
    only be made when it actually holds. aiter's kernel can assert it
    unconditionally because it only ever sees one row stride; once res_in and
    res_out bring their own strides, an arbitrary residual view can break it.
    """
    esize = t.element_size()
    return t.data_ptr() % 16 == 0 and (t.stride(0) * esize) % 16 == 0


@triton.jit
def _fused_add_rmsnorm_kernel(
    input_ptr,
    output_ptr,
    res_in_ptr,
    res_out_ptr,
    g_ptr,
    input_row_stride,
    output_row_stride,
    res_in_row_stride,
    res_out_row_stride,
    n_rows,
    n_cols,
    epsilon,
    ALIGNED: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_PRGMS: tl.constexpr,
):
    """out = rmsnorm(x + res_in) * g, res_out = x + res_in.

    Each program walks rows in a persistent loop. One row is one tile, so
    n_cols <= BLOCK_SIZE.
    """
    row_start = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Shared across every row this program handles.
    g = tl.load(g_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    for row_idx in tl.range(row_start, n_rows, NUM_PRGMS, num_stages=2):
        input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
        res_in_ptrs = res_in_ptr + row_idx * res_in_row_stride + col_offsets
        res_out_ptrs = res_out_ptr + row_idx * res_out_row_stride + col_offsets
        output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets

        if ALIGNED:
            input_ptrs = tl.multiple_of(input_ptrs, (16,))
            res_in_ptrs = tl.multiple_of(res_in_ptrs, (16,))
            res_out_ptrs = tl.multiple_of(res_out_ptrs, (16,))
            output_ptrs = tl.multiple_of(output_ptrs, (16,))

        x = tl.load(input_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
        res_in = tl.load(res_in_ptrs, mask=mask, other=0.0, cache_modifier=".cg")
        x += res_in

        # The next layer consumes the pre-norm sum, so it is written out as well.
        tl.store(res_out_ptrs, x.to(res_out_ptr.dtype.element_ty), mask=mask)

        x = x.to(tl.float32)
        norm_factor = tl.math.rsqrt((tl.sum(x * x, axis=-1) / n_cols) + epsilon)
        tl.store(
            output_ptrs,
            (x * norm_factor * g).to(output_ptr.dtype.element_ty),
            mask=mask,
        )


def fused_add_rmsnorm(
    out: torch.Tensor,
    x: torch.Tensor,
    res_in: torch.Tensor,
    res_out: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> None:
    """In-place: ``out = rmsnorm(x + res_in) * weight``, ``res_out = x + res_in``.

    All four 2-D tensors may carry distinct row strides; each needs a stride-1
    last dim. Mirrors aiter's ``rmsnorm2d_fwd_with_add`` argument order.
    """
    n_rows, n_cols = x.shape
    aligned = all(_row_aligned(t) for t in (x, out, res_in, res_out))
    num_prgms = min(n_rows, _num_cus(x.device.index or 0))

    _fused_add_rmsnorm_kernel[(num_prgms,)](
        x,
        out,
        res_in,
        res_out,
        weight,
        x.stride(0),
        out.stride(0),
        res_in.stride(0),
        res_out.stride(0),
        n_rows,
        n_cols,
        epsilon,
        ALIGNED=aligned,
        BLOCK_SIZE=triton.next_power_of_2(n_cols),
        NUM_PRGMS=num_prgms,
    )
