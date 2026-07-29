"""Embedding-table gather in Triton, for the TP=1 vocab embedding lookup.

Replaces ``aiter.ops.triton.embedding.gather``, which existed only in a fork of
aiter and in no upstream revision, so ATOM could not enable
``ATOM_USE_TRITON_EMBEDDING`` against upstream at all.

One program per token reads that token's whole row in a single tile. Measured on
MI355X against gpt-oss-120b's table (201088 x 2880, bf16), device time per call:

    tokens        1      4     16    128   1024   8192
    F.embedding  1.9u   2.9u   6.9u   2.0u   3.0u  15.4u
    this kernel  1.9u   2.0u   2.1u   2.2u   2.8u  14.8u

So it is at parity where decode lives and slightly ahead at prefill widths. Its
eager launch does cost ~13.5us against aten's ~5.5us, but that is Triton's own
dispatch overhead, not this launcher's: an empty Triton kernel already costs
7.6us to launch. It is invisible where it would matter -- decode is graph
captured, and at prefill it is ~0.01% of the forward.

Out-of-range ids read as zero rather than reading out of bounds. F.embedding on
a negative id reads the row before the table; embed_head.py documents that MTP
spec-decode can transiently carry -1 through an embedding lookup, so the range
check is kept here as it is in ATOM's masked_embedding path. For in-range ids
the result is bit-identical to F.embedding.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    weight_row_stride,
    out_row_stride,
    n_table_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """``out[i, :] = weight[x[i], :]``, or zeros where ``x[i]`` is out of range."""
    row = tl.program_id(0)
    token_id = tl.load(x_ptr + row)
    in_range = (token_id >= 0) & (token_id < n_table_rows)

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    vals = tl.load(
        weight_ptr + token_id * weight_row_stride + col_offsets,
        mask=mask & in_range,
        other=0.0,
    )
    tl.store(out_ptr + row * out_row_stride + col_offsets, vals, mask=mask)


def gather(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Gather embedding rows for the token ids in *x*.

    *x* holds token ids of any shape; the result is ``(x.numel(), dim)``, matching
    what the ids flatten to. *weight* needs a stride-1 last dim.
    """
    n_tokens = x.numel()
    n_cols = weight.shape[1]
    out = torch.empty(n_tokens, n_cols, dtype=weight.dtype, device=weight.device)
    if n_tokens == 0:
        return out

    _gather_kernel[(n_tokens,)](
        x,
        weight,
        out,
        weight.stride(0),
        out.stride(0),
        weight.shape[0],
        n_cols,
        BLOCK_SIZE=triton.next_power_of_2(n_cols),
    )
    return out
