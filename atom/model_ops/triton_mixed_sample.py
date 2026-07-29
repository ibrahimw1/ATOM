"""Mixed greedy/stochastic sampling in Triton, over pre-drawn Exp(1) noise.

Replaces ``aiter.ops.triton.sample.mix_sample``, which existed only in a fork of
aiter and in no upstream revision, so ATOM could not enable
``ATOM_USE_TRITON_SAMPLE`` against upstream at all. The signature and behaviour
follow aiter's HIP ``mixed_sample_outer_exponential``, which stays the fallback
when the lever is off.

Per row: temperature 0 means argmax of the logits, otherwise the winner is the
largest ``exp(logit/T - rowmax) / (noise + eps)``.

That score is computed here as ``logit/T - log(noise + eps)`` instead. Dropping
the exp does not change which index wins -- exp is monotone, and the row max it
subtracts is a constant within the row, so both forms are maximised by the same
index -- but it removes the overflow guard the HIP kernel needs, and with it the
running-max rescale it carries through the reduction, leaving a single pass with
no exp at all. Where HIP's exp underflows to zero for far-from-max logits, those
indices tie at 0.0 and the earliest wins; here they stay ordered. It cannot
change the answer, since the winning index has exp(0) = 1 and so scores above any
underflowed one.

Ties resolve to the lowest index, as hipcub's ArgMax does.
"""

import torch
import triton
import triton.language as tl

# Mirrors the HIP kernel's clamp, which keeps 1/T finite for tiny temperatures.
_MIN_TEMPERATURE = tl.constexpr(1e-5)


@triton.jit
def _mixed_sample_kernel(
    out_ptr,
    logits_ptr,
    exponentials_ptr,
    temperatures_ptr,
    logits_row_stride,
    exponentials_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per row: pick this row's token."""
    row = tl.program_id(0)
    temperature = tl.load(temperatures_ptr + row)
    greedy = temperature == 0.0
    inv_temperature = 1.0 / tl.maximum(temperature, _MIN_TEMPERATURE)

    logits_row = logits_ptr + row * logits_row_stride
    exponentials_row = exponentials_ptr + row * exponentials_row_stride

    best_value = float("-inf")
    best_index = 0

    for start in tl.range(0, n_cols, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols

        logits = tl.load(logits_row + offsets, mask=mask, other=0.0).to(tl.float32)
        if greedy:
            scores = logits
        else:
            noise = tl.load(exponentials_row + offsets, mask=mask, other=0.0)
            scores = logits * inv_temperature - tl.log(noise + eps)
        scores = tl.where(mask, scores, float("-inf"))

        block_value = tl.max(scores, axis=0)
        block_index = start + tl.argmax(scores, axis=0)

        # Strictly greater, so an equal score never displaces a lower index.
        take = block_value > best_value
        best_value = tl.where(take, block_value, best_value)
        best_index = tl.where(take, block_index, best_index)

    tl.store(out_ptr + row, best_index.to(out_ptr.dtype.element_ty))


def mixed_sample_outer_exponential(
    out: torch.Tensor,
    input: torch.Tensor,
    exponentials: torch.Tensor,
    temperatures: torch.Tensor,
    eps: float = 1e-10,
) -> None:
    """Write one sampled token id per row of *input* into *out*.

    Argument order and in-place output match aiter's HIP kernel. *exponentials*
    may be a broadcast view of a single noise row -- a row stride of 0 is read as
    the HIP kernel reads it, with every row drawing on the same noise.
    """
    n_rows, n_cols = input.shape
    if input.numel() == 0:
        return

    _mixed_sample_kernel[(n_rows,)](
        out,
        input,
        exponentials,
        temperatures,
        input.stride(0),
        exponentials.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE=1024,
    )
