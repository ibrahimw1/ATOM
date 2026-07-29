"""Exp(1) noise in Triton, for the sampler's Gumbel-max style draw.

Replaces ``aiter.ops.triton.rng.exponential``, which existed only in a fork of
aiter and in no upstream revision, so ATOM could not enable
``ATOM_USE_TRITON_EXPONENTIAL`` against upstream at all.

Not bit-exact against ``torch.empty(...).exponential_(1)``, which the lever
already documents: aten draws from Philox and this draws from Triton's counter
based RNG. Both give Exp(1) by the same inverse-CDF transform, ``-log(u)`` for
``u`` uniform, which is also what aiter's HIP sampler uses.

The seed is drawn from torch's default generator, so ``torch.manual_seed`` fixes
the sequence, and successive calls draw different noise. One caveat that the aten
path does not have: a call captured into a CUDA graph bakes its seed in, so every
replay would reuse the same noise. ATOM samples outside the captured graph, and
the sampler's default path reuses one cached noise row across the batch anyway
(deliberately, for run-to-run determinism); this allocator is only reached for the
independent-noise draw that ``SamplingParams.n > 1`` needs.
"""

import torch
import triton
import triton.language as tl

# -log(u) with u this small already saturates near 87, and u is only ever this
# small if Triton's uniform returns exactly 0, which would otherwise give inf.
_MIN_UNIFORM = tl.constexpr(1.1754944e-38)


@triton.jit
def _exponential_kernel(
    out_ptr,
    n_elements,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    """Fill *out_ptr* with Exp(1) samples."""
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    uniform = tl.rand(seed, offsets)
    samples = -tl.log(tl.maximum(uniform, _MIN_UNIFORM))

    tl.store(out_ptr + offsets, samples.to(out_ptr.dtype.element_ty), mask=mask)


def exponential(
    shape: tuple[int, ...],
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """A tensor of *shape* filled with independent Exp(1) samples."""
    out = torch.empty(shape, dtype=dtype, device=device)
    n_elements = out.numel()
    if n_elements == 0:
        return out

    # Drawn from torch so torch.manual_seed governs the sequence. int32 keeps it
    # in the range Triton's RNG takes.
    seed = int(torch.randint(0, 2**31 - 1, (1,), dtype=torch.int64).item())

    BLOCK_SIZE = 1024
    _exponential_kernel[(triton.cdiv(n_elements, BLOCK_SIZE),)](
        out,
        n_elements,
        seed,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out
