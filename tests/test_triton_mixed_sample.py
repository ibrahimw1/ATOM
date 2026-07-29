#!/usr/bin/env python3
"""Offline gate for ATOM's Triton mixed sampler.

The kernel replaces aiter's HIP mixed_sample_outer_exponential, which is still
the fallback when ATOM_USE_TRITON_SAMPLE=0, so the contract is that both pick the
same token. That is what is asserted here -- against the HIP kernel itself rather
than against a reimplemented reference, since a reference would only re-state this
kernel's own reading of the algorithm.

The cases that matter beyond plain agreement: temperature 0 rows (greedy) mixed
with sampled rows in one batch, a broadcast noise row (ATOM passes a (1, vocab)
tensor expanded across the batch, so its row stride is 0), a non-contiguous logits
view, and ties.

No model or engine needed."""

import pytest
import torch

try:
    from aiter import mixed_sample_outer_exponential as hip_mixed_sample

    from atom.model_ops.triton_mixed_sample import mixed_sample_outer_exponential
except Exception as _e:  # pragma: no cover - bare-pytest import env
    pytest.skip(f"requires full atom import env: {_e}", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the sampler is a GPU (Triton) kernel"
)

DEV = "cuda"
VOCAB = 201088  # gpt-oss-120b vocab
EPS = 1e-10


def _run(logits, exponentials, temperatures, fn):
    out = torch.empty(logits.shape[0], dtype=torch.int, device=DEV)
    fn(out, logits, exponentials, temperatures, eps=EPS)
    return out


def _assert_same_tokens(logits, exponentials, temperatures, what):
    want = _run(logits, exponentials, temperatures, hip_mixed_sample)
    got = _run(logits, exponentials, temperatures, mixed_sample_outer_exponential)
    mismatch = (got != want).nonzero().flatten().tolist()
    assert not mismatch, (
        f"{what}: rows {mismatch[:8]} differ; "
        f"triton={got[mismatch[:8]].tolist()} hip={want[mismatch[:8]].tolist()}"
    )


def _noise(n_rows, vocab=VOCAB):
    return torch.empty(n_rows, vocab, dtype=torch.float32, device=DEV).exponential_(1)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("n_rows", [1, 4, 33, 128])
def test_agrees_with_hip_when_sampling(dtype, n_rows):
    torch.manual_seed(0)
    logits = torch.randn(n_rows, VOCAB, dtype=dtype, device=DEV)
    temperatures = torch.full((n_rows,), 0.7, dtype=torch.float32, device=DEV)
    _assert_same_tokens(logits, _noise(n_rows), temperatures, f"{dtype} x{n_rows}")


@pytest.mark.parametrize("n_rows", [1, 7, 64])
def test_agrees_with_hip_when_greedy(n_rows):
    torch.manual_seed(1)
    logits = torch.randn(n_rows, VOCAB, dtype=torch.bfloat16, device=DEV)
    temperatures = torch.zeros(n_rows, dtype=torch.float32, device=DEV)
    got = _run(logits, _noise(n_rows), temperatures, mixed_sample_outer_exponential)
    _assert_same_tokens(logits, _noise(n_rows), temperatures, "greedy")
    # Greedy must be the plain argmax, independent of the noise.
    assert torch.equal(got.long(), logits.float().argmax(dim=-1))


def test_mixed_greedy_and_sampled_rows_in_one_batch():
    torch.manual_seed(2)
    n_rows = 16
    logits = torch.randn(n_rows, VOCAB, dtype=torch.bfloat16, device=DEV)
    temperatures = torch.where(
        torch.arange(n_rows, device=DEV) % 2 == 0,
        torch.zeros(n_rows, device=DEV),
        torch.full((n_rows,), 1.3, device=DEV),
    )
    _assert_same_tokens(logits, _noise(n_rows), temperatures, "mixed batch")


def test_broadcast_noise_row():
    """ATOM's default path expands one (1, vocab) noise row across the batch."""
    torch.manual_seed(3)
    n_rows = 24
    logits = torch.randn(n_rows, VOCAB, dtype=torch.bfloat16, device=DEV)
    shared = _noise(1).expand(n_rows, VOCAB)
    assert shared.stride(0) == 0
    temperatures = torch.full((n_rows,), 0.9, dtype=torch.float32, device=DEV)
    _assert_same_tokens(logits, shared, temperatures, "broadcast noise")


def test_row_strided_logits():
    torch.manual_seed(4)
    wide = torch.randn(8, VOCAB + 64, dtype=torch.bfloat16, device=DEV)
    logits = wide[:, :VOCAB]
    assert logits.stride(0) != VOCAB
    temperatures = torch.full((8,), 0.5, dtype=torch.float32, device=DEV)
    _assert_same_tokens(logits, _noise(8), temperatures, "strided logits")


def test_tiny_temperature_is_clamped_like_hip():
    torch.manual_seed(5)
    logits = torch.randn(4, VOCAB, dtype=torch.bfloat16, device=DEV)
    temperatures = torch.full((4,), 1e-9, dtype=torch.float32, device=DEV)
    _assert_same_tokens(logits, _noise(4), temperatures, "temperature underflow")


def test_ties_resolve_to_the_lowest_index():
    """A flat row with flat noise: every score is equal, so index 0 must win."""
    logits = torch.zeros(2, 4096, dtype=torch.float32, device=DEV)
    exponentials = torch.ones(2, 4096, dtype=torch.float32, device=DEV)
    for temperature in (0.0, 1.0):
        temperatures = torch.full((2,), temperature, dtype=torch.float32, device=DEV)
        got = _run(logits, exponentials, temperatures, mixed_sample_outer_exponential)
        assert got.tolist() == [0, 0], f"temperature={temperature}: {got.tolist()}"


def test_noise_actually_steers_the_choice():
    """Guards against the noise being ignored, which agreement alone might miss."""
    n_rows, vocab = 4, 4096
    logits = torch.zeros(n_rows, vocab, dtype=torch.float32, device=DEV)
    exponentials = torch.ones(n_rows, vocab, dtype=torch.float32, device=DEV)
    want = [11, 2000, 4095, 7]
    for row, col in enumerate(want):
        exponentials[row, col] = 1e-6  # smallest divisor wins
    temperatures = torch.full((n_rows,), 1.0, dtype=torch.float32, device=DEV)
    got = _run(logits, exponentials, temperatures, mixed_sample_outer_exponential)
    assert got.tolist() == want
