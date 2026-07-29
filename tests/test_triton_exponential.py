#!/usr/bin/env python3
"""Offline gate for ATOM's Triton Exp(1) noise.

There is no bit-exact reference to check against -- the lever it serves already
documents that it draws from a different RNG than aten's Philox -- so what is
pinned instead is what the sampler actually relies on: the samples are Exp(1)
distributed, none of them is zero or infinite (the sampler divides by them), the
sequence is fixed by torch.manual_seed, and successive calls differ.

No model or engine needed."""

import pytest
import torch

try:
    from atom.model_ops.triton_exponential import exponential
except Exception as _e:  # pragma: no cover - bare-pytest import env
    pytest.skip(f"requires full atom import env: {_e}", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the RNG is a GPU (Triton) kernel"
)

DEV = "cuda"


@pytest.mark.parametrize("shape", [(1, 201088), (7, 4096), (1024,), (3, 5, 7)])
def test_shape_and_dtype(shape):
    got = exponential(shape, dtype=torch.float32, device=DEV)
    assert got.shape == shape
    assert got.dtype == torch.float32
    assert got.device.type == "cuda"


def test_samples_are_usable_as_divisors():
    got = exponential((4, 201088), dtype=torch.float32, device=DEV)
    assert torch.isfinite(got).all(), "an infinite sample would poison the sampler"
    assert (got > 0).all(), "a zero sample would divide to infinity"


def test_distribution_is_exp1():
    """Exp(1) has mean 1, variance 1, and median ln 2."""
    got = exponential((64, 201088), dtype=torch.float32, device=DEV).double()
    assert abs(got.mean().item() - 1.0) < 0.01
    assert abs(got.var().item() - 1.0) < 0.02
    assert abs(got.median().item() - 0.6931) < 0.01
    # P(X > x) = exp(-x): a tail check, since mean and variance alone would also
    # pass for other distributions.
    for x in (0.5, 1.0, 3.0):
        survival = (got > x).double().mean().item()
        assert abs(survival - pow(2.718281828, -x)) < 0.005, x


def test_manual_seed_fixes_the_sequence():
    torch.manual_seed(1234)
    first = exponential((3, 8192), dtype=torch.float32, device=DEV)
    torch.manual_seed(1234)
    second = exponential((3, 8192), dtype=torch.float32, device=DEV)
    assert torch.equal(first, second)


def test_successive_calls_differ():
    torch.manual_seed(7)
    first = exponential((3, 8192), dtype=torch.float32, device=DEV)
    second = exponential((3, 8192), dtype=torch.float32, device=DEV)
    assert not torch.equal(first, second), "n>1 fan-out needs independent draws"


def test_empty_shape():
    got = exponential((0, 128), dtype=torch.float32, device=DEV)
    assert got.shape == (0, 128)
