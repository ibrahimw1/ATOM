#!/usr/bin/env python3
"""Offline gate for ATOM's Triton embedding gather.

The kernel replaces F.embedding under ATOM_USE_TRITON_EMBEDDING, so the contract
is exact equality with it -- not approximate agreement, since a gather only moves
bytes. The out-of-range cases are pinned too: F.embedding on a negative id reads
the row before the table, and embed_head.py documents that MTP spec-decode can
transiently carry -1 into a lookup, so those rows must come back as zeros rather
than as whatever preceded the table.

No model or engine needed."""

import pytest
import torch
import torch.nn.functional as F

try:
    from atom.model_ops.triton_embedding_gather import gather
except Exception as _e:  # pragma: no cover - bare-pytest import env
    pytest.skip(f"requires full atom import env: {_e}", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the gather is a GPU (Triton) kernel"
)

DEV = "cuda"
DTYPE = torch.bfloat16
DIM = 2880  # gpt-oss-120b hidden size
VOCAB = 4096


@pytest.fixture(scope="module")
def weight():
    return torch.randn(VOCAB, DIM, dtype=DTYPE, device=DEV)


@pytest.mark.parametrize("n_tokens", [1, 2, 7, 33, 128, 1024])
def test_matches_f_embedding_exactly(weight, n_tokens):
    x = torch.randint(0, VOCAB, (n_tokens,), device=DEV, dtype=torch.int64)
    assert torch.equal(gather(x, weight), F.embedding(x, weight))


def test_gathers_the_requested_rows(weight):
    """Guards against an index bug that equality against a random table could hide."""
    x = torch.tensor([0, VOCAB - 1, 5, 5, 1], device=DEV, dtype=torch.int64)
    got = gather(x, weight)
    for row, token_id in enumerate(x.tolist()):
        assert torch.equal(got[row], weight[token_id])


def test_out_of_range_ids_read_as_zero(weight):
    x = torch.tensor([-1, 3, VOCAB, VOCAB + 100, 7], device=DEV, dtype=torch.int64)
    got = gather(x, weight)
    assert torch.all(got[0] == 0), "negative id must not read before the table"
    assert torch.all(got[2] == 0) and torch.all(got[3] == 0)
    assert torch.equal(got[1], weight[3])
    assert torch.equal(got[4], weight[7])


def test_multidimensional_ids_flatten(weight):
    x = torch.randint(0, VOCAB, (4, 8), device=DEV, dtype=torch.int64)
    got = gather(x, weight)
    assert got.shape == (32, DIM)
    assert torch.equal(got, F.embedding(x.reshape(-1), weight))


def test_row_strided_table(weight):
    """A table that is a column slice of a wider buffer keeps its own row stride."""
    wide = torch.randn(VOCAB, 2 * DIM, dtype=DTYPE, device=DEV)
    table = wide[:, :DIM]
    assert table.stride(0) != DIM
    x = torch.randint(0, VOCAB, (17,), device=DEV, dtype=torch.int64)
    assert torch.equal(gather(x, table), F.embedding(x, table))


def test_width_not_a_power_of_two(weight):
    """2880 is already not one; check an odd width so the column mask is exercised."""
    table = torch.randn(VOCAB, 1235, dtype=DTYPE, device=DEV)
    x = torch.randint(0, VOCAB, (9,), device=DEV, dtype=torch.int64)
    assert torch.equal(gather(x, table), F.embedding(x, table))


def test_int32_ids(weight):
    x = torch.randint(0, VOCAB, (11,), device=DEV, dtype=torch.int32)
    assert torch.equal(gather(x, weight), F.embedding(x, weight))


def test_empty_input(weight):
    x = torch.empty(0, device=DEV, dtype=torch.int64)
    got = gather(x, weight)
    assert got.shape == (0, DIM)


def test_fp16_and_fp32_tables():
    x = torch.randint(0, VOCAB, (13,), device=DEV, dtype=torch.int64)
    for dtype in (torch.float16, torch.float32):
        table = torch.randn(VOCAB, DIM, dtype=dtype, device=DEV)
        assert torch.equal(gather(x, table), F.embedding(x, table)), dtype
