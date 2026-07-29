#!/usr/bin/env python3
"""Contract gate for the sliding-window right edge ATOM sends to aiter's Triton MHA.

CK expresses a causal sliding window as right edge 0; aiter's Triton wrapper
rejects anything but -1 and derives the window from the left edge alone, so
ATOM_USE_TRITON_MHA_PREFILL substitutes -1 (see attention_mha.py). This pins the
substitution empirically rather than by reading aiter: the -1 form must still
apply the window, so it must match a windowed reference *and* differ from plain
causal. If a future aiter made the right edge meaningful, or made the window
silently drop out, the second assertion fails.

No model or engine needed."""

import pytest
import torch

try:
    from aiter.ops.triton.attention.mha import flash_attn_varlen_func
except Exception as _e:  # pragma: no cover - bare-pytest import env
    pytest.skip(f"requires aiter Triton MHA: {_e}", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Triton MHA is a GPU kernel"
)

DEV = "cuda"
DTYPE = torch.bfloat16
SEQ = 1024
H_Q, H_K, HEAD_DIM = 64, 8, 64  # gpt-oss-120b prefill shape
WINDOW_LEFT = 127


def _reference(q, k, v, scale, window_left):
    """fp32 attention with a bottom-right causal mask, optionally windowed.

    window_left None means plain causal.
    """
    # [S, H, D] -> [H, S, D], expanding kv heads across their query group.
    group = q.shape[1] // k.shape[1]
    qf = q.float().transpose(0, 1)
    kf = k.float().repeat_interleave(group, dim=1).transpose(0, 1)
    vf = v.float().repeat_interleave(group, dim=1).transpose(0, 1)

    scores = (qf @ kf.transpose(-1, -2)) * scale
    idx = torch.arange(SEQ, device=q.device)
    allowed = idx[None, :] <= idx[:, None]
    if window_left is not None:
        allowed &= idx[None, :] >= (idx[:, None] - window_left)
    scores = scores.masked_fill(~allowed, float("-inf"))
    return (torch.softmax(scores, dim=-1) @ vf).transpose(0, 1)


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def _run(window_size):
    torch.manual_seed(0)
    q = torch.randn(SEQ, H_Q, HEAD_DIM, dtype=DTYPE, device=DEV)
    k = torch.randn(SEQ, H_K, HEAD_DIM, dtype=DTYPE, device=DEV)
    v = torch.randn(SEQ, H_K, HEAD_DIM, dtype=DTYPE, device=DEV)
    cu = torch.tensor([0, SEQ], dtype=torch.int32, device=DEV)
    scale = HEAD_DIM**-0.5
    out = flash_attn_varlen_func(
        q,
        k,
        v,
        cu,
        cu,
        SEQ,
        SEQ,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
    )
    return (out[0] if isinstance(out, tuple) else out), q, k, v, scale


def test_right_edge_minus_one_still_applies_the_window():
    got, q, k, v, scale = _run((WINDOW_LEFT, -1))

    windowed = _reference(q, k, v, scale, WINDOW_LEFT)
    causal_only = _reference(q, k, v, scale, None)

    cos_windowed = _cos(got, windowed)
    cos_causal = _cos(got, causal_only)

    assert cos_windowed > 0.9999, (
        f"(left,-1)+causal does not match a windowed reference: cos={cos_windowed:.8f}"
    )
    # The window must actually bite; if it silently dropped out the kernel would
    # be computing plain causal and this would also be ~1.0.
    assert cos_causal < 0.99, (
        f"(left,-1)+causal matches plain causal (cos={cos_causal:.8f}): the "
        f"sliding window is not being applied"
    )


def test_no_window_matches_plain_causal():
    """Control: (-1,-1) must be plain causal, so the above contrast is meaningful."""
    got, q, k, v, scale = _run((-1, -1))
    assert _cos(got, _reference(q, k, v, scale, None)) > 0.9999
