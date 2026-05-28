# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

# When ATOM_USE_TRITON_PA_REDUCE=1, force aiter's paged_attention_decode_v2
# reduce path to use the PURE Triton fallback kernel
# (paged_attention_decode_ps_reduce_kernel) instead of the C++ HIP variant
# (pa_decode_ps_reduce_hip_kernel).
#
# The wrapper at pa_decode_gluon.py:5086-5163 has three paths:
#   1. C++ HIP — taken if CXX_PS_REDUCE_AVAILABLE is True (default)
#   2. FlyDSL Triton — has a compiler bug (NameError sink_rsrc in __else_7)
#   3. Pure Triton fallback — runs only if FlyDSL raises ImportError
#
# We disable (1) and force (2) to raise ImportError so the wrapper falls
# through to (3), the pure Triton kernel.
#
# WARNING: the pure Triton fallback is CORRECT but ~75× slower than HIP
# (TPOT 4.75s vs 0.063s on gpt-oss-120b). Leave this env OFF for any
# performance-oriented run; turn it on only when full-Triton coverage is
# required regardless of speed.
import os as _os

# --- Expand new backend envs into legacy ATOM_USE_TRITON_* envs ---
# The downstream dispatch sites (linear.py, layernorm.py, embed_head.py,
# sampler.py, fused_moe_triton.py) still read the legacy ATOM_USE_TRITON_*
# envs. The new ATOM_BACKEND / ATOM_<FAMILY>_BACKEND envs are a higher-level
# convenience: this block translates them into the legacy envs BEFORE the
# import-time monkey-patches below run, so all three knob styles (legacy
# env, per-family env, umbrella env) end up driving the same code path.
#
# Precedence: per-family > umbrella > unset (no change to legacy envs).
# Per-family wins so `ATOM_BACKEND=triton ATOM_LINEAR_BACKEND=aiter` is the
# "everything Triton except linear" bisecting idiom.
#
# setdefault() is used for legacy envs: an explicit `ATOM_USE_TRITON_MOE=0`
# from the caller is preserved (not overwritten by ATOM_BACKEND=triton).
_BACKEND_TO_LEGACY = {
    "ATTENTION": ["ATOM_USE_TRITON_MHA_PREFILL"],
    "LINEAR": ["ATOM_USE_TRITON_BF16_DENSE"],
    "NORM": ["ATOM_USE_TRITON_RMSNORM"],
    "SAMPLER": ["ATOM_USE_TRITON_SAMPLE", "ATOM_USE_TRITON_EXPONENTIAL"],
    "EMBEDDING": ["ATOM_USE_TRITON_EMBEDDING"],
    "MOE": ["ATOM_USE_TRITON_MOE"],
    # PA_REDUCE is intentionally NOT mapped to attention=triton: the pure
    # Triton paged-attention reduce kernel is ~75x slower than HIP on long
    # contexts, so it must remain an explicit opt-in for audit runs only.
}
_umbrella_backend = _os.environ.get("ATOM_BACKEND")
if _umbrella_backend not in (None, "aiter", "triton"):
    raise ValueError(
        f"ATOM_BACKEND must be 'aiter' or 'triton', got {_umbrella_backend!r}"
    )
for _family, _legacy_envs in _BACKEND_TO_LEGACY.items():
    _fam_env = _os.environ.get(f"ATOM_{_family}_BACKEND")
    if _fam_env not in (None, "aiter", "triton"):
        raise ValueError(
            f"ATOM_{_family}_BACKEND must be 'aiter' or 'triton', got {_fam_env!r}"
        )
    _resolved = _fam_env if _fam_env is not None else _umbrella_backend
    if _resolved == "triton":
        for _le in _legacy_envs:
            _os.environ.setdefault(_le, "1")
# attention=triton also wants FLASH_ATTENTION_TRITON_AMD_ENABLE on (the
# Triton FA backend reads it). Tied to the legacy env so it's right
# regardless of which knob style the user used to flip MHA prefill.
if _os.environ.get("ATOM_USE_TRITON_MHA_PREFILL") == "1":
    _os.environ.setdefault("FLASH_ATTENTION_TRITON_AMD_ENABLE", "TRUE")

# When ATOM_USE_TRITON_MHA_PREFILL=1, force aiter's flash_attn_varlen_func
# (called by ATOM's prefill path) to route through the Triton MHA kernel
# (aiter/ops/triton/attention/mha.py) instead of CK's FmhaFwdKernel. aiter
# reads ENABLE_CK from os.environ exactly once at import time
# (aiter/jit/core.py:29), so we MUST flip it BEFORE the first `import aiter`
# below — that's why this block is at the very top of this file.
#
# Why the global switch (rather than a per-call backend hook): aiter does not
# expose a per-call CK/Triton selector for MHA today; the only knob is the
# import-time ENABLE_CK env. Routing this through ENABLE_CK is the smallest-
# diff way to unlock the Triton SWA path on gpt-oss.
if _os.getenv("ATOM_USE_TRITON_MHA_PREFILL", "0") == "1":
    _os.environ["ENABLE_CK"] = "0"
    # Belt-and-suspenders: aiter captures ENABLE_CK at module-import time
    # (aiter/jit/core.py:29). The env-var flip above ONLY works if no other
    # module has imported aiter before us. If something does, the captured
    # constant is already 1 and the dispatch keeps routing to CK. Patch the
    # already-imported module's attribute too so the dispatch function sees
    # the flipped value at call time (Python closes over the module global,
    # not the captured constant). Idea adopted from the fork/triton branch.
    try:
        import aiter.ops.mha as _aiter_mha

        _aiter_mha.ENABLE_CK = False
    except ImportError:
        # aiter not yet importable — the env var alone will catch it on the
        # eventual first import. Don't fail ATOM startup.
        pass

# When ATOM_USE_TRITON_BF16_DENSE=1, also redirect aiter.tuned_gemm's auto-
# tuned "torch" libtype fallback to use the Triton gemm_a16w16 path. The
# auto-tuner picks libtype="torch" for certain small/odd shapes (e.g. the
# Tensile MT64x16x128 kernels that fire from non-LinearBase callsites), and
# the torch fallback calls F.linear → aten::mm → rocBLAS. Re-pointing the
# dispatcher's "torch" slot at `triton_gemm` keeps those shapes on Triton too.
if _os.getenv("ATOM_USE_TRITON_BF16_DENSE", "0") == "1":
    try:
        import aiter.tuned_gemm as _tg

        if "torch" in _tg.solMap and "triton" in _tg.solMap:
            _tg.solMap["torch"] = _tg.solMap["triton"]
    except Exception:
        pass

if _os.getenv("ATOM_USE_TRITON_PA_REDUCE", "0") == "1":
    try:
        import aiter.ops.triton.gluon.pa_decode_gluon as _pa_decode_mod

        _pa_decode_mod.CXX_PS_REDUCE_AVAILABLE = False

        def _force_pure_triton_fallback(*_a, **_kw):
            raise ImportError(
                "ATOM: forcing pure Triton paged_attention_decode_ps_reduce_kernel "
                "fallback (FlyDSL variant blocked by sink_rsrc closure-capture bug)"
            )

        _pa_decode_mod.launch_pa_decode_ps_reduce_flydsl = _force_pure_triton_fallback
    except Exception:
        pass

from atom.model_engine.llm_engine import LLMEngine
from atom.sampling_params import SamplingParams

# interface for upper framework to construct the model from ATOM
from atom.plugin import prepare_model

__all__ = [
    "LLMEngine",
    "SamplingParams",
    "prepare_model",
]
