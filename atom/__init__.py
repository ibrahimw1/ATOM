# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging as _logging
import os as _os

_triton_lever_logger = _logging.getLogger("atom")


def _lever_enabled(name: str) -> bool:
    return _os.getenv(name, "0") == "1"


# ATOM_USE_TRITON_MHA_PREFILL: route aiter's flash_attn_varlen_func (ATOM's
# prefill path) through the Triton MHA kernel instead of CK's FmhaFwdKernel.
# aiter reads ENABLE_CK from os.environ exactly once at import time
# (aiter/jit/core.py), so this MUST run before the first `import aiter` — hence
# its position at the very top of this file, ahead of the patches below.
#
# aiter exposes no per-call CK/Triton selector for MHA, so the import-time env
# is the only lever. Note it is GLOBAL: every CK-vs-Triton dispatch switches,
# not just MHA.
if _lever_enabled("ATOM_USE_TRITON_MHA_PREFILL"):
    _os.environ["ENABLE_CK"] = "0"

# ATOM_USE_TRITON_PA_REDUCE forces paged_attention_decode_v2's reduce step onto
# the pure-Triton paged_attention_decode_ps_reduce_kernel. The wrapper picks
# C++ HIP when CXX_PS_REDUCE_AVAILABLE, else FlyDSL, else the pure kernel; we
# disable the first and make the second raise ImportError to fall through.
#
# REACHABILITY: the reduce only runs when the decode is split across context
# partitions (pa_decode_gluon: `one_shot = max_context_partition_num <= 1`, and
# the reduce wrapper is called only `if not one_shot`). Short contexts are
# single-partition, so this lever is a no-op there — a short-context A/B will
# show no difference rather than a regression.
#
# WARNING: correct but ~75x slower than HIP when it does engage. Coverage
# measurements only; never enable for a timing run.
if _lever_enabled("ATOM_USE_TRITON_PA_REDUCE"):
    try:
        import aiter.ops.triton.gluon.pa_decode_gluon as _pa_decode_mod
    except ImportError as exc:
        _triton_lever_logger.warning(
            "ATOM_USE_TRITON_PA_REDUCE=1 but pa_decode_gluon is unavailable "
            "(%s); the reduce stays on the C++ HIP kernel.",
            exc,
        )
    else:
        if not hasattr(_pa_decode_mod, "launch_pa_decode_ps_reduce_flydsl"):
            _triton_lever_logger.warning(
                "ATOM_USE_TRITON_PA_REDUCE=1 but pa_decode_gluon has no "
                "launch_pa_decode_ps_reduce_flydsl; aiter's reduce dispatch "
                "changed and this patch no longer applies."
            )
        else:

            def _force_pure_triton_ps_reduce(*_args, **_kwargs):
                raise ImportError(
                    "ATOM: forcing the pure Triton "
                    "paged_attention_decode_ps_reduce_kernel fallback"
                )

            _pa_decode_mod.CXX_PS_REDUCE_AVAILABLE = False
            _pa_decode_mod.launch_pa_decode_ps_reduce_flydsl = (
                _force_pure_triton_ps_reduce
            )

from atom.model_engine.llm_engine import LLMEngine
from atom.sampling_params import SamplingParams

from atom.plugin.sglang import prepare_model_for_sglang

__all__ = [
    "LLMEngine",
    "SamplingParams",
    "prepare_model_for_sglang",
]
