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

from atom.model_engine.llm_engine import LLMEngine
from atom.sampling_params import SamplingParams

from atom.plugin.sglang import prepare_model_for_sglang

__all__ = [
    "LLMEngine",
    "SamplingParams",
    "prepare_model_for_sglang",
]
