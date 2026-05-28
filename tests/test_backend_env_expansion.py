# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Regression test for the ATOM_BACKEND / ATOM_<FAMILY>_BACKEND env expansion
in atom/__init__.py.

The expansion translates the new umbrella + per-family envs into the legacy
ATOM_USE_TRITON_* envs that the dispatch sites still read. We test:
  - umbrella alone flips all families
  - per-family overrides umbrella (bisecting story)
  - per-family alone leaves untouched families unset
  - explicit legacy env wins (setdefault never overwrites)
  - invalid values raise ValueError

Uses subprocess for isolation -- the expansion runs at import time and we
need a clean os.environ per scenario.
"""

import os
import subprocess
import sys
import textwrap

LEGACY_ENVS = [
    "ATOM_USE_TRITON_MHA_PREFILL",
    "ATOM_USE_TRITON_BF16_DENSE",
    "ATOM_USE_TRITON_RMSNORM",
    "ATOM_USE_TRITON_SAMPLE",
    "ATOM_USE_TRITON_EXPONENTIAL",
    "ATOM_USE_TRITON_EMBEDDING",
    "ATOM_USE_TRITON_MOE",
    "FLASH_ATTENTION_TRITON_AMD_ENABLE",
]
BACKEND_ENVS = [
    "ATOM_BACKEND",
    "ATOM_ATTENTION_BACKEND",
    "ATOM_LINEAR_BACKEND",
    "ATOM_NORM_BACKEND",
    "ATOM_SAMPLER_BACKEND",
    "ATOM_EMBEDDING_BACKEND",
    "ATOM_MOE_BACKEND",
]


def _run_expansion(env_overrides: dict[str, str], expect_raise: bool = False):
    """Spawn fresh python that strips backend+legacy envs, applies overrides,
    execs just the expansion block from atom/__init__.py, prints the
    resulting legacy envs as a parseable summary.
    """
    init_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "atom",
        "__init__.py",
    )
    overrides_repr = repr(env_overrides)
    legacy_repr = repr(LEGACY_ENVS)
    backend_repr = repr(BACKEND_ENVS)
    expect_raise_repr = repr(expect_raise)
    code = textwrap.dedent(f"""
        import json, os, sys
        for k in {legacy_repr} + {backend_repr}:
            os.environ.pop(k, None)
        for k, v in {overrides_repr}.items():
            os.environ[k] = v
        src = open({init_path!r}).read()
        start = src.index("import os as _os")
        end = src.index("# When ATOM_USE_TRITON_MHA_PREFILL=1")
        try:
            exec(src[start:end], {{"__name__": "__main__"}})
        except ValueError as exc:
            print("RAISED:" + str(exc))
            sys.exit(0 if {expect_raise_repr} else 1)
        result = {{k: os.environ.get(k) for k in {legacy_repr}}}
        print("RESULT:" + json.dumps(result))
        """)
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert (
        proc.returncode == 0
    ), f"subprocess failed:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    if expect_raise:
        assert "RAISED:" in proc.stdout
        return proc.stdout
    import json

    line = next(l for l in proc.stdout.splitlines() if l.startswith("RESULT:"))
    return json.loads(line[len("RESULT:") :])


def test_umbrella_flips_all_families():
    result = _run_expansion({"ATOM_BACKEND": "triton"})
    for k in LEGACY_ENVS:
        assert result[k] == (
            "1" if k != "FLASH_ATTENTION_TRITON_AMD_ENABLE" else "TRUE"
        ), f"{k} = {result[k]!r}"


def test_per_family_overrides_umbrella():
    """Bisecting idiom: ATOM_BACKEND=triton ATOM_LINEAR_BACKEND=aiter
    leaves the linear legacy env unset (so dispatch falls back to aiter)."""
    result = _run_expansion({"ATOM_BACKEND": "triton", "ATOM_LINEAR_BACKEND": "aiter"})
    assert result["ATOM_USE_TRITON_BF16_DENSE"] is None
    # Other families still flipped:
    assert result["ATOM_USE_TRITON_MHA_PREFILL"] == "1"
    assert result["ATOM_USE_TRITON_MOE"] == "1"


def test_per_family_alone():
    """Per-family without umbrella flips only that family."""
    result = _run_expansion({"ATOM_SAMPLER_BACKEND": "triton"})
    assert result["ATOM_USE_TRITON_SAMPLE"] == "1"
    assert result["ATOM_USE_TRITON_EXPONENTIAL"] == "1"
    # Other families untouched:
    assert result["ATOM_USE_TRITON_MHA_PREFILL"] is None
    assert result["ATOM_USE_TRITON_BF16_DENSE"] is None


def test_legacy_env_wins_over_umbrella():
    """An explicit legacy env (even '0') survives because we use setdefault."""
    result = _run_expansion({"ATOM_BACKEND": "triton", "ATOM_USE_TRITON_MOE": "0"})
    assert result["ATOM_USE_TRITON_MOE"] == "0"
    # Everything else still flipped:
    assert result["ATOM_USE_TRITON_MHA_PREFILL"] == "1"


def test_invalid_umbrella_raises():
    _run_expansion({"ATOM_BACKEND": "junk"}, expect_raise=True)


def test_invalid_per_family_raises():
    _run_expansion({"ATOM_LINEAR_BACKEND": "junk"}, expect_raise=True)


def test_default_is_aiter():
    """No envs set -> no legacy envs touched."""
    result = _run_expansion({})
    for k in LEGACY_ENVS:
        assert result[k] is None, f"{k} unexpectedly set to {result[k]!r}"
