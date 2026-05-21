# branch: `ibrahim/triton-swap-gpt-oss-20b`

Status: **WORKING.** gpt-oss-20b runs with ~78% of GPU time on Triton
kernels, 1.68× faster decode than stock main.

## Goal

For gpt-oss-20b, replace every kernel that has an existing Triton
equivalent with that equivalent. Scope changes to gpt-oss only — other
models keep their existing dispatch.

## Result (BS=4, ISL=128, OSL=32, BF16, single MI355X)

| run | TPOT | Triton% | CK% | AITER-HIP% | rocBLAS% |
|---|---:|---:|---:|---:|---:|
| gpt_oss_20b_main (stock) | 104 ms | 27.9% | 26.0% | 38.5% | 0.2% |
| gpt_oss_20b_ours (this branch) | **62 ms** | **77.6%** | **0.2%** | **5.5%** | 4.8% |

- `ck_tile::MoeFlatmmKernel` (884k µs, 26% of GPU time) → 0
- `aiter::opus_moe_sorting_entry` (235k µs) → 0
- `aiter::bf16gemm_fp32bf16_tn_*` (~975k µs, 38.5% of GPU time) → 0
- `aiter::swiglu_act_and_mul_bias_kernel` (fused into MoE matmul) → 0

## Commits

```
ff581e3c  gpt_oss: route BF16 dense GEMM through aiter Triton gemm_a16w16
d6624197  gpt_oss: route MoE through aiter Triton (matmul_ogs)
d2b05475  gpt_oss/rmsnorm: Stage 1 partial — wire infra, dispatch blocked
5c6fd78d  branch-notes: stage 1 RMSNorm swap reproduces stash crash
0607082f  gpt_oss: route plain RMSNorm through aiter Triton kernels   (Stage 1 v1, replaced by d2b05475)
```

`origin/main` is `d41076e5`.

## What works (Stages 2 / 3 / 4)

### Stage 3 — Triton MoE (`gpt_oss.py` MLPBlock.__init__)

The biggest single win. `FusedMoE.quant_method` reads
`ATOM_USE_TRITON_MOE` at construction time. We set it via `os.environ`
just before constructing `self.experts` and restore the previous value
in a `finally`. The MoE block then dispatches to
`triton_kernels.matmul_ogs` instead of the CK MoeFlatmm path.

### Stage 2 — SiluAndMul

Implicit. gpt-oss does not instantiate `SiluAndMul` separately; the
swiglu activation lives inside `FusedMoE` (`activation=Swiglu`). After
Stage 3, the MoE matmul kernels carry the `_swiglu` suffix
(`_matmul_ogs_NNT_bf16xbf16xmxfp4_32x256x256x1_swiglu`) — activation
is fused into the Triton matmul.

### Stage 4 — BF16 dense GEMM (`linear.py` + `gpt_oss.py`)

- `LinearBase.__init__` gets `use_triton_bf16: bool = False` (per-
  instance flag).
- `LinearBase.forward` adds an `if self.use_triton_bf16 and bf16/fp16 +
  2D + not shuffled:` branch in the `QuantType.No` path, dispatching to
  the new `_triton_bf16_linear_plain` helper (calls
  `aiter.ops.triton.gemm.basic.gemm_a16w16`).
- In `gpt_oss.py` we set `self.{qkv_proj,o_proj,router}.use_triton_bf16
  = True` after constructing the wrappers (less invasive than threading
  the kwarg through `QKVParallelLinear`/`RowParallelLinear`/
  `ReplicatedLinear`).

## What does NOT work yet (Stage 1)

Wiring `aiter.ops.triton.normalization.rmsnorm.{rms_norm,
rmsnorm2d_fwd_with_add}` into `RMSNorm.forward` via an
`if self.use_triton_bf16:` branch **reproducibly crashes** the AITER
FlyDSL MoE heuristic kernel with `Memory access fault by GPU node-4`
during the first warmup forward — even when the if-branch's body never
executes for the failing layer. Bisected over many runs (see commits
`0607082f`, `d2b05475`):

- bare main: passes.
- kwarg + flag, no forward change: passes.
- helpers defined but unused: passes.
- if/else dispatch in `RMSNorm.forward`: crashes.
- if/else with helper wrapped in `torch.no_grad()` + `.detach()`: crashes.
- bisect to only `norm` site (Triton-on, runs AFTER MoE): crashes.

Suspected root cause: `@support_torch_compile` on `GptOssModel` plus
`@mark_trace(prefix="rmsnorm", torch_compile=True)` on
`RMSNorm.forward` hash the forward bytecode for cache lookup; the
bytecode change invalidates downstream CUDA-graph kernel-pointer state
in a way that the AITER MoE heuristic-path FlyDSL kernel cannot
recover from.

Note that `LinearBase.forward` uses plain `@mark_trace` (no
`torch_compile=True`) and the same dispatch pattern works there
(Stage 4). The decorator's `torch_compile=True` setting is the
discriminator.

**Workaround paths not yet attempted:** subclass RMSNorm and rebind
`self.forward` to a Triton-aware variant at `__init__` time (keeps the
parent class bytecode intact); move the dispatch inside the existing
`rmsnorm2d_fwd_` free function and gate on a thread-local (affects all
callers); or strip `torch_compile=True` from the `@mark_trace`
decorator on `RMSNorm.forward` and verify nothing else depends on it.

Stage 1 contribution kept on the branch: the kwarg, the flag, the two
plain helper functions, and the docstring documenting the blocker.
Dispatch is intentionally NOT wired so the branch as a whole runs.

## Reproducing

```bash
cd /home/ibrawani/testing/atom-profiling
source scripts/env_ibrahimw1.sh
HIP_VISIBLE_DEVICES=2 bash scripts/profile_gpt_oss_20b_ours.sh
bash scripts/regen.sh
```

Container `rocm/atom-dev:nightly_202605081558`, mounts
`/home/ibrawani/dev/ATOM` (this branch) at `/app/ATOM`, single GPU TP=1.
Both cells of `profile_gpt_oss_20b_ours.sh` produce ~5.0 MB traces;
the dashboard at `results/users/ibrahimw1/figures/dashboard.html`
shows the kernel-mix shift.

## Out of scope

- `aiter::wv_splitk_small_fp16_bf16_kernel` (KV-write helper): no
  Triton equivalent in AITER.
- Small residual `rocblas-tensile` Cijk_* GEMMs (~7k µs total): AITER
  has Triton coverage in principle but the dispatch path is different
  and not exercised by Stage 4's `LinearBase.QuantType.No` branch.
- Other models: every per-instance flag in this branch defaults False,
  so Llama, Qwen, MiniMax, DeepSeek etc. are unaffected.

## Stash

`stash@{0}: triton-swap-wip 2026-05-20` (Ibrahim's earlier env-var-gated
attempt) is preserved untouched. Pop with `git stash pop` to restore
that version, or `git stash drop stash@{0}` to discard now that this
branch supersedes it.
