# branch: `ibrahim/triton-swap-gpt-oss-20b`

Status: **WORKING.** Every kernel for gpt-oss-20b that has a known
Triton equivalent in AITER is now routed to Triton. Triton share went
from 27.9% (stock main) to 92.4%. TPOT improved 104 ms → 64 ms (1.63×
faster decode).

## Result

BS=4, ISL=128, OSL=32, BF16, single MI355X (gfx950):

| run | TPOT | Triton% | CK% | AITER-HIP% | rocBLAS% |
|---|---:|---:|---:|---:|---:|
| `gpt_oss_20b_main` (stock origin/main) | 104 ms | 27.9% | 26.0% | 38.5% | 0.2% |
| `gpt_oss_20b_ours` (this branch) | **64 ms** | **92.4%** | **0.1%** | **1.1%** | **0.0%** |

`triton-available` bucket (what the classifier says could be swapped):
**2,127,131 µs → 0 µs**. Every Triton-equivalent kernel that the
classifier knows about is now firing as Triton.

## Stages (in commit order)

| Stage | Kernel(s) replaced | µs | Status |
|---|---|---:|---|
| 3 | `ck_tile::MoeFlatmmKernel`, `aiter::opus_moe_sorting_entry` | 884k + 235k | ✓ via `ATOM_USE_TRITON_MOE=1` in `MLPBlock.__init__` |
| 2 | `aiter::swiglu_act_and_mul_bias_kernel` | 4.2k | ✓ implicit (swiglu fuses into Triton MoE matmul) |
| 4 | `aiter::bf16gemm_fp32bf16_tn_*` (BF16 GEMM in LinearBase) | ~975k | ✓ via `use_triton_bf16` flag on LinearBase + per-instance set in gpt_oss.py |
| 5 | `aiter::add_rmsnorm_quant_kernel` (BF16 RMSNorm) | 4.2k | ✓ via `TritonRMSNorm` subclass; uses `rmsnorm_forward_inference` for contiguity guard |
| 6 | `aiter::pa_decode_ps_reduce_hip_kernel` | 2.0k | ✗ **blocked** — Triton FlyDSL equivalent in `aiter` has `NameError: 'sink_rsrc' not defined` (aiter bug, not ATOM); HIP path retained |
| 7 | rocBLAS `Cijk_Alik_Bljk_BBS_BH_Bias_…` (ParallelLMHead) | 7.0k | ✓ via `use_triton_bf16` flag on ParallelLMHead |
| 8 | `ck_tile::FmhaFwdKernel` (CK MHA prefill) + aiter-hip decode helpers | ~140k combined | ✓ via `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` (Kevin's "SWA crash" warning no longer applies on current aiter) |

Stage 1 (the earlier env-gated RMSNorm attempt) is superseded by Stage 5.

## Critical lessons learned

1. **Never modify the bytecode of methods captured by `@support_torch_compile`**
   (CLAUDE.md rule). RMSNorm.forward in `layernorm.py` is captured by
   `@support_torch_compile` on `GptOssModel`; adding an `if` branch
   inside it mid-flight invalidated the Dynamo cache and crashed the
   MoE FlyDSL kernel during warmup. **Workaround:** subclass with a
   fresh forward method (TritonRMSNorm). New method = stable cache key.

2. **Use `rmsnorm_forward_inference`, not `rms_norm`**. The former does
   `.contiguous()` on inputs before the kernel launch; the latter
   doesn't, and the Triton kernel reads garbage from non-contiguous
   reshape views.

3. **`@mark_trace` is NOT the trigger by itself** — `LinearBase.forward`
   has `@mark_trace` (no `torch_compile=True`) and we added an if/else
   inside it (Stage 4) without crashing. The trigger is specifically
   `@mark_trace(torch_compile=True)` plus a class that ends up captured
   by an outer `@support_torch_compile` model.

4. **Kevin's `profile_gptoss_triton_flags.sh` notes were stale** —
   `FLASH_ATTENTION_TRITON_AMD_ENABLE` no longer crashes on SWA layers
   in the current `rocm/atom-dev:nightly_202605081558` image. Worth
   re-trying any "Kevin said this is broken" note.

## Branch commits

```
073580ea  gpt_oss: enable FLASH_ATTENTION_TRITON_AMD for prefill + decode
8919d900  gpt_oss: route ParallelLMHead BF16 GEMM through aiter Triton gemm_a16w16
6fa8c3a4  gpt_oss: document Stage 6 (paged-attn FlyDSL reduce) blocker — aiter NameError bug
f31f10a1  gpt_oss: route RMSNorm through aiter Triton via TritonRMSNorm subclass
43a2f624  branch-notes: (interim) Stages 2/3/4 working, Stage 1 blocked
ff581e3c  gpt_oss: route BF16 dense GEMM through aiter Triton gemm_a16w16
d6624197  gpt_oss: route MoE through aiter Triton (matmul_ogs)
d2b05475  gpt_oss/rmsnorm: Stage 1 partial — wire infra, dispatch blocked (superseded by f31f10a1)
5c6fd78d  branch-notes: stage 1 RMSNorm swap reproduces stash crash (historical)
0607082f  gpt_oss: route plain RMSNorm through aiter Triton kernels (Stage 1 v1, broken, superseded)
```

`origin/main` is `d41076e5`.

## What remains non-Triton

After Stage 8 the only kernels that still fire on non-Triton paths:

| µs | calls | kernel | why still non-Triton |
|---:|---:|---|---|
| 1,756 | 33 | `aiter::mix_sample_outer_exponential_kernel` | aiter has no Triton sampling kernel; only HIP `module_sample` |
| 251 | 24 | `ck_tile::FmhaFwdKernel` (residual) | Probably the few SWA prefill calls that the Triton MHA gates back to CK; might be an aiter dispatcher choice |
| 141 | 24 | `cp_mha_gather_cache_kernel` | **already Triton** (ATOM-internal `@triton.jit` in `base_attention.py:33`); classifier mislabels |
| 126 | 32 | `kv_indices_generate_kernel` | **already Triton** (ATOM-internal `@triton.jit` in `block_convert.py:143`); classifier mislabels |
| 2,011 | 1,976 | `_combined_routing_compute / memset / _topk_forward / _sum_bitmatrix_rows` | **already Triton** (from `triton_kernels` package); classifier mislabels |
| 91 | 24 | `rocprim::…scan_impl…trampoline_kernel` | rocPRIM scan called by `torch.cumsum`; aiter has no Triton replacement for this generic op |
| 65 | 24 | `compute_cuda_kernel<long>` | aten internal; not aiter-replaceable |
| ~133k | many | aten `elementwise_kernel` / `copy_kernel` / `fill_kernel` / `__amd_rocclr_copyBuffer` | PyTorch internal memcpy / fill / cast; not a kernel to "swap" |

**Truly non-Triton with no equivalent: ~1.9 k µs (0.04% of GPU time).**

Stage 6 (paged-attn decode reduce) is the one we'd unblock if aiter
fixes the FlyDSL `NameError`. That'd recover another ~2 k µs.

## Reproducing

```bash
cd /home/ibrawani/testing/atom-profiling
source scripts/env_ibrahimw1.sh
HIP_VISIBLE_DEVICES=2 bash scripts/profile_gpt_oss_20b_ours.sh
bash scripts/regen.sh
```

Container `rocm/atom-dev:nightly_202605081558`, mounts
`/home/ibrawani/dev/ATOM` (this branch) at `/app/ATOM`, single GPU TP=1.
Dashboard: `results/users/ibrahimw1/figures/dashboard.html`.

## Out of scope

- `aiter::mix_sample_outer_exponential_kernel`: AITER has no Triton
  sampling kernel anywhere. Would require new aiter dev.
- Stage 6 paged-attn reduce: aiter FlyDSL bug, file upstream.
- Classifier false-negatives (`_combined_routing_*`, `_topk_forward`,
  `_sum_bitmatrix_rows`, `cp_mha_gather_cache_kernel`,
  `kv_indices_generate_kernel`): live in `atom-profiling`'s
  `analysis/kernel_inventory.py`, not in this ATOM branch. Easy fix
  there: add the kernel name patterns to the "already" bucket.
- Other models: every per-instance flag in this branch defaults False
  and every os.environ patch is scoped to gpt_oss.py module import, so
  Llama / Qwen / MiniMax / DeepSeek etc. are unaffected.

## Stash

`stash@{0}: triton-swap-wip 2026-05-20` (your earlier env-var-gated
attempt) is preserved untouched. Drop with `git stash drop stash@{0}`
once you're satisfied this branch supersedes it.
