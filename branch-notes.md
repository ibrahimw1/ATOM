# branch: `ibrahim/triton-swap-gpt-oss-20b`

Status: **HALTED at Stage 1**. RMSNorm Triton swap alone reproduces the
same GPU memory access fault seen with the earlier stashed Triton swap.

## What was attempted

- Stage 0: branched off `origin/main` (`d41076e5`). Clean.
- **Stage 1**: in `atom/model_ops/layernorm.py`, added `use_triton: bool=False`
  kwarg to `RMSNorm.__init__`; when set, the unquantized BF16 forward path
  dispatches to `aiter.ops.triton.normalization.rmsnorm.{rms_norm,
  rmsnorm2d_fwd_with_add}` (with `dim <= 8192` cap, same as the stash). In
  `atom/models/gpt_oss.py`, passed `use_triton=True` at all three RMSNorm
  sites (input/post-attn/final). Commit: `0607082f`.

Stages 2–4 not attempted (will short-circuit while Stage 1 is broken).

## Reproduction

```bash
cd /home/ibrawani/testing/atom-profiling
source scripts/env_ibrahimw1.sh
HIP_VISIBLE_DEVICES=2 bash scripts/profile_gpt_oss_20b_ours.sh
```

Container: `rocm/atom-dev:nightly_202605081558` with this branch mounted
at `/app/ATOM`, aiter HEAD at `/app/aiter-test`, single GPU TP=1.

## Failure mode (verbatim from `/tmp/stage1_run.log:30-40`)

```
[atom 21:30:45] Model Runner0/1: DP size=1 too large, warmup_max_tokens=16384
                < max_model_len=131072. Using 1 seq with length 16384 for warmup.
[aiter] [fused_moe] duplicate tuned rows after disabling act_type in
        /tmp/aiter_configs/tuned_fmoe.csv; keeping first match for 431 rows
[aiter] [fused_moe] no tuned FlyDSL config for (256, 16384, 3072, 3072, 32, 4,
        <ActivationType.Swiglu: 2>, 'torch.bfloat16', 'torch.float4_e2m1fn_x2',
        'torch.float4_e2m1fn_x2', 'QuantType.per_1x32', True, False),
        using heuristic FlyDSL fallback (kn1='flydsl_moe1_afp4_wfp4_bf16_t64x128x256_w4_bnt0',
        kn2='flydsl_moe2_afp4_wfp4_bf16_t64x128x256_atomic')
[aiter] type hints mismatch, override to --> moe_sorting_opus_fwd(...)
Memory access fault by GPU node-4 (Agent handle: 0x3a78fc60)
        on address 0x7ebc70a65000. Reason: Unknown.
GPU coredump: handler exited with error (status: 1)
[atom 21:31:24] AsyncIOProcManager(ModelRunner): [ModelRunner0/1]
                proc died unexpectedly (exitcode=-6), shutting down.
```

Crash signature is byte-for-byte identical to the earlier stash-combo run
(`stash@{0}: triton-swap-wip 2026-05-20`, which had RMSNorm + BF16 Linear
both swapped). On stock `main` with no swap, the same model + same command
runs to completion (see `traces/gpt_oss_20b_main/default.json.gz`, 526 KB,
fresh).

## Hypothesis

The fault site is the **AITER MoE FlyDSL heuristic fallback** during the
first decode warmup, not in our Triton RMSNorm code. But our swap is the
trigger: stock `main` doesn't crash here.

Most plausible chain:

1. Our `aiter.ops.triton.normalization.rmsnorm.rms_norm` output for
   `(BS=4, ISL=128 → 16384-token warmup, hidden=2880, bf16)` may differ
   in stride / memory layout / contiguity from the aiter HIP
   `rmsnorm2d_fwd` it replaces.
2. The MoE kernel reads its input tensor's stride metadata strictly. With
   an unexpected stride, the FlyDSL fallback (already in "no tuned config,
   using heuristic" mode) hands a bad pointer to `moe_sorting_opus_fwd`.
3. `Memory access fault by GPU node-4` is the consequence — the kernel
   dereferenced an out-of-bounds address.

Alternative theory: the Triton RMSNorm output dtype/scale handling for
gpt-oss's swiglu-MoE chain differs subtly from what AITER MoE expects.
Without a debugger or numerical comparison this is conjecture; the
correlation (swap on → crash, swap off → runs) is the only solid signal.

## What didn't fix it

- Capping the Triton path at `dim <= 8192` (2880 is well under).
- Cell isolation (default cell with `unset ATOM_USE_TRITON_*` still
  crashes — confirms the trigger is the code change, not any env var).

## Things to try next (none attempted on this branch)

1. **Force contiguous output**: wrap the Triton RMSNorm calls with
   `.contiguous()` before returning, to match HIP's contract.
2. **Numerical diff**: capture the RMSNorm output for one layer using both
   paths (call the Triton variant on a side branch, store the HIP output,
   compare numerically). If outputs diverge in stride / layout, that's the
   smoking gun.
3. **Bisect the dispatch sites**: enable Triton on ONLY one of the three
   RMSNorm sites (e.g., final norm only — least connected to MoE) to see
   if any specific site is the culprit. The post_attention_layernorm has
   `x_pad_to_multiple=256` at TP=1 which routes through a DIFFERENT path
   (already Triton via `fused_rmsnorm_pad_`) so our flag is a no-op
   there for TP=1; the suspect sites are `input_layernorm` and `norm`.
4. **Pre-compile via warmup**: AITER's FlyDSL warning about
   "no tuned config, heuristic fallback" suggests the MoE tuned CSV
   (`/tmp/aiter_configs/tuned_fmoe.csv`) lacks an entry for
   `(256, 16384, 3072, 3072, 32, 4, ...)`. Even on stock main this
   warning fires but doesn't crash, so the heuristic kernel works on
   stock inputs — only our swap inputs trigger the fault.
5. **Run with `AMD_LOG_LEVEL=3 HSA_ENABLE_DEBUG=1`** to get more detail
   on the faulting kernel and address.

## Branch state at halt

- HEAD `0607082f` — Stage 1 commit (intentionally NOT reverted; preserves
  the exact failing change for diagnostic / future fix-forward).
- `git stash list` shows `stash@{0}: triton-swap-wip 2026-05-20`
  preserved untouched (Ibrahim's earlier attempt).

## Recommendation

This is a real ATOM/AITER integration bug on MI355X (gfx950), not just a
"we typed something wrong" problem. The Triton RMSNorm output is
incompatible with downstream consumers in a way that's only visible at
runtime in the MoE path. Need numerical-diff and stride-inspection
debugging before any of stages 2–4 are tried; otherwise we'd just
re-discover the same fault in a different guise.
