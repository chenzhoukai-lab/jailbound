# JailBound v4 Boundary/Suffix Attribution Track

This folder is intentionally separate from `src/jailbound`, `jailbound_v2`,
and `jailbound_v3`.

v4 is for low-cost attribution experiments. The goal is not to rerun all 1680
samples each time, but to answer one question:

```text
Did ASR drop because the safety boundary changed, or because suffix optimization changed?
```

## Experiment Design

v4 exposes two independent factors:

| Factor | Option A | Option B |
|---|---|---|
| Boundary | `baseline_fixed_safe` | `matched_safe_pair` |
| Suffix | `suffix_bank` | `readable_hotflip` |

This gives a 2x2 experiment:

| Run | Boundary | Suffix | Purpose |
|---|---|---|---|
| `b0_baseline_suffix` | baseline fixed safe prompt | suffix bank | baseline small-run reference |
| `b1_matched_suffix` | matched safe/unsafe pairs | suffix bank | boundary-only effect |
| `h0_baseline_hotflip` | baseline fixed safe prompt | readable HotFlip | suffix-only effect under old boundary |
| `h1_matched_hotflip` | matched safe/unsafe pairs | readable HotFlip | combined v4 effect |

Recommended first category:

```text
Financial_Advice, 100 samples
```

It is useful because previous full results showed high Qwen2.5 judge ASR in
this category, so small changes in task-following are easier to observe.

## Common Setup

```bash
git pull
export PYTHONPATH=$PWD/src:$PWD/jailbound_v4/src
```

## Four Parallel 4xH100 Runs

Run each command in a separate 4xH100 instance.

### A. Baseline Boundary + Suffix Bank

```bash
accelerate launch --num_processes 4 --mixed_precision bf16 \
  -m jailbound_v4 run \
  --config jailbound_v4/configs/qwen25vl_v4.json \
  --categories Financial_Advice \
  --per-category-limit 100 \
  --boundary-mode baseline_fixed_safe \
  --suffix-mode suffix_bank \
  --output-suffix b0_baseline_suffix \
  --resume
```

### B. Matched Boundary + Suffix Bank

```bash
accelerate launch --num_processes 4 --mixed_precision bf16 \
  -m jailbound_v4 run \
  --config jailbound_v4/configs/qwen25vl_v4.json \
  --categories Financial_Advice \
  --per-category-limit 100 \
  --boundary-mode matched_safe_pair \
  --suffix-mode suffix_bank \
  --output-suffix b1_matched_suffix \
  --resume
```

### C. Baseline Boundary + Readable HotFlip

```bash
accelerate launch --num_processes 4 --mixed_precision bf16 \
  -m jailbound_v4 run \
  --config jailbound_v4/configs/qwen25vl_v4.json \
  --categories Financial_Advice \
  --per-category-limit 100 \
  --boundary-mode baseline_fixed_safe \
  --suffix-mode readable_hotflip \
  --output-suffix h0_baseline_hotflip \
  --resume
```

### D. Matched Boundary + Readable HotFlip

```bash
accelerate launch --num_processes 4 --mixed_precision bf16 \
  -m jailbound_v4 run \
  --config jailbound_v4/configs/qwen25vl_v4.json \
  --categories Financial_Advice \
  --per-category-limit 100 \
  --boundary-mode matched_safe_pair \
  --suffix-mode readable_hotflip \
  --output-suffix h1_matched_hotflip \
  --resume
```

Outputs:

```text
outputs/qwen25vl_jailbound_v4_b0_baseline_suffix
outputs/qwen25vl_jailbound_v4_b1_matched_suffix
outputs/qwen25vl_jailbound_v4_h0_baseline_hotflip
outputs/qwen25vl_jailbound_v4_h1_matched_hotflip
```

## Extra Evaluations

`run` already performs Qwen3Guard evaluation. After the four runs finish, run
Qwen2.5 judge ASR for each output directory:

```bash
for name in b0_baseline_suffix b1_matched_suffix h0_baseline_hotflip h1_matched_hotflip; do
  accelerate launch --num_processes 4 --mixed_precision bf16 \
    -m jailbound_v4 follow-eval \
    --config jailbound_v4/configs/qwen25vl_v4.json \
    --attack-results outputs/qwen25vl_jailbound_v4_${name}/attack_results.jsonl \
    --guard-eval outputs/qwen25vl_jailbound_v4_${name}/guard_eval.jsonl \
    --output-suffix ${name} \
    --mode both
done
```

Non-refusal ASR is fast and does not load a model:

```bash
for name in b0_baseline_suffix b1_matched_suffix h0_baseline_hotflip h1_matched_hotflip; do
  python -m jailbound_v4 loose-eval \
    --config jailbound_v4/configs/qwen25vl_v4.json \
    --attack-results outputs/qwen25vl_jailbound_v4_${name}/attack_results.jsonl \
    --guard-eval outputs/qwen25vl_jailbound_v4_${name}/guard_eval.jsonl \
    --output-suffix ${name}
done
```

## Unified Report

```bash
python -m jailbound_v4 report \
  --run b0_baseline_suffix=outputs/qwen25vl_jailbound_v4_b0_baseline_suffix \
  --run b1_matched_suffix=outputs/qwen25vl_jailbound_v4_b1_matched_suffix \
  --run h0_baseline_hotflip=outputs/qwen25vl_jailbound_v4_h0_baseline_hotflip \
  --run h1_matched_hotflip=outputs/qwen25vl_jailbound_v4_h1_matched_hotflip \
  --output outputs/qwen25vl_jailbound_v4_financial_ablation_report
```

Key interpretation:

```text
b1 - b0 = boundary effect with suffix fixed
h0 - b0 = suffix effect with boundary fixed
h1 - b1 = suffix effect under matched boundary
h1 - h0 = boundary effect under readable HotFlip
```

## Boundary Comparison

Compare old and matched boundary directions:

```bash
python -m jailbound_v4 compare-boundaries \
  --left outputs/qwen25vl_jailbound_v4_b0_baseline_suffix/boundary_probes.pt \
  --right outputs/qwen25vl_jailbound_v4_b1_matched_suffix/boundary_probes_v4.pt \
  --output outputs/qwen25vl_jailbound_v4_financial_ablation_report/boundary_cosine.json
```

If ASR drops from `b0` to `b1` and boundary cosine is low, the ASR drop is likely
caused by boundary construction. If `b0` and `b1` are close but HotFlip runs
change ASR, the suffix optimizer is the main driver.

