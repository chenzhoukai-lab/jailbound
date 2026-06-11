# JailBound v5: Prompt / Crossing / Task-Token Attribution

`jailbound_v5` is a separate experimental track. It does not overwrite the
baseline, v2, v3, or v4 code.

The goal is to test three changes with low compute cost:

1. **Prompt boundary**: generate a safe counterpart for each unsafe prompt with a local model.
2. **Boundary crossing strength**: use layer 19-24 and `epsilon_scale`.
3. **Suffix tokens**: keep HotFlip readable while biasing it toward task-following tokens.

Recommended first run:

```text
Financial_Advice, 100 samples
```

## Setup

```bash
export PYTHONPATH=$PWD/src:$PWD/jailbound_v5/src
```

## Step 1: Generate Safe Prompt Cache

Run this once before the four experiments. It uses
`follow_judge_model_path` from `configs/qwen25vl_local.json` by default.

```bash
accelerate launch --num_processes 4 --mixed_precision bf16 \
  -m jailbound_v5 generate-safe-prompts \
  --config jailbound_v5/configs/qwen25vl_v5.json \
  --categories Financial_Advice \
  --per-category-limit 100 \
  --output outputs/qwen25vl_jailbound_v5/safe_prompts_v5.jsonl
```

The cache file can be manually inspected before probing:

```bash
head -n 5 outputs/qwen25vl_jailbound_v5/safe_prompts_v5.jsonl
```

## Step 2: Four 4xH100 Ablation Runs

Run each command in a separate 4xH100 instance.

### A. Baseline Reference

Old fixed-safe boundary, suffix bank, original crossing strength.

```bash
accelerate launch --num_processes 4 --mixed_precision bf16 \
  -m jailbound_v5 run \
  --config jailbound_v5/configs/qwen25vl_v5.json \
  --categories Financial_Advice \
  --per-category-limit 100 \
  --boundary-mode baseline_fixed_safe \
  --suffix-mode suffix_bank \
  --epsilon-scale 1.0 \
  --crossing-layers all \
  --output-suffix a0_baseline \
  --resume
```

### B. Generated Prompt Boundary

Only replace the fixed-safe boundary with generated safe counterparts.

```bash
accelerate launch --num_processes 4 --mixed_precision bf16 \
  -m jailbound_v5 run \
  --config jailbound_v5/configs/qwen25vl_v5.json \
  --categories Financial_Advice \
  --per-category-limit 100 \
  --boundary-mode matched_safe_pair \
  --suffix-mode suffix_bank \
  --epsilon-scale 1.0 \
  --crossing-layers all \
  --safe-prompt-cache outputs/qwen25vl_jailbound_v5/safe_prompts_v5.jsonl \
  --output-suffix a1_generated_prompt \
  --resume
```

### C. Generated Prompt + Stronger Crossing

Keep generated boundary, then use layer 19-24 and stronger epsilon.

```bash
accelerate launch --num_processes 4 --mixed_precision bf16 \
  -m jailbound_v5 run \
  --config jailbound_v5/configs/qwen25vl_v5.json \
  --categories Financial_Advice \
  --per-category-limit 100 \
  --boundary-mode matched_safe_pair \
  --suffix-mode suffix_bank \
  --epsilon-scale 1.5 \
  --crossing-layers 19 20 21 22 23 24 \
  --safe-prompt-cache outputs/qwen25vl_jailbound_v5/safe_prompts_v5.jsonl \
  --output-suffix a2_prompt_cross \
  --resume
```

### D. Generated Prompt + Stronger Crossing + Task HotFlip

Add readable task-token HotFlip on top of C.

```bash
accelerate launch --num_processes 4 --mixed_precision bf16 \
  -m jailbound_v5 run \
  --config jailbound_v5/configs/qwen25vl_v5.json \
  --categories Financial_Advice \
  --per-category-limit 100 \
  --boundary-mode matched_safe_pair \
  --suffix-mode task_hotflip \
  --epsilon-scale 1.5 \
  --crossing-layers 19 20 21 22 23 24 \
  --safe-prompt-cache outputs/qwen25vl_jailbound_v5/safe_prompts_v5.jsonl \
  --output-suffix a3_prompt_cross_task \
  --resume
```

## Step 3: Extra Evaluations

`run` already does Qwen3Guard evaluation. After the four runs finish, run:

```bash
for name in a0_baseline a1_generated_prompt a2_prompt_cross a3_prompt_cross_task; do
  accelerate launch --num_processes 4 --mixed_precision bf16 \
    -m jailbound_v5 follow-eval \
    --config jailbound_v5/configs/qwen25vl_v5.json \
    --attack-results outputs/qwen25vl_jailbound_v5_${name}/attack_results.jsonl \
    --guard-eval outputs/qwen25vl_jailbound_v5_${name}/guard_eval.jsonl \
    --output-suffix ${name} \
    --mode both
done
```

Non-refusal ASR is lightweight:

```bash
for name in a0_baseline a1_generated_prompt a2_prompt_cross a3_prompt_cross_task; do
  python -m jailbound_v5 loose-eval \
    --config jailbound_v5/configs/qwen25vl_v5.json \
    --attack-results outputs/qwen25vl_jailbound_v5_${name}/attack_results.jsonl \
    --guard-eval outputs/qwen25vl_jailbound_v5_${name}/guard_eval.jsonl \
    --output-suffix ${name}
done
```

## Step 4: Unified Report

```bash
python -m jailbound_v5 report \
  --run a0_baseline=outputs/qwen25vl_jailbound_v5_a0_baseline \
  --run a1_generated_prompt=outputs/qwen25vl_jailbound_v5_a1_generated_prompt \
  --run a2_prompt_cross=outputs/qwen25vl_jailbound_v5_a2_prompt_cross \
  --run a3_prompt_cross_task=outputs/qwen25vl_jailbound_v5_a3_prompt_cross_task \
  --output outputs/qwen25vl_jailbound_v5_financial_ablation_report
```

Interpretation:

```text
a1 - a0 = generated prompt boundary effect
a2 - a1 = crossing layer/epsilon effect
a3 - a2 = task-token readable HotFlip effect
```

## Boundary Comparison

```bash
python -m jailbound_v5 compare-boundaries \
  --left outputs/qwen25vl_jailbound_v5_a0_baseline/boundary_probes.pt \
  --right outputs/qwen25vl_jailbound_v5_a1_generated_prompt/boundary_probes_v5.pt \
  --output outputs/qwen25vl_jailbound_v5_financial_ablation_report/boundary_cosine.json
```
