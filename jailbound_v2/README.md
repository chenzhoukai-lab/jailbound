# JailBound v2 Experimental Track

This folder is intentionally separate from the original `src/jailbound` package.
It is for iterating on the known gaps without overwriting the runnable baseline.

Implemented in v2:

- matched safe/unsafe prompt-pair probing
- expanded suffix-bank generation for text perturbation experiments
- HotFlip-style token replacement integrated into the v2 attack path
- raw-image perturbation/export utilities for moving beyond processor `pixel_values`
- lightweight experiment analysis by category

Run with both source roots on `PYTHONPATH`:

```bash
PYTHONPATH=$PWD/src:$PWD/jailbound_v2/src python -m jailbound_v2 --help
```

Example matched-pair probing:

```bash
PYTHONPATH=$PWD/src:$PWD/jailbound_v2/src accelerate launch --num_processes 4 --mixed_precision bf16 \
  -m jailbound_v2 probe --config jailbound_v2/configs/qwen25vl_v2.json --limit 200
```

Full v2 probe + attack + Qwen3Guard evaluation:

```bash
PYTHONPATH=$PWD/src:$PWD/jailbound_v2/src accelerate launch --num_processes 4 --mixed_precision bf16 \
  -m jailbound_v2 run --config jailbound_v2/configs/qwen25vl_v2.json --limit 20
```

The v2 attack now first selects a suffix candidate, locates its token span after
Qwen's chat template/image-token expansion, then applies HotFlip-style token
replacement before the visual `pixel_values` optimization. Each output row keeps
the initial suffix, final `adv_suffix`, token span metadata, and per-step token
replacement trace in:

```text
attack.initial_suffix
attack.hotflip_trace
```

If `boundary_probes_v2.pt` already exists, resume/continue the attack:

```bash
PYTHONPATH=$PWD/src:$PWD/jailbound_v2/src accelerate launch --num_processes 4 --mixed_precision bf16 \
  -m jailbound_v2 run --config jailbound_v2/configs/qwen25vl_v2.json --resume
```

Run two independent 2xH100 instances without conflicts by using different
dataset splits and output suffixes. If you already have a full v2 boundary file,
pass it to both jobs with `--boundary`:

```bash
# instance A
PYTHONPATH=$PWD/src:$PWD/jailbound_v2/src accelerate launch --num_processes 2 --mixed_precision bf16 \
  -m jailbound_v2 run --config jailbound_v2/configs/qwen25vl_v2.json \
  --boundary outputs/qwen25vl_jailbound_v2/boundary_probes_v2.pt \
  --num-splits 2 --split-index 0 --output-suffix split0 --resume

# instance B
PYTHONPATH=$PWD/src:$PWD/jailbound_v2/src accelerate launch --num_processes 2 --mixed_precision bf16 \
  -m jailbound_v2 run --config jailbound_v2/configs/qwen25vl_v2.json \
  --boundary outputs/qwen25vl_jailbound_v2/boundary_probes_v2.pt \
  --num-splits 2 --split-index 1 --output-suffix split1 --resume
```

Analyze existing baseline outputs:

```bash
PYTHONPATH=$PWD/src:$PWD/jailbound_v2/src python -m jailbound_v2 analyze \
  --input outputs/qwen25vl_jailbound/guard_eval.jsonl \
  --output outputs/qwen25vl_jailbound_v2/analysis
```

Evaluate v2 outputs with the looser paper-style metrics:

```bash
# Merge two split jobs first.
mkdir -p outputs/qwen25vl_jailbound_v2_merged
cat outputs/qwen25vl_jailbound_v2_split0/attack_results.jsonl \
    outputs/qwen25vl_jailbound_v2_split1/attack_results.jsonl \
    > outputs/qwen25vl_jailbound_v2_merged/attack_results.jsonl
cat outputs/qwen25vl_jailbound_v2_split0/guard_eval.jsonl \
    outputs/qwen25vl_jailbound_v2_split1/guard_eval.jsonl \
    > outputs/qwen25vl_jailbound_v2_merged/guard_eval.jsonl

# Non-refusal ASR, matching the paper's looser non-refusal framing.
PYTHONPATH=$PWD/src:$PWD/jailbound_v2/src python -m jailbound_v2 loose-eval \
  --config jailbound_v2/configs/qwen25vl_v2.json \
  --attack-results outputs/qwen25vl_jailbound_v2_merged/attack_results.jsonl \
  --guard-eval outputs/qwen25vl_jailbound_v2_merged/guard_eval.jsonl \
  --output-suffix merged

# Qwen2.5 follow judge ASR. Uses follow_judge_model_path from the base config.
PYTHONPATH=$PWD/src:$PWD/jailbound_v2/src accelerate launch --num_processes 2 --mixed_precision bf16 \
  -m jailbound_v2 follow-eval \
  --config jailbound_v2/configs/qwen25vl_v2.json \
  --attack-results outputs/qwen25vl_jailbound_v2_merged/attack_results.jsonl \
  --guard-eval outputs/qwen25vl_jailbound_v2_merged/guard_eval.jsonl \
  --output-suffix merged \
  --mode both
```
