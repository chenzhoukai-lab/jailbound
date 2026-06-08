# JailBound v2 Experimental Track

This folder is intentionally separate from the original `src/jailbound` package.
It is for iterating on the known gaps without overwriting the runnable baseline.

Implemented in v2:

- matched safe/unsafe prompt-pair probing
- expanded suffix-bank generation for text perturbation experiments
- HotFlip-style token replacement utilities for future integration
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

If `boundary_probes_v2.pt` already exists, resume/continue the attack:

```bash
PYTHONPATH=$PWD/src:$PWD/jailbound_v2/src accelerate launch --num_processes 4 --mixed_precision bf16 \
  -m jailbound_v2 run --config jailbound_v2/configs/qwen25vl_v2.json --resume
```

Analyze existing baseline outputs:

```bash
PYTHONPATH=$PWD/src:$PWD/jailbound_v2/src python -m jailbound_v2 analyze \
  --input outputs/qwen25vl_jailbound/guard_eval.jsonl \
  --output outputs/qwen25vl_jailbound_v2/analysis
```
