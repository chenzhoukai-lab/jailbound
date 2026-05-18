# JailBound Reproduction for Local Qwen2.5-VL

This workspace contains a practical reproduction scaffold for **JailBound: Jailbreaking Internal Safety Boundaries of Vision-Language Models** using:

- local MM-SafetyBench data
- local `Qwen2.5-VL-7B-Instruct`
- local `Qwen3Guard` as the ASR / attack-effectiveness judge

The implementation follows the paper's two-stage pipeline:

1. **Safety Boundary Probing**: collect layer-wise fused hidden states on safe/unsafe prompt pairs and train logistic boundary probes.
2. **Safety Boundary Crossing**: optimize a multimodal adversarial input against the learned boundary objectives, then evaluate outputs with Qwen3Guard.

## Quick Start

Edit paths in `configs/qwen25vl_local.json`:

```json
{
  "dataset_root": "D:/path/to/mm-safetybench",
  "target_model_path": "D:/models/Qwen2.5-VL-7B-Instruct",
  "guard_model_path": "D:/models/Qwen3Guard-Gen-8B"
}
```

Run all stages:

```powershell
python -m jailbound run --config configs/qwen25vl_local.json --limit 20
```

Or run them separately:

```powershell
python -m jailbound probe --config configs/qwen25vl_local.json --limit 100
python -m jailbound attack --config configs/qwen25vl_local.json --limit 20
python -m jailbound eval --config configs/qwen25vl_local.json
```

If you use a `src` layout without installing the package, run from this repo root with:

```powershell
$env:PYTHONPATH = "$PWD/src"
python -m jailbound run --config configs/qwen25vl_local.json --limit 20
```

## Outputs

Default outputs are written under `outputs/qwen25vl_jailbound/`:

- `boundary_probes.pt`: layer-wise logistic probe weights and metadata
- `attack_results.jsonl`: per-sample model outputs and attack metadata
- `guard_eval.jsonl`: Qwen3Guard judgments per sample
- `summary.json`: ASR and attack-effectiveness summary

## Notes

Qwen2.5-VL model internals differ from the abstract notation in the paper. This implementation treats decoder hidden states after multimodal token fusion as the fusion-layer representations `h(l)`, which is the accessible equivalent for Hugging Face Qwen2.5-VL.

