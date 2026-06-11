from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from jailbound.boundary import resolve_layers, train_logistic_probe
from jailbound.config import Config
from jailbound.dataset import SafetySample
from jailbound.distributed import get_accelerator, reset_shard_dir, runtime_device, shard_items
from jailbound.modeling_qwen import Qwen25VL

from .prompt_pairs import build_prompt_pairs


def collect_matched_probe_matrix(
    model: Qwen25VL,
    samples: list[SafetySample],
    cfg: Config,
    safe_modes: list[str],
    unsafe_modes: list[str],
) -> tuple[dict[int, list[np.ndarray]], np.ndarray, list[dict[str, Any]]]:
    layer_features: dict[int, list[np.ndarray]] = {}
    labels: list[int] = []
    records: list[dict[str, Any]] = []
    for sample in samples:
        pairs = build_prompt_pairs(sample, safe_modes=safe_modes, unsafe_modes=unsafe_modes)
        for pair in pairs:
            features = model.hidden_features(sample.image_path, pair.prompt, cfg.boundary.pooling)
            layers = resolve_layers(cfg.boundary.layers, len(features))
            for layer in layers:
                layer_features.setdefault(layer, []).append(features[layer].numpy())
            labels.append(pair.label)
            records.append({"sample_id": sample.sample_id, "category": sample.category, "kind": pair.kind, "label": pair.label})
    return layer_features, np.asarray(labels, dtype=np.float32), records


def probe_boundaries_v4(
    cfg: Config,
    samples: list[SafetySample],
    safe_modes: list[str],
    unsafe_modes: list[str],
) -> Path:
    accelerator = get_accelerator()
    cfg.validate_paths()
    torch = __import__("torch")
    out = cfg.output_path / "boundary_probes_v4.pt"
    shard_dir = cfg.output_path / "_probe_v4_shards"
    reset_shard_dir(shard_dir, accelerator)

    local_samples = shard_items(samples, accelerator)
    if local_samples:
        model = Qwen25VL(cfg.target_model_path, runtime_device(cfg, accelerator), cfg.torch_dtype, cfg.attn_implementation)
        layer_features, labels, records = collect_matched_probe_matrix(model, local_samples, cfg, safe_modes, unsafe_modes)
    else:
        layer_features, labels, records = {}, np.asarray([], dtype=np.float32), []

    payload: dict[str, Any] = {
        "labels": labels,
        "layers": np.asarray(sorted(layer_features), dtype=np.int64),
        "records": np.asarray([json.dumps(x, ensure_ascii=False) for x in records]),
    }
    for layer, feats in sorted(layer_features.items()):
        payload[f"layer_{layer}"] = np.stack(feats)
    np.savez_compressed(shard_dir / f"rank_{accelerator.process_index}.npz", **payload)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        merged_features: dict[int, list[np.ndarray]] = {}
        merged_labels: list[np.ndarray] = []
        merged_records: list[dict[str, Any]] = []
        for shard_path in sorted(shard_dir.glob("rank_*.npz")):
            with np.load(shard_path) as shard:
                if shard["labels"].size == 0:
                    continue
                merged_labels.append(shard["labels"])
                merged_records.extend(json.loads(x) for x in shard["records"].tolist())
                for layer in shard["layers"].tolist():
                    merged_features.setdefault(int(layer), []).append(shard[f"layer_{int(layer)}"])
        labels = np.concatenate(merged_labels, axis=0)
        probes = {}
        for layer, arrays in sorted(merged_features.items()):
            probes[layer] = train_logistic_probe(np.concatenate(arrays, axis=0), labels, cfg)
            print(f"[v4 probe] layer={layer} acc={probes[layer]['accuracy']:.4f} eps={probes[layer]['epsilon']:.4f}")

        torch.save(
            {
                "config": asdict(cfg),
                "safe_modes": safe_modes,
                "unsafe_modes": unsafe_modes,
                "layers": sorted(probes),
                "probes": probes,
                "num_samples": len(samples),
                "num_prompt_records": len(merged_records),
                "records": merged_records,
                "note": "v4 uses matched safe prompts derived from each harmful instruction",
            },
            out,
        )
        (cfg.output_path / "boundary_probes_v4_meta.json").write_text(
            json.dumps(
                {
                    "boundary_path": str(out),
                    "safe_modes": safe_modes,
                    "unsafe_modes": unsafe_modes,
                    "num_samples": len(samples),
                    "num_prompt_records": len(merged_records),
                    "layers": sorted(probes),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    accelerator.wait_for_everyone()
    return out




