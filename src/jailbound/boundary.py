from __future__ import annotations

import math
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .dataset import SafetySample
from .distributed import get_accelerator, reset_shard_dir, runtime_device, shard_items
from .modeling_qwen import Qwen25VL


def resolve_layers(selector: str | list[int], n_layers: int) -> list[int]:
    # 把配置里的 "all" / "last_10" / "1,2,3" 统一转成层号列表。
    if isinstance(selector, list):
        return [int(x) for x in selector]
    if selector == "all":
        return list(range(n_layers))
    if selector.startswith("last_"):
        n = int(selector.split("_", 1)[1])
        return list(range(max(0, n_layers - n), n_layers))
    return [int(x.strip()) for x in selector.split(",") if x.strip()]


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def train_logistic_probe(features: np.ndarray, labels: np.ndarray, cfg: Config) -> dict[str, Any]:
    # 论文 Algorithm 1 的核心：对每层 h(l) 训练一个线性安全分类器。
    # 分类器形式是 sigmoid(w^T h + b)，w 的单位方向就是边界法向量 v。
    import torch

    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    x = (x - mean) / std

    w = torch.zeros(x.shape[1], dtype=torch.float32, requires_grad=True)
    b = torch.zeros((), dtype=torch.float32, requires_grad=True)
    opt = torch.optim.AdamW([w, b], lr=cfg.boundary.learning_rate, weight_decay=cfg.boundary.weight_decay)

    for _ in range(cfg.boundary.epochs):
        logits = x @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        # epsilon 是样本到目标阈值 P0 对应超平面的平均距离。
        # 后续攻击会沿着 v 的方向移动大约 epsilon，模拟“跨过内部安全边界”。
        logits = x @ w + b
        probs = torch.sigmoid(logits)
        acc = ((probs >= 0.5).float() == y).float().mean().item()
        norm = torch.linalg.vector_norm(w).clamp_min(1e-6)
        eps = torch.mean(torch.abs(_logit(cfg.boundary.p0) - logits) / norm).item()
        v = (w / norm).detach().cpu().numpy()

    return {
        "w": w.detach().cpu().numpy(),
        "b": float(b.detach().cpu()),
        "mean": mean.squeeze(0).cpu().numpy(),
        "std": std.squeeze(0).cpu().numpy(),
        "v": v,
        "epsilon": eps,
        "accuracy": acc,
    }


def collect_probe_matrix(model: Qwen25VL, samples: list[SafetySample], cfg: Config) -> tuple[dict[int, list[np.ndarray]], np.ndarray]:
    # 对每条 unsafe 样本额外构造一条 safe 对照：
    # unsafe = 原图 + 原始 MM-SafetyBench prompt
    # safe   = 原图 + safe_prompt
    # 这样每层都有 [unsafe, safe, unsafe, safe, ...] 的训练数据。
    layer_features: dict[int, list[np.ndarray]] = {}
    labels: list[int] = []
    for sample in samples:
        unsafe = model.hidden_features(sample.image_path, sample.prompt, cfg.boundary.pooling)
        safe = model.hidden_features(sample.image_path, cfg.boundary.safe_prompt, cfg.boundary.pooling)
        layers = resolve_layers(cfg.boundary.layers, len(unsafe))
        for layer in layers:
            layer_features.setdefault(layer, []).append(unsafe[layer].numpy())
            layer_features.setdefault(layer, []).append(safe[layer].numpy())
        labels.extend([1, 0])
    return layer_features, np.asarray(labels, dtype=np.float32)


def probe_boundaries(cfg: Config, samples: list[SafetySample]) -> Path:
    # 完整的 Safety Boundary Probing 阶段：
    # 1. 加载 Qwen2.5-VL；
    # 2. 提取各层 hidden states；
    # 3. 每层训练 logistic probe；
    # 4. 保存 w/b/v/epsilon，供攻击阶段读取。
    accelerator = get_accelerator()
    cfg.validate_paths()
    torch = __import__("torch")
    out = cfg.output_path / "boundary_probes.pt"
    shard_dir = cfg.output_path / "_probe_shards"
    reset_shard_dir(shard_dir, accelerator)

    local_samples = shard_items(samples, accelerator)
    if local_samples:
        model = Qwen25VL(
            cfg.target_model_path,
            runtime_device(cfg, accelerator),
            cfg.torch_dtype,
            cfg.attn_implementation,
        )
        layer_features, labels = collect_probe_matrix(model, local_samples, cfg)
    else:
        layer_features, labels = {}, np.asarray([], dtype=np.float32)

    shard_payload: dict[str, Any] = {"labels": labels, "layers": np.asarray(sorted(layer_features), dtype=np.int64)}
    for layer, feats in sorted(layer_features.items()):
        shard_payload[f"layer_{layer}"] = np.stack(feats)
    np.savez_compressed(shard_dir / f"rank_{accelerator.process_index}.npz", **shard_payload)
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        merged_features: dict[int, list[np.ndarray]] = {}
        merged_labels: list[np.ndarray] = []
        for shard_path in sorted(shard_dir.glob("rank_*.npz")):
            with np.load(shard_path) as shard:
                if shard["labels"].size == 0:
                    continue
                merged_labels.append(shard["labels"])
                for layer in shard["layers"].tolist():
                    merged_features.setdefault(int(layer), []).append(shard[f"layer_{int(layer)}"])
        if not merged_labels:
            raise ValueError("No samples were available for boundary probing.")
        labels = np.concatenate(merged_labels, axis=0)
        probes = {}
        for layer, shard_arrays in sorted(merged_features.items()):
            feats = np.concatenate(shard_arrays, axis=0)
            probes[layer] = train_logistic_probe(feats, labels, cfg)
            print(f"layer={layer} probe_acc={probes[layer]['accuracy']:.4f} epsilon={probes[layer]['epsilon']:.4f}")

        torch.save(
            {
                "config": asdict(cfg),
                "layers": sorted(probes),
                "probes": probes,
                "num_samples": len(samples),
                "num_processes": accelerator.num_processes,
                "shards": [p.name for p in sorted(shard_dir.glob("rank_*.npz"))],
                "label_note": "unsafe=MM-SafetyBench original prompt, safe=config.boundary.safe_prompt on same image",
            },
            out,
        )
        metadata = {
            "boundary_path": str(out),
            "num_processes": accelerator.num_processes,
            "num_samples": len(samples),
            "layers": sorted(probes),
        }
        (cfg.output_path / "boundary_probes_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    accelerator.wait_for_everyone()
    return out
