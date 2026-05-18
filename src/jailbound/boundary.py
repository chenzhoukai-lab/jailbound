from __future__ import annotations

import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .dataset import SafetySample
from .modeling_qwen import Qwen25VL


def resolve_layers(selector: str | list[int], n_layers: int) -> list[int]:
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
    cfg.validate_paths()
    torch = __import__("torch")
    model = Qwen25VL(cfg.target_model_path, cfg.device, cfg.torch_dtype, cfg.attn_implementation)
    layer_features, labels = collect_probe_matrix(model, samples, cfg)
    probes = {}
    for layer, feats in sorted(layer_features.items()):
        probes[layer] = train_logistic_probe(np.stack(feats), labels, cfg)
        print(f"layer={layer} probe_acc={probes[layer]['accuracy']:.4f} epsilon={probes[layer]['epsilon']:.4f}")

    out = cfg.output_path / "boundary_probes.pt"
    torch.save(
        {
            "config": asdict(cfg),
            "layers": sorted(probes),
            "probes": probes,
            "num_samples": len(samples),
            "label_note": "unsafe=MM-SafetyBench original prompt, safe=config.boundary.safe_prompt on same image",
        },
        out,
    )
    return out

