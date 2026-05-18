from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .dataset import SafetySample
from .modeling_qwen import Qwen25VL


def _tensor_probe(torch, probe: dict[str, Any], device):
    return {
        "v": torch.tensor(probe["v"], dtype=torch.float32, device=device),
        "epsilon": float(probe["epsilon"]),
    }


def _select_suffix(model: Qwen25VL, sample: SafetySample, base_prompt: str, suffixes: list[str], probes, cfg: Config) -> str:
    if not suffixes:
        return cfg.attack.suffix
    best_suffix = suffixes[0]
    best_score = float("inf")
    with model.torch.no_grad():
        original = model.pooled_hidden(model.forward_hidden(sample.image_path, base_prompt), cfg.boundary.pooling)
        for suffix in suffixes:
            current = model.pooled_hidden(model.forward_hidden(sample.image_path, base_prompt + suffix), cfg.boundary.pooling)
            score = 0.0
            for layer, probe in probes.items():
                direction = cfg.attack.boundary_direction * probe["epsilon"] * probe["v"]
                target = original[layer].float().to(model.device) + direction
                score += model.torch.nn.functional.mse_loss(current[layer].float(), target).item()
            if score < best_score:
                best_score = score
                best_suffix = suffix
    return best_suffix


def optimize_sample(model: Qwen25VL, sample: SafetySample, probes: dict[int, dict[str, Any]], cfg: Config):
    torch = model.torch
    suffix = _select_suffix(model, sample, sample.prompt, cfg.attack.suffix_candidates, probes, cfg)
    prompt = sample.prompt + suffix
    inputs = model.build_inputs(sample.image_path, prompt)
    if "pixel_values" not in inputs:
        return None, suffix

    base_pixels = inputs["pixel_values"].detach()
    delta = torch.zeros_like(base_pixels, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=cfg.attack.visual_lr)

    with torch.no_grad():
        base_outputs = model.model(**inputs, output_hidden_states=True, return_dict=True)
        base_hidden = model.pooled_hidden(base_outputs, cfg.boundary.pooling)

    t_probes = {layer: _tensor_probe(torch, probe, model.device) for layer, probe in probes.items()}

    for step in range(cfg.attack.iterations):
        opt.zero_grad()
        adv_inputs = dict(inputs)
        adv_inputs["pixel_values"] = base_pixels + delta
        outputs = model.model(**adv_inputs, output_hidden_states=True, return_dict=True)
        current_hidden = model.pooled_hidden(outputs, cfg.boundary.pooling)

        align_loss = torch.zeros((), device=model.device)
        geo_loss = torch.zeros((), device=model.device)
        for layer, probe in t_probes.items():
            original_h = base_hidden[layer].float().detach()
            current_h = current_hidden[layer].float()
            target = original_h + cfg.attack.boundary_direction * probe["epsilon"] * probe["v"]
            delta_h = current_h - original_h
            align_loss = align_loss + torch.nn.functional.mse_loss(current_h, target)
            normed = delta_h / torch.linalg.vector_norm(delta_h).clamp_min(1e-6)
            geo_target = cfg.attack.boundary_direction * probe["v"]
            geo_loss = geo_loss + torch.nn.functional.mse_loss(normed, geo_target)

        sem_loss = torch.mean(delta * delta)
        total = align_loss + cfg.attack.lambda_geo * geo_loss + cfg.attack.lambda_sem * sem_loss
        total.backward()
        opt.step()
        with torch.no_grad():
            delta.clamp_(min=-cfg.attack.pixel_epsilon, max=cfg.attack.pixel_epsilon)

        if (step + 1) % 25 == 0:
            print(
                f"id={sample.sample_id} step={step + 1} "
                f"loss={total.item():.4f} align={align_loss.item():.4f} geo={geo_loss.item():.4f}"
            )

    return delta.detach(), suffix


def load_probes(boundary_path: str | Path) -> dict[int, dict[str, Any]]:
    torch = __import__("torch")
    try:
        obj = torch.load(boundary_path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(boundary_path, map_location="cpu")
    probes = {}
    for layer, probe in obj["probes"].items():
        probes[int(layer)] = {
            "v": np.asarray(probe["v"], dtype=np.float32),
            "epsilon": float(probe["epsilon"]),
        }
    return probes


def run_attack(cfg: Config, samples: list[SafetySample], boundary_path: str | Path | None = None) -> Path:
    cfg.validate_paths()
    boundary_path = Path(boundary_path or (cfg.output_path / "boundary_probes.pt"))
    probes = load_probes(boundary_path)
    model = Qwen25VL(cfg.target_model_path, cfg.device, cfg.torch_dtype, cfg.attn_implementation)
    out_path = cfg.output_path / "attack_results.jsonl"
    max_samples = cfg.attack.max_samples if cfg.attack.max_samples is not None else len(samples)
    selected = samples[:max_samples]
    with open(out_path, "w", encoding="utf-8") as f:
        for index, sample in enumerate(selected, start=1):
            print(f"[attack] {index}/{len(selected)} category={sample.category} id={sample.sample_id}")
            delta, suffix = optimize_sample(model, sample, probes, cfg)
            prompt = sample.prompt + suffix
            response = model.generate(sample.image_path, prompt, pixel_delta=delta, **cfg.attack.generate)
            row = {
                "sample_id": sample.sample_id,
                "category": sample.category,
                "image_path": str(sample.image_path),
                "ori_prompt": sample.prompt,
                "adv_suffix": suffix,
                "response": response,
                "attack": {
                    "iterations": cfg.attack.iterations,
                    "pixel_epsilon": cfg.attack.pixel_epsilon,
                    "layers": sorted(probes),
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
    return out_path
