from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .dataset import SafetySample
from .distributed import get_accelerator, reset_shard_dir, runtime_device, shard_items
from .modeling_qwen import Qwen25VL


def _tensor_probe(torch, probe: dict[str, Any], device):
    return {
        "v": torch.tensor(probe["v"], dtype=torch.float32, device=device),
        "epsilon": float(probe["epsilon"]),
    }


def _select_suffix(model: Qwen25VL, sample: SafetySample, base_prompt: str, suffixes: list[str], probes, cfg: Config) -> str:
    # 论文里的文本扰动是 token-level 梯度替换，完整实现比较绕。
    # 这里先做一个可读、可调试的版本：从候选 suffix 中选边界损失最低的。
    # 后续如果要升级成 HotFlip/梯度替换，可以从这个函数扩展。
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
    # Safety Boundary Crossing 的单样本优化。
    # 输入：原图 + 原始 unsafe prompt
    # 输出：pixel_delta + 被选中的文本 suffix
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
        # base_hidden 是攻击前的内部状态 h(l)，用来构造目标 h_target。
        base_outputs = model.model(**inputs, output_hidden_states=True, return_dict=True)
        base_hidden = model.pooled_hidden(base_outputs, cfg.boundary.pooling)

    t_probes = {layer: _tensor_probe(torch, probe, model.device) for layer, probe in probes.items()}

    for step in range(cfg.attack.iterations):
        # 每一步都重新前向，得到当前扰动后的 hidden states 。
        # 损失由三部分组成：L_align、L_geo、L_sem，对应论文 Eq. 6/7/8。
        opt.zero_grad()
        adv_inputs = dict(inputs)
        adv_inputs["pixel_values"] = base_pixels + delta
        outputs = model.model(**adv_inputs, output_hidden_states=True, return_dict=True)
        current_hidden = model.pooled_hidden(outputs, cfg.boundary.pooling)

        align_loss = torch.zeros((), device=model.device)
        geo_loss = torch.zeros((), device=model.device)
        for layer, probe in t_probes.items():
            # L_align：把当前 h_adv 拉向 h + direction * epsilon * v。
            # config 里 boundary_direction 默认 -1，是为了贴近论文 h_target = h - epsilon * v。
            original_h = base_hidden[layer].float().detach()
            current_h = current_hidden[layer].float()
            target = original_h + cfg.attack.boundary_direction * probe["epsilon"] * probe["v"]
            delta_h = current_h - original_h
            align_loss = align_loss + torch.nn.functional.mse_loss(current_h, target)
            # L_geo：要求移动方向和边界法向量一致，避免只靠 L2 乱飘。
            normed = delta_h / torch.linalg.vector_norm(delta_h).clamp_min(1e-6)
            geo_target = cfg.attack.boundary_direction * probe["v"]
            geo_loss = geo_loss + torch.nn.functional.mse_loss(normed, geo_target)

        sem_loss = torch.mean(delta * delta)
        total = align_loss + cfg.attack.lambda_geo * geo_loss + cfg.attack.lambda_sem * sem_loss
        total.backward()
        opt.step()
        with torch.no_grad():
            # 投影步骤，对应论文里的 Pi_Gamma，保证扰动不超过预算。
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
    # 批量攻击入口：逐条样本优化扰动、生成回答，并写成 jsonl。
    # jsonl 的好处是中途失败时前面已经完成的样本还能保留。
    accelerator = get_accelerator()
    cfg.validate_paths()
    boundary_path = Path(boundary_path or (cfg.output_path / "boundary_probes.pt"))
    probes = load_probes(boundary_path)
    model = Qwen25VL(cfg.target_model_path, runtime_device(cfg, accelerator), cfg.torch_dtype, cfg.attn_implementation)
    out_path = cfg.output_path / "attack_results.jsonl"
    shard_dir = cfg.output_path / "_attack_shards"
    reset_shard_dir(shard_dir, accelerator)

    max_samples = cfg.attack.max_samples if cfg.attack.max_samples is not None else len(samples)
    selected = samples[:max_samples]
    indexed = list(enumerate(selected))
    local_indexed = shard_items(indexed, accelerator)
    local_path = shard_dir / f"rank_{accelerator.process_index}.jsonl"
    with open(local_path, "w", encoding="utf-8") as f:
        for local_pos, (global_index, sample) in enumerate(local_indexed, start=1):
            print(
                f"[attack][rank {accelerator.process_index}] "
                f"{local_pos}/{len(local_indexed)} global={global_index + 1}/{len(selected)} "
                f"category={sample.category} id={sample.sample_id}",
                flush=True,
            )
            delta, suffix = optimize_sample(model, sample, probes, cfg)
            prompt = sample.prompt + suffix
            response = model.generate(sample.image_path, prompt, pixel_delta=delta, **cfg.attack.generate)
            row = {
                "_order": global_index,
                "rank": accelerator.process_index,
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
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        rows = []
        for shard_path in sorted(shard_dir.glob("rank_*.jsonl")):
            rows.extend(json.loads(line) for line in shard_path.read_text(encoding="utf-8").splitlines() if line.strip())
        rows.sort(key=lambda x: x["_order"])
        with open(out_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"merged {len(rows)} attack rows from {accelerator.num_processes} process(es): {out_path}")
    accelerator.wait_for_everyone()
    return out_path
