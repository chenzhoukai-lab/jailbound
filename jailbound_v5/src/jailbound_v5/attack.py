from __future__ import annotations

import json
import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from jailbound.attack import _load_completed_attack_rows, _merge_attack_shards, _select_suffix, _tensor_probe, load_probes
from jailbound.config import Config
from jailbound.dataset import SafetySample
from jailbound.distributed import get_accelerator, reset_shard_dir, runtime_device, shard_items
from jailbound.modeling_qwen import Qwen25VL

from .config import V5Config


def _ids(inputs: dict[str, Any]) -> list[int]:
    return inputs["input_ids"][0].detach().cpu().tolist()


def _chat_text(model: Qwen25VL, prompt: str) -> str:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    return model.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _offset_span(model: Qwen25VL, full_inputs: dict[str, Any], base_prompt: str, suffix: str):
    tokenizer = model.tokenizer
    if not suffix:
        return None
    try:
        full_text = _chat_text(model, base_prompt + suffix)
        suffix_start = full_text.rfind(suffix)
        if suffix_start < 0:
            return None
        suffix_end = suffix_start + len(suffix)
        encoded = tokenizer(full_text, return_offsets_mapping=True, add_special_tokens=False)
    except Exception:
        return None

    token_ids = list(encoded["input_ids"])
    full_ids = _ids(full_inputs)
    offset_shift = -1
    for start in range(0, len(full_ids) - len(token_ids) + 1):
        if full_ids[start : start + len(token_ids)] == token_ids:
            offset_shift = start
            break
    if offset_shift < 0:
        return None

    positions = [
        offset_shift + idx
        for idx, (left, right) in enumerate(encoded["offset_mapping"])
        if right > suffix_start and left < suffix_end
    ]
    if not positions:
        return None
    return min(positions), max(positions) + 1, {
        "method": "offset_mapping",
        "suffix_char_span": [suffix_start, suffix_end],
        "template_token_offset": offset_shift,
    }


def locate_suffix_span(
    model: Qwen25VL, image_path: str | Path, base_prompt: str, suffix: str
) -> tuple[int, int, dict[str, Any], dict[str, Any]]:
    """Locate suffix token span after Qwen chat-template expansion.

    Prefer tokenizer offset mappings on the expanded chat template, then fall
    back to input-id diffing. This avoids guessing where Qwen's chat-template
    and image special tokens begin and end.
    """

    full_inputs = model.build_inputs(image_path, base_prompt + suffix)
    offset = _offset_span(model, full_inputs, base_prompt, suffix)
    if offset is not None:
        start, end, meta = offset
        meta.update({"base_prompt_tokens": None, "full_prompt_tokens": len(_ids(full_inputs))})
        return start, end, full_inputs, meta

    base_inputs = model.build_inputs(image_path, base_prompt)
    base_ids = _ids(base_inputs)
    full_ids = _ids(full_inputs)

    prefix = 0
    max_prefix = min(len(base_ids), len(full_ids))
    while prefix < max_prefix and base_ids[prefix] == full_ids[prefix]:
        prefix += 1

    suffix_len = 0
    while (
        suffix_len < len(base_ids) - prefix
        and suffix_len < len(full_ids) - prefix
        and base_ids[len(base_ids) - 1 - suffix_len] == full_ids[len(full_ids) - 1 - suffix_len]
    ):
        suffix_len += 1

    end = len(full_ids) - suffix_len
    if prefix >= end:
        raise ValueError(
            f"Could not locate suffix span. base_len={len(base_ids)} full_len={len(full_ids)} suffix={suffix!r}"
        )
    return prefix, end, full_inputs, {
        "method": "id_diff",
        "base_prompt_tokens": len(base_ids),
        "full_prompt_tokens": len(full_ids),
    }


def _boundary_loss(model: Qwen25VL, outputs, original_hidden, probes: dict[int, dict[str, Any]], cfg: Config):
    torch = model.torch
    current_hidden = model.pooled_hidden(outputs, cfg.boundary.pooling)
    align_loss = torch.zeros((), device=model.device)
    geo_loss = torch.zeros((), device=model.device)
    for layer, probe in probes.items():
        v = torch.as_tensor(probe["v"], dtype=torch.float32, device=model.device)
        original_h = original_hidden[layer].float().detach()
        current_h = current_hidden[layer].float()
        target = original_h + cfg.attack.boundary_direction * float(probe["epsilon"]) * v
        delta_h = current_h - original_h
        align_loss = align_loss + torch.nn.functional.mse_loss(current_h, target)
        normed = delta_h / torch.linalg.vector_norm(delta_h).clamp_min(1e-6)
        geo_loss = geo_loss + torch.nn.functional.mse_loss(normed, cfg.attack.boundary_direction * v)
    return align_loss + cfg.attack.lambda_geo * geo_loss, align_loss.detach(), geo_loss.detach()


def _freeze_model_parameters(model: Qwen25VL) -> None:
    for param in model.model.parameters():
        param.requires_grad_(False)


def _resolve_probe_layers(layer_spec, available_layers: list[int]) -> list[int]:
    available = sorted(int(x) for x in available_layers)
    if layer_spec is None:
        return available
    if isinstance(layer_spec, list):
        wanted = {int(x) for x in layer_spec}
        selected = [layer for layer in available if layer in wanted]
    elif isinstance(layer_spec, str):
        if layer_spec in {"all", "probe"}:
            return available
        if layer_spec.startswith("last_"):
            count = int(layer_spec.split("_", 1)[1])
            selected = available[-count:]
        else:
            selected = [layer for layer in available if layer == int(layer_spec)]
    else:
        raise ValueError(f"Unsupported v5 crossing.layers: {layer_spec!r}")
    if not selected:
        raise ValueError(f"v5 crossing.layers selected no probe layers from available={available}")
    return selected


def prepare_v5_probes(probes: dict[int, dict[str, Any]], v5: V5Config) -> dict[int, dict[str, Any]]:
    selected_layers = _resolve_probe_layers(v5.crossing.layers, sorted(probes))
    prepared: dict[int, dict[str, Any]] = {}
    for layer in selected_layers:
        probe = copy.deepcopy(probes[int(layer)])
        probe["epsilon"] = float(probe["epsilon"]) * float(v5.crossing.epsilon_scale)
        prepared[int(layer)] = probe
    return prepared


def _suffix_grad_loss(
    model: Qwen25VL,
    sample: SafetySample,
    base_prompt: str,
    suffix: str,
    original_hidden,
    probes: dict[int, dict[str, Any]],
    cfg: Config,
):
    torch = model.torch
    start, end, inputs, span_meta = locate_suffix_span(model, sample.image_path, base_prompt, suffix)
    emb_layer = model.model.get_input_embeddings()
    captured = []

    def hook(_module, _args, output):
        grad_output = output.detach().requires_grad_(True)
        grad_output.retain_grad()
        captured.append(grad_output)
        return grad_output

    handle = emb_layer.register_forward_hook(hook)
    try:
        model.model.zero_grad(set_to_none=True)
        outputs = model.model(**inputs, output_hidden_states=True, return_dict=True)
        loss, align, geo = _boundary_loss(model, outputs, original_hidden, probes, cfg)
        loss.backward()
        if not captured or captured[0].grad is None:
            raise RuntimeError("Failed to capture embedding gradients for suffix HotFlip.")
        grad = captured[0].grad[0, start:end].detach()
        token_ids = inputs["input_ids"][0, start:end].detach().cpu().tolist()
        return loss.detach(), align, geo, grad, token_ids, (start, end), span_meta
    finally:
        handle.remove()
        model.model.zero_grad(set_to_none=True)


def _allowed_token_mask(tokenizer, embedding_weight):
    import torch

    mask = torch.ones(embedding_weight.shape[0], dtype=torch.bool, device=embedding_weight.device)
    for token_id in getattr(tokenizer, "all_special_ids", []) or []:
        if token_id is not None and 0 <= int(token_id) < mask.numel():
            mask[int(token_id)] = False
    return mask


def _is_readable_token(tokenizer, token_id: int) -> bool:
    if token_id in set(getattr(tokenizer, "all_special_ids", []) or []):
        return False
    text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    if not text:
        return False
    if any(ch in text for ch in ("\n", "\r", "\t", "\x00", "\ufffd", "锟")):
        return False
    if not all(ch.isascii() and (ch.isprintable() or ch == " ") for ch in text):
        return False
    stripped = text.strip()
    if not stripped:
        return False
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?'-_/()[]")
    return all(ch in allowed for ch in text)


def _readability_penalty(text: str, initial_suffix: str) -> float:
    penalty = 0.0
    if not text.strip():
        return 100.0
    penalty += sum(10.0 for ch in text if not (ch.isascii() and (ch.isprintable() or ch == " ")))
    penalty += 10.0 * text.count("\ufffd")
    penalty += 10.0 * text.count("锟")
    if "\n" in text or "\r" in text or "\t" in text:
        penalty += 10.0
    words = text.split()
    if len(words) < 4:
        penalty += 2.0
    if len(text) > max(120, len(initial_suffix) * 2):
        penalty += 2.0
    return penalty


def _decode_clean(tokenizer, token_ids: list[int]) -> str:
    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return " " + " ".join(text.split()) if text and not text.startswith(" ") else text


def _task_token_ids(tokenizer, task_tokens: list[str]) -> set[int]:
    ids: set[int] = set()
    for token in task_tokens:
        for text in (token, " " + token, token.capitalize(), " " + token.capitalize()):
            try:
                encoded = tokenizer.encode(text, add_special_tokens=False)
            except Exception:
                encoded = []
            if len(encoded) == 1:
                ids.add(int(encoded[0]))
    return ids


def _suffix_task_bonus(text: str, task_tokens: list[str]) -> float:
    lower = text.lower()
    return float(sum(1 for token in task_tokens if token.lower() in lower))


def _candidate_task_bonus(tokenizer, token_id: int, task_ids: set[int], task_tokens: list[str]) -> float:
    if int(token_id) in task_ids:
        return 1.0
    decoded = tokenizer.decode([int(token_id)], skip_special_tokens=False).strip().lower()
    return 1.0 if decoded in {token.lower() for token in task_tokens} else 0.0


def _evaluate_suffix_loss(
    model: Qwen25VL,
    sample: SafetySample,
    suffix: str,
    original_hidden,
    probes: dict[int, dict[str, Any]],
    cfg: Config,
) -> float:
    with model.torch.no_grad():
        inputs = model.build_inputs(sample.image_path, sample.prompt + suffix)
        outputs = model.model(**inputs, output_hidden_states=True, return_dict=True)
        loss, _align, _geo = _boundary_loss(model, outputs, original_hidden, probes, cfg)
        return float(loss.detach().cpu())


def hotflip_suffix(
    model: Qwen25VL,
    sample: SafetySample,
    probes: dict[int, dict[str, Any]],
    cfg: Config,
    v5: V5Config,
    initial_suffix: str,
):
    torch = model.torch
    tokenizer = model.tokenizer
    suffix = initial_suffix
    trace = []

    with torch.no_grad():
        base_outputs = model.forward_hidden(sample.image_path, sample.prompt)
        original_hidden = [x.detach().float() for x in model.pooled_hidden(base_outputs, cfg.boundary.pooling)]

    embedding_weight = model.model.get_input_embeddings().weight.detach()
    allowed = _allowed_token_mask(tokenizer, embedding_weight)
    readable_ids = None
    task_ids: set[int] = set()
    if v5.text_attack.mode in {"readable_hotflip", "task_hotflip"}:
        readable_ids = {
            token_id
            for token_id in range(int(embedding_weight.shape[0]))
            if bool(allowed[token_id].detach().cpu()) and _is_readable_token(tokenizer, token_id)
        }
    if v5.text_attack.mode == "task_hotflip":
        task_ids = _task_token_ids(tokenizer, v5.text_attack.task_tokens)
    best_suffix = suffix
    best_loss = _evaluate_suffix_loss(model, sample, suffix, original_hidden, probes, cfg)
    best_score = (
        best_loss
        + v5.text_attack.readability_weight * _readability_penalty(suffix, initial_suffix)
        - v5.text_attack.task_token_weight * _suffix_task_bonus(suffix, v5.text_attack.task_tokens)
    )
    initial_start, initial_end, _, initial_span_meta = locate_suffix_span(
        model, sample.image_path, sample.prompt, suffix
    )
    trace.append(
        {
            "step": -1,
            "span": [initial_start, initial_end],
            "span_meta": initial_span_meta,
            "suffix": suffix,
            "loss": best_loss,
            "score": best_score,
            "readability_penalty": _readability_penalty(suffix, initial_suffix),
            "task_bonus": _suffix_task_bonus(suffix, v5.text_attack.task_tokens),
            "tokens": tokenizer.convert_ids_to_tokens(
                model.build_inputs(sample.image_path, sample.prompt + suffix)["input_ids"][0, initial_start:initial_end]
                .detach()
                .cpu()
                .tolist()
            ),
        }
    )

    for step in range(max(0, v5.text_attack.steps)):
        loss, align, geo, grad, token_ids, span, span_meta = _suffix_grad_loss(
            model, sample, sample.prompt, suffix, original_hidden, probes, cfg
        )
        if not token_ids:
            break

        current_penalty = _readability_penalty(suffix, initial_suffix)
        current_task_bonus = _suffix_task_bonus(suffix, v5.text_attack.task_tokens)
        current_score = (
            float(loss.cpu())
            + v5.text_attack.readability_weight * current_penalty
            - v5.text_attack.task_token_weight * current_task_bonus
        )
        candidate_pool = []
        for pos, token_id in enumerate(token_ids):
            current = embedding_weight[int(token_id)]
            scores = torch.matmul(embedding_weight - current, -grad[pos].to(embedding_weight.device))
            scores = scores.masked_fill(~allowed, float("-inf"))
            top_ids = torch.topk(scores, k=min(v5.text_attack.hotflip_top_k, scores.numel())).indices.tolist()
            candidate_ids = list(dict.fromkeys([int(x) for x in top_ids] + (list(task_ids) if task_ids else [])))
            for candidate_id in candidate_ids:
                if int(candidate_id) == int(token_id):
                    continue
                candidate_score = float(scores[int(candidate_id)].detach().cpu())
                if readable_ids is not None and int(candidate_id) not in readable_ids:
                    continue
                task_bonus = _candidate_task_bonus(tokenizer, int(candidate_id), task_ids, v5.text_attack.task_tokens)
                candidate_pool.append(
                    {
                        "pos": pos,
                        "old": int(token_id),
                        "new": int(candidate_id),
                        "grad_score": candidate_score,
                        "task_token_bonus": task_bonus,
                        "rank_score": candidate_score + v5.text_attack.task_token_weight * task_bonus,
                    }
                )
        candidate_pool.sort(key=lambda x: x["rank_score"], reverse=True)
        if not candidate_pool:
            break

        accepted = None
        evaluated = []
        for candidate in candidate_pool[: max(1, v5.text_attack.max_eval_candidates)]:
            new_ids = list(token_ids)
            new_ids[candidate["pos"]] = candidate["new"]
            candidate_suffix = _decode_clean(tokenizer, new_ids)
            penalty = _readability_penalty(candidate_suffix, initial_suffix)
            if v5.text_attack.mode in {"readable_hotflip", "task_hotflip"} and penalty >= 10.0:
                continue
            real_loss = _evaluate_suffix_loss(model, sample, candidate_suffix, original_hidden, probes, cfg)
            task_bonus = _suffix_task_bonus(candidate_suffix, v5.text_attack.task_tokens)
            combined = real_loss + v5.text_attack.readability_weight * penalty - v5.text_attack.task_token_weight * task_bonus
            item = {
                **candidate,
                "loss": real_loss,
                "penalty": penalty,
                "task_bonus": task_bonus,
                "combined": combined,
                "suffix": candidate_suffix,
            }
            evaluated.append(item)
            if accepted is None or combined < accepted["combined"]:
                accepted = item

        if accepted is None or accepted["combined"] > current_score - v5.text_attack.min_improvement:
            trace.append(
                {
                    "step": step,
                    "loss": float(loss.cpu()),
                    "align": float(align.cpu()),
                    "geo": float(geo.cpu()),
                    "span": list(span),
                    "span_meta": span_meta,
                    "current_score": current_score,
                    "readability_penalty": current_penalty,
                    "task_bonus": current_task_bonus,
                    "evaluated_candidates": len(evaluated),
                    "accepted": False,
                    "suffix": suffix,
                }
            )
            break

        suffix = accepted["suffix"]
        if accepted["combined"] < best_score:
            best_suffix = suffix
            best_loss = accepted["loss"]
            best_score = accepted["combined"]
        trace.append(
            {
                "step": step,
                "loss": float(loss.cpu()),
                "align": float(align.cpu()),
                "geo": float(geo.cpu()),
                "span": list(span),
                "span_meta": span_meta,
                "position": accepted["pos"],
                "old_token_id": accepted["old"],
                "new_token_id": accepted["new"],
                "old_token": tokenizer.decode([accepted["old"]], skip_special_tokens=False),
                "new_token": tokenizer.decode([accepted["new"]], skip_special_tokens=False),
                "grad_score": accepted["grad_score"],
                "candidate_loss": accepted["loss"],
                "candidate_score": accepted["combined"],
                "readability_penalty": accepted["penalty"],
                "task_bonus": accepted["task_bonus"],
                "evaluated_candidates": len(evaluated),
                "accepted": True,
                "best_suffix": best_suffix,
                "best_loss": best_loss,
                "best_score": best_score,
                "suffix": suffix,
            }
        )

    return best_suffix, trace


def optimize_sample_v5(
    model: Qwen25VL,
    sample: SafetySample,
    probes: dict[int, dict[str, Any]],
    cfg: Config,
    v5: V5Config,
):
    torch = model.torch
    initial_suffix = _select_suffix(model, sample, sample.prompt, cfg.attack.suffix_candidates, probes, cfg)
    if v5.text_attack.mode in {"readable_hotflip", "task_hotflip"} and v5.text_attack.steps > 0:
        suffix, hotflip_trace = hotflip_suffix(model, sample, probes, cfg, v5, initial_suffix)
    else:
        suffix = initial_suffix
        hotflip_trace = [{"step": -1, "suffix": suffix, "note": "hotflip_disabled"}]
    prompt = sample.prompt + suffix
    inputs = model.build_inputs(sample.image_path, prompt)
    if "pixel_values" not in inputs:
        return None, suffix, {"initial_suffix": initial_suffix, "hotflip_trace": hotflip_trace}

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
            geo_loss = geo_loss + torch.nn.functional.mse_loss(normed, cfg.attack.boundary_direction * probe["v"])

        sem_loss = torch.mean(delta * delta)
        total = align_loss + cfg.attack.lambda_geo * geo_loss + cfg.attack.lambda_sem * sem_loss
        total.backward()
        opt.step()
        with torch.no_grad():
            delta.clamp_(min=-cfg.attack.pixel_epsilon, max=cfg.attack.pixel_epsilon)

        if (step + 1) % 25 == 0:
            print(
                f"id={sample.sample_id} step={step + 1} "
                f"loss={total.item():.4f} align={align_loss.item():.4f} geo={geo_loss.item():.4f}",
                flush=True,
            )

    return delta.detach(), suffix, {"initial_suffix": initial_suffix, "hotflip_trace": hotflip_trace}


def run_attack_v5(
    cfg: Config,
    v5: V5Config,
    samples: list[SafetySample],
    boundary_path: str | Path | None = None,
    resume: bool = False,
) -> Path:
    accelerator = get_accelerator()
    cfg.validate_paths()
    boundary_path = Path(boundary_path or (cfg.output_path / "boundary_probes_v5.pt"))
    probes = prepare_v5_probes(load_probes(boundary_path), v5)
    attack_variant = f"jailbound_v5_{v5.probe.boundary_mode}_{v5.text_attack.mode}"
    out_path = cfg.output_path / "attack_results.jsonl"
    shard_dir = cfg.output_path / "_attack_shards"
    if resume:
        shard_dir.mkdir(parents=True, exist_ok=True)
        accelerator.wait_for_everyone()
    else:
        reset_shard_dir(shard_dir, accelerator)

    max_samples = cfg.attack.max_samples if cfg.attack.max_samples is not None else len(samples)
    selected = samples[:max_samples]
    indexed = list(enumerate(selected))
    completed_rows = _load_completed_attack_rows(out_path, shard_dir) if resume else {}
    pending_indexed = [(idx, sample) for idx, sample in indexed if idx not in set(completed_rows)]
    local_indexed = shard_items(pending_indexed, accelerator)
    local_path = shard_dir / f"rank_{accelerator.process_index}.jsonl"

    if accelerator.is_main_process and resume:
        print(
            f"[v5 attack][resume] completed={len(completed_rows)} pending={len(pending_indexed)} total={len(indexed)}",
            flush=True,
        )

    model = None
    if local_indexed:
        model = Qwen25VL(cfg.target_model_path, runtime_device(cfg, accelerator), cfg.torch_dtype, cfg.attn_implementation)
        _freeze_model_parameters(model)

    with open(local_path, "a" if resume else "w", encoding="utf-8") as f:
        for local_pos, (global_index, sample) in enumerate(local_indexed, start=1):
            print(
                f"[v5 attack][rank {accelerator.process_index}] "
                f"{local_pos}/{len(local_indexed)} global={global_index + 1}/{len(selected)} "
                f"category={sample.category} id={sample.sample_id}",
                flush=True,
            )
            assert model is not None
            delta, suffix, text_meta = optimize_sample_v5(model, sample, probes, cfg, v5)
            response = model.generate(sample.image_path, sample.prompt + suffix, pixel_delta=delta, **cfg.attack.generate)
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
                    "variant": attack_variant,
                    "iterations": cfg.attack.iterations,
                    "pixel_epsilon": cfg.attack.pixel_epsilon,
                    "layers": sorted(probes),
                    "crossing": asdict(v5.crossing),
                    "text_attack": asdict(v5.text_attack),
                    **text_meta,
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        rows = _merge_attack_shards(out_path, shard_dir)
        print(f"merged {len(rows)} v5 attack rows from {accelerator.num_processes} process(es): {out_path}")
    accelerator.wait_for_everyone()
    return out_path




