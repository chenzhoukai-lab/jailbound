from __future__ import annotations

import argparse
from pathlib import Path

from jailbound.boundary import probe_boundaries
from jailbound.config import Config
from jailbound.dataset import load_mm_safetybench
from jailbound.guard import evaluate_results
from jailbound.loose_eval import evaluate_follow_asr, evaluate_loose_asr

from .analyze import read_jsonl, summarize_by_category, write_category_csv, write_markdown_report
from .attack import run_attack_v5
from .boundary import probe_boundaries_v5
from .config import V5Config
from .prompt_pairs import expanded_suffix_candidates
from .prompt_generator import generate_safe_prompt_cache
from .report import build_report


def _samples(cfg: Config, limit: int | None):
    root = Path(cfg.dataset_root)
    candidates = [root]
    if cfg.dataset_name:
        candidates.extend([root / cfg.dataset_name, root / "safebench", root / "mm-safetybench"])
    for candidate in candidates:
        try:
            return load_mm_safetybench(candidate, image_format=cfg.image_format, limit=limit)
        except FileNotFoundError:
            raise
        except Exception:
            continue
    raise FileNotFoundError(f"Could not find MM-SafetyBench data under {root}")


def _filter_categories(samples: list, categories: list[str] | None, per_category_limit: int | None) -> list:
    if not categories and per_category_limit is None:
        return samples
    wanted = set(categories or [])
    counts: dict[str, int] = {}
    selected = []
    for sample in samples:
        if wanted and sample.category not in wanted:
            continue
        if per_category_limit is not None:
            used = counts.get(sample.category, 0)
            if used >= per_category_limit:
                continue
            counts[sample.category] = used + 1
        selected.append(sample)
    print(f"Using category subset: {len(selected)} samples; counts={counts or 'unlimited'}")
    return selected


def _apply_job_options(base: Config, samples: list, args) -> list:
    num_splits = int(getattr(args, "num_splits", 1) or 1)
    split_index = int(getattr(args, "split_index", 0) or 0)
    if num_splits < 1:
        raise ValueError("--num-splits must be >= 1")
    if split_index < 0 or split_index >= num_splits:
        raise ValueError("--split-index must satisfy 0 <= split_index < num_splits")
    if num_splits == 1:
        return samples
    split = samples[split_index::num_splits]
    print(f"Using split {split_index}/{num_splits}: {len(split)} of {len(samples)} samples.")
    return split


def _apply_experiment_overrides(v5, base: Config, args) -> None:
    if getattr(args, "boundary_mode", None):
        v5.probe.boundary_mode = args.boundary_mode
    if getattr(args, "suffix_mode", None):
        v5.text_attack.mode = args.suffix_mode
    if getattr(args, "epsilon_scale", None) is not None:
        v5.crossing.epsilon_scale = args.epsilon_scale
    if getattr(args, "crossing_layers", None):
        v5.crossing.layers = _parse_layer_override(args.crossing_layers)
    if getattr(args, "safe_prompt_cache", None):
        v5.prompt_generation.cache_path = args.safe_prompt_cache
        v5.probe.generated_safe_prompt_path = args.safe_prompt_cache
    if v5.text_attack.mode == "suffix_bank":
        v5.text_attack.steps = 0
    elif v5.text_attack.steps <= 0:
        v5.text_attack.steps = 4
    tag = f"{v5.probe.boundary_mode}_{v5.text_attack.mode}"
    if getattr(args, "output_suffix", None):
        base.output_dir = f"{base.output_dir}_{args.output_suffix}"
    else:
        base.output_dir = f"{base.output_dir}_{tag}"


def _default_boundary_path(base: Config, v5) -> Path:
    if v5.probe.boundary_mode == "baseline_fixed_safe":
        return base.output_path / "boundary_probes.pt"
    return base.output_path / "boundary_probes_v5.pt"


def _generated_safe_prompt_path(v5: V5Config) -> str:
    return v5.probe.generated_safe_prompt_path or v5.prompt_generation.cache_path


def _parse_layer_override(values: list[str]):
    if len(values) == 1 and (values[0].startswith("last_") or values[0] in {"all", "probe"}):
        return values[0]
    return [int(x) for x in values]


def _probe_for_mode(base: Config, v5, samples: list) -> Path:
    if v5.probe.boundary_mode == "baseline_fixed_safe":
        return probe_boundaries(base, samples)
    if v5.probe.boundary_mode == "matched_safe_pair":
        generated_path = _generated_safe_prompt_path(v5)
        if "generated" in v5.probe.safe_pair_modes and not Path(generated_path).exists():
            raise FileNotFoundError(
                f"Missing generated safe prompt cache: {generated_path}. "
                "Run `python -m jailbound_v5 generate-safe-prompts ...` first."
            )
        return probe_boundaries_v5(
            base,
            samples,
            safe_modes=v5.probe.safe_pair_modes,
            unsafe_modes=v5.probe.unsafe_rephrase_modes,
            generated_safe_prompt_path=generated_path,
        )
    raise ValueError("--boundary-mode must be baseline_fixed_safe or matched_safe_pair")


def cmd_probe(args) -> None:
    v5, base = V5Config.from_json(args.config)
    _apply_experiment_overrides(v5, base, args)
    base.validate_paths()
    samples = _samples(base, args.limit)
    samples = _filter_categories(samples, args.categories, args.per_category_limit)
    samples = _apply_job_options(base, samples, args)
    out = _probe_for_mode(base, v5, samples)
    print(f"Saved v5 boundary probes: {out}")


def _prepare_v5_attack_config(v5: V5Config, base: Config, samples) -> None:
    base.attack.suffix = v5.text_attack.suffix_init
    expanded = []
    for sample in samples[: min(len(samples), 32)]:
        expanded.extend(expanded_suffix_candidates(sample.prompt))
    seen = set()
    merged = []
    for suffix in [v5.text_attack.suffix_init, *base.attack.suffix_candidates, *expanded]:
        if suffix and suffix not in seen:
            seen.add(suffix)
            merged.append(suffix)
    base.attack.suffix_candidates = merged


def cmd_attack(args) -> None:
    v5, base = V5Config.from_json(args.config)
    _apply_experiment_overrides(v5, base, args)
    base.validate_paths()
    samples = _samples(base, args.limit)
    samples = _filter_categories(samples, args.categories, args.per_category_limit)
    samples = _apply_job_options(base, samples, args)
    _prepare_v5_attack_config(v5, base, samples)
    boundary = Path(args.boundary) if args.boundary else _default_boundary_path(base, v5)
    if not boundary.exists():
        raise FileNotFoundError(f"Missing v5 boundary probes: {boundary}. Run `jailbound_v5 probe` first.")
    out = run_attack_v5(base, v5, samples, boundary, resume=args.resume)
    print(f"Saved v5 attack results: {out}")


def cmd_run(args) -> None:
    v5, base = V5Config.from_json(args.config)
    _apply_experiment_overrides(v5, base, args)
    base.validate_paths(require_guard=True)
    samples = _samples(base, args.limit)
    samples = _filter_categories(samples, args.categories, args.per_category_limit)
    samples = _apply_job_options(base, samples, args)
    _prepare_v5_attack_config(v5, base, samples)
    boundary = Path(args.boundary) if args.boundary else _default_boundary_path(base, v5)
    if not args.resume or not boundary.exists():
        boundary = _probe_for_mode(base, v5, samples)
    attack_results = run_attack_v5(base, v5, samples, boundary, resume=args.resume)
    evaluate_results(base, attack_results)


def cmd_generate_safe_prompts(args) -> None:
    v5, base = V5Config.from_json(args.config)
    if args.output:
        v5.prompt_generation.cache_path = args.output
        v5.probe.generated_safe_prompt_path = args.output
    if args.overwrite:
        v5.prompt_generation.overwrite = True
    base.validate_paths(require_follow_judge=not bool(v5.prompt_generation.model_path))
    samples = _samples(base, args.limit)
    samples = _filter_categories(samples, args.categories, args.per_category_limit)
    samples = _apply_job_options(base, samples, args)
    out = generate_safe_prompt_cache(base, v5, samples, output_path=args.output)
    print(f"Saved v5 generated safe prompt cache: {out}")


def cmd_analyze(args) -> None:
    rows = read_jsonl(args.input)
    summary = summarize_by_category(rows)
    out_dir = Path(args.output)
    csv_path = write_category_csv(summary, out_dir / "category_summary.csv")
    md_path = write_markdown_report(summary, out_dir / "category_report.md")
    print(f"Saved category CSV: {csv_path}")
    print(f"Saved markdown report: {md_path}")


def cmd_loose_eval(args) -> None:
    _v5, base = V5Config.from_json(args.config)
    if args.output_suffix:
        base.output_dir = f"{base.output_dir}_{args.output_suffix}"
    out, summary = evaluate_loose_asr(base, args.attack_results, args.guard_eval)
    print(f"Saved v5 non-refusal ASR evaluation: {out}")
    print(f"Saved v5 non-refusal ASR summary: {summary}")


def cmd_follow_eval(args) -> None:
    _v5, base = V5Config.from_json(args.config)
    if args.output_suffix:
        base.output_dir = f"{base.output_dir}_{args.output_suffix}"
    out, summary = evaluate_follow_asr(
        base,
        attack_results=args.attack_results,
        guard_eval=args.guard_eval,
        mode=args.mode,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    print(f"Saved v5 follow ASR evaluation: {out}")
    print(f"Saved v5 follow ASR summary: {summary}")


def _parse_run_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Run must be formatted as label=path, got: {raw}")
    label, path = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Run label cannot be empty: {raw}")
    return label, Path(path)


def cmd_report(args) -> None:
    runs = [_parse_run_arg(raw) for raw in args.run]
    paths = build_report(runs, args.output)
    for name, path in paths.items():
        print(f"Saved {name}: {path}")


def cmd_compare_boundaries(args) -> None:
    import json

    import numpy as np

    from jailbound.attack import load_probes

    left = load_probes(args.left)
    right = load_probes(args.right)
    rows = []
    for layer in sorted(set(left) & set(right)):
        lv = np.asarray(left[layer]["v"], dtype=np.float32)
        rv = np.asarray(right[layer]["v"], dtype=np.float32)
        cosine = float(np.dot(lv, rv) / (np.linalg.norm(lv) * np.linalg.norm(rv) + 1e-8))
        rows.append(
            {
                "layer": layer,
                "cosine": cosine,
                "left_epsilon": float(left[layer]["epsilon"]),
                "right_epsilon": float(right[layer]["epsilon"]),
                "epsilon_delta": float(right[layer]["epsilon"] - left[layer]["epsilon"]),
            }
        )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"left": args.left, "right": args.right, "layers": rows}, indent=2), encoding="utf-8")
    print(f"Saved boundary comparison: {out}")


def _add_subset_args(parser) -> None:
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--per-category-limit", type=int, default=None)
    parser.add_argument("--boundary-mode", choices=["baseline_fixed_safe", "matched_safe_pair"], default=None)
    parser.add_argument("--suffix-mode", choices=["suffix_bank", "readable_hotflip", "task_hotflip"], default=None)
    parser.add_argument("--safe-prompt-cache", default=None)
    parser.add_argument("--epsilon-scale", type=float, default=None)
    parser.add_argument("--crossing-layers", nargs="*", default=None)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="JailBound v5 experimental commands.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_generate = sub.add_parser("generate-safe-prompts", help="Generate v5 matched safe prompts from unsafe prompts.")
    p_generate.add_argument("--config", default="jailbound_v5/configs/qwen25vl_v5.json")
    p_generate.add_argument("--limit", type=int, default=None)
    p_generate.add_argument("--num-splits", type=int, default=1)
    p_generate.add_argument("--split-index", type=int, default=0)
    p_generate.add_argument("--categories", nargs="*", default=None)
    p_generate.add_argument("--per-category-limit", type=int, default=None)
    p_generate.add_argument("--output", default=None)
    p_generate.add_argument("--overwrite", action="store_true")
    p_generate.set_defaults(func=cmd_generate_safe_prompts)

    p_probe = sub.add_parser("probe", help="Run matched safe/unsafe probing.")
    p_probe.add_argument("--config", default="jailbound_v5/configs/qwen25vl_v5.json")
    p_probe.add_argument("--limit", type=int, default=None)
    p_probe.add_argument("--num-splits", type=int, default=1)
    p_probe.add_argument("--split-index", type=int, default=0)
    p_probe.add_argument("--output-suffix", default=None)
    _add_subset_args(p_probe)
    p_probe.set_defaults(func=cmd_probe)

    p_attack = sub.add_parser("attack", help="Run v5 attack using matched boundary probes.")
    p_attack.add_argument("--config", default="jailbound_v5/configs/qwen25vl_v5.json")
    p_attack.add_argument("--limit", type=int, default=None)
    p_attack.add_argument("--boundary", default=None)
    p_attack.add_argument("--resume", action="store_true")
    p_attack.add_argument("--num-splits", type=int, default=1)
    p_attack.add_argument("--split-index", type=int, default=0)
    p_attack.add_argument("--output-suffix", default=None)
    _add_subset_args(p_attack)
    p_attack.set_defaults(func=cmd_attack)

    p_run = sub.add_parser("run", help="Run v5 probe, attack, and Qwen3Guard evaluation.")
    p_run.add_argument("--config", default="jailbound_v5/configs/qwen25vl_v5.json")
    p_run.add_argument("--limit", type=int, default=None)
    p_run.add_argument("--resume", action="store_true")
    p_run.add_argument("--boundary", default=None)
    p_run.add_argument("--num-splits", type=int, default=1)
    p_run.add_argument("--split-index", type=int, default=0)
    p_run.add_argument("--output-suffix", default=None)
    _add_subset_args(p_run)
    p_run.set_defaults(func=cmd_run)

    p_analyze = sub.add_parser("analyze", help="Analyze guard_eval.jsonl by category.")
    p_analyze.add_argument("--input", required=True)
    p_analyze.add_argument("--output", default="outputs/qwen25vl_jailbound_v5/analysis")
    p_analyze.set_defaults(func=cmd_analyze)

    p_loose = sub.add_parser("loose-eval", help="Evaluate v5 outputs with non-refusal ASR.")
    p_loose.add_argument("--config", default="jailbound_v5/configs/qwen25vl_v5.json")
    p_loose.add_argument("--attack-results", default=None)
    p_loose.add_argument("--guard-eval", default=None)
    p_loose.add_argument("--output-suffix", default=None)
    p_loose.set_defaults(func=cmd_loose_eval)

    p_follow = sub.add_parser("follow-eval", help="Evaluate v5 outputs with heuristic and/or model-based follow ASR.")
    p_follow.add_argument("--config", default="jailbound_v5/configs/qwen25vl_v5.json")
    p_follow.add_argument("--attack-results", default=None)
    p_follow.add_argument("--guard-eval", default=None)
    p_follow.add_argument("--output-suffix", default=None)
    p_follow.add_argument("--mode", choices=["heuristic", "judge", "both"], default="both")
    p_follow.add_argument("--batch-size", type=int, default=None)
    p_follow.add_argument("--max-new-tokens", type=int, default=8)
    p_follow.set_defaults(func=cmd_follow_eval)

    p_report = sub.add_parser("report", help="Build unified ASR tables across baseline/v5 runs.")
    p_report.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run label and directory, formatted as label=path. Pass multiple times.",
    )
    p_report.add_argument("--output", default="outputs/qwen25vl_jailbound_report")
    p_report.set_defaults(func=cmd_report)

    p_compare = sub.add_parser("compare-boundaries", help="Compare boundary directions from two probe files.")
    p_compare.add_argument("--left", required=True)
    p_compare.add_argument("--right", required=True)
    p_compare.add_argument("--output", default="outputs/qwen25vl_jailbound_v5_boundary_compare.json")
    p_compare.set_defaults(func=cmd_compare_boundaries)

    args = parser.parse_args(argv)
    args.func(args)




