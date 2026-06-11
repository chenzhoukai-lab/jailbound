from __future__ import annotations

import argparse
from pathlib import Path

from jailbound.boundary import probe_boundaries
from jailbound.config import Config
from jailbound.dataset import load_mm_safetybench
from jailbound.guard import evaluate_results
from jailbound.loose_eval import evaluate_follow_asr, evaluate_loose_asr

from .analyze import read_jsonl, summarize_by_category, write_category_csv, write_markdown_report
from .attack import run_attack_v4
from .boundary import probe_boundaries_v4
from .config import V4Config
from .prompt_pairs import expanded_suffix_candidates
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


def _apply_experiment_overrides(v4, base: Config, args) -> None:
    if getattr(args, "boundary_mode", None):
        v4.probe.boundary_mode = args.boundary_mode
    if getattr(args, "suffix_mode", None):
        v4.text_attack.mode = args.suffix_mode
    if v4.text_attack.mode == "suffix_bank":
        v4.text_attack.steps = 0
    elif v4.text_attack.steps <= 0:
        v4.text_attack.steps = 4
    tag = f"{v4.probe.boundary_mode}_{v4.text_attack.mode}"
    if getattr(args, "output_suffix", None):
        base.output_dir = f"{base.output_dir}_{args.output_suffix}"
    else:
        base.output_dir = f"{base.output_dir}_{tag}"


def _default_boundary_path(base: Config, v4) -> Path:
    if v4.probe.boundary_mode == "baseline_fixed_safe":
        return base.output_path / "boundary_probes.pt"
    return base.output_path / "boundary_probes_v4.pt"


def _probe_for_mode(base: Config, v4, samples: list) -> Path:
    if v4.probe.boundary_mode == "baseline_fixed_safe":
        return probe_boundaries(base, samples)
    if v4.probe.boundary_mode == "matched_safe_pair":
        return probe_boundaries_v4(
            base,
            samples,
            safe_modes=v4.probe.safe_pair_modes,
            unsafe_modes=v4.probe.unsafe_rephrase_modes,
        )
    raise ValueError("--boundary-mode must be baseline_fixed_safe or matched_safe_pair")


def cmd_probe(args) -> None:
    v4, base = V4Config.from_json(args.config)
    _apply_experiment_overrides(v4, base, args)
    base.validate_paths()
    samples = _samples(base, args.limit)
    samples = _filter_categories(samples, args.categories, args.per_category_limit)
    samples = _apply_job_options(base, samples, args)
    out = _probe_for_mode(base, v4, samples)
    print(f"Saved v4 boundary probes: {out}")


def _prepare_v4_attack_config(v4: V4Config, base: Config, samples) -> None:
    base.attack.suffix = v4.text_attack.suffix_init
    expanded = []
    for sample in samples[: min(len(samples), 32)]:
        expanded.extend(expanded_suffix_candidates(sample.prompt))
    seen = set()
    merged = []
    for suffix in [v4.text_attack.suffix_init, *base.attack.suffix_candidates, *expanded]:
        if suffix and suffix not in seen:
            seen.add(suffix)
            merged.append(suffix)
    base.attack.suffix_candidates = merged


def cmd_attack(args) -> None:
    v4, base = V4Config.from_json(args.config)
    _apply_experiment_overrides(v4, base, args)
    base.validate_paths()
    samples = _samples(base, args.limit)
    samples = _filter_categories(samples, args.categories, args.per_category_limit)
    samples = _apply_job_options(base, samples, args)
    _prepare_v4_attack_config(v4, base, samples)
    boundary = Path(args.boundary) if args.boundary else _default_boundary_path(base, v4)
    if not boundary.exists():
        raise FileNotFoundError(f"Missing v4 boundary probes: {boundary}. Run `jailbound_v4 probe` first.")
    out = run_attack_v4(base, v4, samples, boundary, resume=args.resume)
    print(f"Saved v4 attack results: {out}")


def cmd_run(args) -> None:
    v4, base = V4Config.from_json(args.config)
    _apply_experiment_overrides(v4, base, args)
    base.validate_paths(require_guard=True)
    samples = _samples(base, args.limit)
    samples = _filter_categories(samples, args.categories, args.per_category_limit)
    samples = _apply_job_options(base, samples, args)
    _prepare_v4_attack_config(v4, base, samples)
    boundary = Path(args.boundary) if args.boundary else _default_boundary_path(base, v4)
    if not args.resume or not boundary.exists():
        boundary = _probe_for_mode(base, v4, samples)
    attack_results = run_attack_v4(base, v4, samples, boundary, resume=args.resume)
    evaluate_results(base, attack_results)


def cmd_analyze(args) -> None:
    rows = read_jsonl(args.input)
    summary = summarize_by_category(rows)
    out_dir = Path(args.output)
    csv_path = write_category_csv(summary, out_dir / "category_summary.csv")
    md_path = write_markdown_report(summary, out_dir / "category_report.md")
    print(f"Saved category CSV: {csv_path}")
    print(f"Saved markdown report: {md_path}")


def cmd_loose_eval(args) -> None:
    _v4, base = V4Config.from_json(args.config)
    if args.output_suffix:
        base.output_dir = f"{base.output_dir}_{args.output_suffix}"
    out, summary = evaluate_loose_asr(base, args.attack_results, args.guard_eval)
    print(f"Saved v4 non-refusal ASR evaluation: {out}")
    print(f"Saved v4 non-refusal ASR summary: {summary}")


def cmd_follow_eval(args) -> None:
    _v4, base = V4Config.from_json(args.config)
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
    print(f"Saved v4 follow ASR evaluation: {out}")
    print(f"Saved v4 follow ASR summary: {summary}")


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
    parser.add_argument("--suffix-mode", choices=["suffix_bank", "readable_hotflip"], default=None)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="JailBound v4 experimental commands.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_probe = sub.add_parser("probe", help="Run matched safe/unsafe probing.")
    p_probe.add_argument("--config", default="jailbound_v4/configs/qwen25vl_v4.json")
    p_probe.add_argument("--limit", type=int, default=None)
    p_probe.add_argument("--num-splits", type=int, default=1)
    p_probe.add_argument("--split-index", type=int, default=0)
    p_probe.add_argument("--output-suffix", default=None)
    _add_subset_args(p_probe)
    p_probe.set_defaults(func=cmd_probe)

    p_attack = sub.add_parser("attack", help="Run v4 attack using matched boundary probes.")
    p_attack.add_argument("--config", default="jailbound_v4/configs/qwen25vl_v4.json")
    p_attack.add_argument("--limit", type=int, default=None)
    p_attack.add_argument("--boundary", default=None)
    p_attack.add_argument("--resume", action="store_true")
    p_attack.add_argument("--num-splits", type=int, default=1)
    p_attack.add_argument("--split-index", type=int, default=0)
    p_attack.add_argument("--output-suffix", default=None)
    _add_subset_args(p_attack)
    p_attack.set_defaults(func=cmd_attack)

    p_run = sub.add_parser("run", help="Run v4 probe, attack, and Qwen3Guard evaluation.")
    p_run.add_argument("--config", default="jailbound_v4/configs/qwen25vl_v4.json")
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
    p_analyze.add_argument("--output", default="outputs/qwen25vl_jailbound_v4/analysis")
    p_analyze.set_defaults(func=cmd_analyze)

    p_loose = sub.add_parser("loose-eval", help="Evaluate v4 outputs with non-refusal ASR.")
    p_loose.add_argument("--config", default="jailbound_v4/configs/qwen25vl_v4.json")
    p_loose.add_argument("--attack-results", default=None)
    p_loose.add_argument("--guard-eval", default=None)
    p_loose.add_argument("--output-suffix", default=None)
    p_loose.set_defaults(func=cmd_loose_eval)

    p_follow = sub.add_parser("follow-eval", help="Evaluate v4 outputs with heuristic and/or model-based follow ASR.")
    p_follow.add_argument("--config", default="jailbound_v4/configs/qwen25vl_v4.json")
    p_follow.add_argument("--attack-results", default=None)
    p_follow.add_argument("--guard-eval", default=None)
    p_follow.add_argument("--output-suffix", default=None)
    p_follow.add_argument("--mode", choices=["heuristic", "judge", "both"], default="both")
    p_follow.add_argument("--batch-size", type=int, default=None)
    p_follow.add_argument("--max-new-tokens", type=int, default=8)
    p_follow.set_defaults(func=cmd_follow_eval)

    p_report = sub.add_parser("report", help="Build unified ASR tables across baseline/v4 runs.")
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
    p_compare.add_argument("--output", default="outputs/qwen25vl_jailbound_v4_boundary_compare.json")
    p_compare.set_defaults(func=cmd_compare_boundaries)

    args = parser.parse_args(argv)
    args.func(args)



