from __future__ import annotations

import argparse
from pathlib import Path

from jailbound.attack import run_attack
from jailbound.config import Config
from jailbound.dataset import load_mm_safetybench
from jailbound.guard import evaluate_results

from .analyze import read_jsonl, summarize_by_category, write_category_csv, write_markdown_report
from .boundary import probe_boundaries_v2
from .config import V2Config
from .prompt_pairs import expanded_suffix_candidates


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


def cmd_probe(args) -> None:
    v2, base = V2Config.from_json(args.config)
    base.validate_paths()
    samples = _samples(base, args.limit)
    out = probe_boundaries_v2(
        base,
        samples,
        safe_modes=v2.probe.safe_pair_modes,
        unsafe_modes=v2.probe.unsafe_rephrase_modes,
    )
    print(f"Saved v2 boundary probes: {out}")


def _prepare_v2_attack_config(v2: V2Config, base: Config, samples) -> None:
    base.attack.suffix = v2.text_attack.suffix_init
    expanded = []
    for sample in samples[: min(len(samples), 32)]:
        expanded.extend(expanded_suffix_candidates(sample.prompt))
    seen = set()
    merged = []
    for suffix in [v2.text_attack.suffix_init, *base.attack.suffix_candidates, *expanded]:
        if suffix and suffix not in seen:
            seen.add(suffix)
            merged.append(suffix)
    base.attack.suffix_candidates = merged


def cmd_attack(args) -> None:
    v2, base = V2Config.from_json(args.config)
    base.validate_paths()
    samples = _samples(base, args.limit)
    _prepare_v2_attack_config(v2, base, samples)
    boundary = Path(args.boundary or (base.output_path / "boundary_probes_v2.pt"))
    if not boundary.exists():
        raise FileNotFoundError(f"Missing v2 boundary probes: {boundary}. Run `jailbound_v2 probe` first.")
    out = run_attack(base, samples, boundary, resume=args.resume)
    print(f"Saved v2 attack results: {out}")


def cmd_run(args) -> None:
    v2, base = V2Config.from_json(args.config)
    base.validate_paths(require_guard=True)
    samples = _samples(base, args.limit)
    _prepare_v2_attack_config(v2, base, samples)
    boundary = base.output_path / "boundary_probes_v2.pt"
    if not args.resume or not boundary.exists():
        boundary = probe_boundaries_v2(
            base,
            samples,
            safe_modes=v2.probe.safe_pair_modes,
            unsafe_modes=v2.probe.unsafe_rephrase_modes,
        )
    attack_results = run_attack(base, samples, boundary, resume=args.resume)
    evaluate_results(base, attack_results)


def cmd_analyze(args) -> None:
    rows = read_jsonl(args.input)
    summary = summarize_by_category(rows)
    out_dir = Path(args.output)
    csv_path = write_category_csv(summary, out_dir / "category_summary.csv")
    md_path = write_markdown_report(summary, out_dir / "category_report.md")
    print(f"Saved category CSV: {csv_path}")
    print(f"Saved markdown report: {md_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="JailBound v2 experimental commands.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_probe = sub.add_parser("probe", help="Run matched safe/unsafe probing.")
    p_probe.add_argument("--config", default="jailbound_v2/configs/qwen25vl_v2.json")
    p_probe.add_argument("--limit", type=int, default=None)
    p_probe.set_defaults(func=cmd_probe)

    p_attack = sub.add_parser("attack", help="Run v2 attack using matched boundary probes.")
    p_attack.add_argument("--config", default="jailbound_v2/configs/qwen25vl_v2.json")
    p_attack.add_argument("--limit", type=int, default=None)
    p_attack.add_argument("--boundary", default=None)
    p_attack.add_argument("--resume", action="store_true")
    p_attack.set_defaults(func=cmd_attack)

    p_run = sub.add_parser("run", help="Run v2 probe, attack, and Qwen3Guard evaluation.")
    p_run.add_argument("--config", default="jailbound_v2/configs/qwen25vl_v2.json")
    p_run.add_argument("--limit", type=int, default=None)
    p_run.add_argument("--resume", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_analyze = sub.add_parser("analyze", help="Analyze guard_eval.jsonl by category.")
    p_analyze.add_argument("--input", required=True)
    p_analyze.add_argument("--output", default="outputs/qwen25vl_jailbound_v2/analysis")
    p_analyze.set_defaults(func=cmd_analyze)

    args = parser.parse_args(argv)
    args.func(args)
