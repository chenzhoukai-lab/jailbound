from __future__ import annotations

import argparse
from pathlib import Path

from jailbound.config import Config
from jailbound.dataset import load_mm_safetybench

from .analyze import read_jsonl, summarize_by_category, write_category_csv, write_markdown_report
from .boundary import probe_boundaries_v2
from .config import V2Config


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

    p_analyze = sub.add_parser("analyze", help="Analyze guard_eval.jsonl by category.")
    p_analyze.add_argument("--input", required=True)
    p_analyze.add_argument("--output", default="outputs/qwen25vl_jailbound_v2/analysis")
    p_analyze.set_defaults(func=cmd_analyze)

    args = parser.parse_args(argv)
    args.func(args)

