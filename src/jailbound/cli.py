from __future__ import annotations

import argparse
from pathlib import Path

from .attack import run_attack
from .boundary import probe_boundaries
from .config import Config
from .dataset import load_mm_safetybench
from .guard import evaluate_results


def _samples(cfg: Config, limit: int | None):
    # 数据集路径做了多候选兼容：
    # 可以把 dataset_root 指到总目录，也可以直接指到 safebench/mm-safetybench 子目录。
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
    tried = ", ".join(str(x) for x in candidates)
    raise FileNotFoundError(f"Could not find MM-SafetyBench data.json files. Tried: {tried}")


def cmd_probe(args) -> None:
    cfg = Config.from_json(args.config)
    cfg.validate_paths()
    samples = _samples(cfg, args.limit)
    print(f"Loaded {len(samples)} samples for boundary probing.")
    out = probe_boundaries(cfg, samples)
    print(f"Saved boundary probes: {out}")


def cmd_attack(args) -> None:
    cfg = Config.from_json(args.config)
    cfg.validate_paths()
    samples = _samples(cfg, args.limit)
    print(f"Loaded {len(samples)} samples for attack.")
    out = run_attack(cfg, samples, args.boundary)
    print(f"Saved attack results: {out}")


def cmd_eval(args) -> None:
    cfg = Config.from_json(args.config)
    out = evaluate_results(cfg, args.attack_results)
    print(f"Saved guard evaluation: {out}")


def cmd_run(args) -> None:
    # 一键复现：probe -> attack -> eval。
    # 初学时建议先用 --limit 2 或 --limit 5，确认路径和显存都没问题。
    cfg = Config.from_json(args.config)
    cfg.validate_paths(require_guard=True)
    samples = _samples(cfg, args.limit)
    print(f"Loaded {len(samples)} samples for full JailBound reproduction.")
    boundary = probe_boundaries(cfg, samples)
    attack_results = run_attack(cfg, samples, boundary)
    evaluate_results(cfg, attack_results)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="JailBound reproduction on local Qwen2.5-VL and MM-SafetyBench.")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", required=True, help="Path to JSON config.")
    common.add_argument("--limit", type=int, default=None, help="Optional sample limit for quick reproduction runs.")

    p_probe = sub.add_parser("probe", parents=[common], help="Train layer-wise safety boundary probes.")
    p_probe.set_defaults(func=cmd_probe)

    p_attack = sub.add_parser("attack", parents=[common], help="Run Safety Boundary Crossing attack.")
    p_attack.add_argument("--boundary", default=None, help="Boundary probe checkpoint path.")
    p_attack.set_defaults(func=cmd_attack)

    p_eval = sub.add_parser("eval", help="Evaluate attack outputs with local Qwen3Guard.")
    p_eval.add_argument("--config", required=True, help="Path to JSON config.")
    p_eval.add_argument("--attack-results", default=None, help="Path to attack_results.jsonl.")
    p_eval.set_defaults(func=cmd_eval)

    p_run = sub.add_parser("run", parents=[common], help="Run probe, attack, and guard evaluation.")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    args.func(args)
