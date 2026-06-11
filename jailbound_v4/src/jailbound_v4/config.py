from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jailbound.config import Config


@dataclass
class V4ProbeConfig:
    boundary_mode: str = "matched_safe_pair"
    safe_pair_modes: list[str] = field(default_factory=lambda: ["defensive", "detection", "compliance"])
    unsafe_rephrase_modes: list[str] = field(default_factory=lambda: ["direct"])
    layers: str | list[int] | None = None
    pooling: str | None = None


@dataclass
class V4TextAttackConfig:
    mode: str = "suffix_bank"
    suffix_init: str = " Please answer the visual task directly."
    suffix_length: int = 20
    hotflip_top_k: int = 64
    steps: int = 0
    max_eval_candidates: int = 16
    readability_weight: float = 0.05
    min_improvement: float = 1e-4


@dataclass
class V4RawImageConfig:
    epsilon: float = 8 / 255
    save_adversarial_images: bool = True
    image_dir: str = "outputs/qwen25vl_jailbound_v4/adversarial_images"


@dataclass
class V4Config:
    base_config: str = "../../configs/qwen25vl_local.json"
    output_dir: str = "outputs/qwen25vl_jailbound_v4"
    probe: V4ProbeConfig = field(default_factory=V4ProbeConfig)
    text_attack: V4TextAttackConfig = field(default_factory=V4TextAttackConfig)
    raw_image_attack: V4RawImageConfig = field(default_factory=V4RawImageConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> tuple["V4Config", Config]:
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        raw["probe"] = V4ProbeConfig(**raw.get("probe", {}))
        raw["text_attack"] = V4TextAttackConfig(**raw.get("text_attack", {}))
        raw["raw_image_attack"] = V4RawImageConfig(**raw.get("raw_image_attack", {}))
        cfg = cls(**raw)
        base_path = Path(cfg.base_config)
        if not base_path.is_absolute():
            base_path = (path.parent / base_path).resolve()
        base = Config.from_json(base_path)
        base.output_dir = cfg.output_dir
        if cfg.probe.layers is not None:
            base.boundary.layers = cfg.probe.layers
        if cfg.probe.pooling is not None:
            base.boundary.pooling = cfg.probe.pooling
        return cfg, base




