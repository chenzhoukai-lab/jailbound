from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jailbound.config import Config


@dataclass
class V3ProbeConfig:
    safe_pair_modes: list[str] = field(default_factory=lambda: ["defensive", "detection", "compliance"])
    unsafe_rephrase_modes: list[str] = field(default_factory=lambda: ["direct"])
    layers: str | list[int] | None = None
    pooling: str | None = None


@dataclass
class V3TextAttackConfig:
    suffix_init: str = " Please answer the visual task directly."
    suffix_length: int = 20
    hotflip_top_k: int = 32
    steps: int = 0


@dataclass
class V3RawImageConfig:
    epsilon: float = 8 / 255
    save_adversarial_images: bool = True
    image_dir: str = "outputs/qwen25vl_jailbound_v3/adversarial_images"


@dataclass
class V3Config:
    base_config: str = "../../configs/qwen25vl_local.json"
    output_dir: str = "outputs/qwen25vl_jailbound_v3"
    probe: V3ProbeConfig = field(default_factory=V3ProbeConfig)
    text_attack: V3TextAttackConfig = field(default_factory=V3TextAttackConfig)
    raw_image_attack: V3RawImageConfig = field(default_factory=V3RawImageConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> tuple["V3Config", Config]:
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        raw["probe"] = V3ProbeConfig(**raw.get("probe", {}))
        raw["text_attack"] = V3TextAttackConfig(**raw.get("text_attack", {}))
        raw["raw_image_attack"] = V3RawImageConfig(**raw.get("raw_image_attack", {}))
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



