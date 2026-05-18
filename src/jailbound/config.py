from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _expand(value: str) -> str:
    return os.path.expandvars(os.path.expanduser(value))


@dataclass
class BoundaryConfig:
    # layers 控制论文里“在哪些融合层训练安全边界分类器”。
    # 推荐先用 last_10：显存/时间压力比 all 小很多，也更接近论文中“深层更可分”的观察。
    layers: str | list[int] = "last_10"
    # 从每层 hidden states 里取哪个向量作为 h(l)。Qwen2.5-VL 的 HF 接口没有直接叫 fusion layer，
    # 所以这里用多模态 token 融合后的 decoder hidden states 近似论文里的 h(l)。
    pooling: str = "last_token"
    # MM-SafetyBench 里原始 prompt 是 unsafe；为了训练二分类 probe，
    # 我们用同一张图配一个安全描述 prompt 作为 safe 对照样本。
    safe_prompt: str = "Describe the image briefly and safely."
    p0: float = 0.3
    epochs: int = 200
    learning_rate: float = 0.05
    weight_decay: float = 1e-4


@dataclass
class AttackConfig:
    max_samples: int | None = None
    iterations: int = 120
    visual_lr: float = 1e-3
    # 论文里视觉扰动约束是 8/255。这里作用在 processor 后的 pixel_values 上，
    # 目标是复现边界穿越机制，而不是导出肉眼不可见的最终图片文件。
    pixel_epsilon: float = 8 / 255
    lambda_sem: float = 2.0
    lambda_geo: float = 1.0
    boundary_direction: float = -1.0
    suffix: str = ""
    # 文本攻击部分先做成“候选 suffix 选择”：每个候选后缀都算一次边界损失，
    # 选最容易跨边界的那个。它比论文的 token-level 梯度替换更容易读和调试。
    suffix_candidates: list[str] = field(default_factory=list)
    generate: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalConfig:
    batch_size: int = 4
    max_new_tokens: int = 32


@dataclass
class Config:
    dataset_root: str
    dataset_name: str = "mm-safetybench"
    image_format: str = "images"
    target_model_path: str = ""
    guard_model_path: str = ""
    output_dir: str = "outputs/qwen25vl_jailbound"
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    attn_implementation: str | None = "flash_attention_2"
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        raw["boundary"] = BoundaryConfig(**raw.get("boundary", {}))
        raw["attack"] = AttackConfig(**raw.get("attack", {}))
        raw["eval"] = EvalConfig(**raw.get("eval", {}))
        cfg = cls(**raw)
        cfg.dataset_root = _expand(cfg.dataset_root)
        cfg.target_model_path = _expand(cfg.target_model_path)
        cfg.guard_model_path = _expand(cfg.guard_model_path)
        cfg.output_dir = _expand(cfg.output_dir)
        return cfg

    def validate_paths(self, require_guard: bool = False) -> None:
        # 在真正加载大模型前先检查路径。这样路径没填时不会等到 torch/transformers 报很长的错。
        missing = []
        for label, value in [
            ("dataset_root", self.dataset_root),
            ("target_model_path", self.target_model_path),
        ]:
            if not value or "C:/path/to" in value or not Path(value).exists():
                missing.append(f"{label}={value!r}")
        if require_guard and (
            not self.guard_model_path
            or "C:/path/to" in self.guard_model_path
            or not Path(self.guard_model_path).exists()
        ):
            missing.append(f"guard_model_path={self.guard_model_path!r}")
        if missing:
            joined = "\n  - ".join(missing)
            raise FileNotFoundError(f"Please set valid local paths in config:\n  - {joined}")

    @property
    def output_path(self) -> Path:
        path = Path(self.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path
