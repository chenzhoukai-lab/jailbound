from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SafetySample:
    sample_id: str
    category: str
    image_path: Path
    prompt: str
    raw: dict


def _candidate_image_dirs(category_dir: Path, image_format: str) -> list[Path]:
    if image_format == "auto":
        preferred = ["images", "images_figstep", "images_qr", "images_wr", "images_rotate", "images_mirror", "images_base64"]
        dirs = [category_dir / name for name in preferred if (category_dir / name).is_dir()]
        dirs.extend(sorted(p for p in category_dir.iterdir() if p.is_dir() and p.name.startswith("images_") and p not in dirs))
        return dirs
    return [category_dir / image_format]


def _candidate_image_paths(category_dir: Path, image_format: str, sample_id: str) -> Iterable[Path]:
    image_dirs = _candidate_image_dirs(category_dir, image_format)
    for image_dir in image_dirs:
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            yield image_dir / f"{sample_id}{ext}"


def _prompt_from_item(item: dict) -> str:
    for key in ("original_prompt", "prompt", "question", "text", "instruction"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise KeyError(f"Cannot find prompt field in item keys={sorted(item)}")


def _id_from_item(item: dict, index: int) -> str:
    for key in ("id", "image_id", "idx", "sample_id"):
        value = item.get(key)
        if value is not None:
            return str(value)
    return str(index)


def find_category_dirs(dataset_root: str | Path) -> list[Path]:
    root = Path(dataset_root)
    if (root / "data.json").exists():
        return [root]
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "data.json").exists())


def load_mm_safetybench(
    dataset_root: str | Path,
    image_format: str = "images",
    limit: int | None = None,
) -> list[SafetySample]:
    samples: list[SafetySample] = []
    category_dirs = find_category_dirs(dataset_root)
    if not category_dirs:
        raise FileNotFoundError(f"No MM-SafetyBench data.json files found under {dataset_root}")
    for category_dir in category_dirs:
        with open(category_dir / "data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = list(data.values())
        for index, item in enumerate(data):
            sample_id = _id_from_item(item, index)
            prompt = _prompt_from_item(item)
            image_path = next((p for p in _candidate_image_paths(category_dir, image_format, sample_id) if p.exists()), None)
            if image_path is None:
                available = sorted(p.name for p in category_dir.iterdir() if p.is_dir() and p.name.startswith("images"))
                raise FileNotFoundError(
                    f"Missing image for id={sample_id!r} in {category_dir / image_format}. "
                    f"Available image folders: {available}"
                )
            samples.append(
                SafetySample(
                    sample_id=sample_id,
                    category=category_dir.name,
                    image_path=image_path,
                    prompt=prompt,
                    raw=item,
                )
            )
            if limit is not None and len(samples) >= limit:
                return samples
    return samples
