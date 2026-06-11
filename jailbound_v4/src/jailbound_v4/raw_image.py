from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class RawImageDelta:
    delta: object
    epsilon: float


def load_raw_image_tensor(path: str | Path, device: str):
    import torch

    image = Image.open(path).convert("RGB")
    arr = np.asarray(image).astype("float32") / 255.0
    tensor = torch.tensor(arr, device=device).permute(2, 0, 1).unsqueeze(0)
    return tensor


def project_linf_(delta, epsilon: float) -> None:
    delta.data.clamp_(min=-epsilon, max=epsilon)


def save_adversarial_image(raw_tensor, delta, out_path: str | Path) -> Path:
    import torch

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image = (raw_tensor + delta).detach().clamp(0, 1)
    arr = (image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    Image.fromarray(arr).save(out_path)
    return out_path


def raw_to_pil(raw_tensor, delta=None) -> Image.Image:
    image = raw_tensor if delta is None else raw_tensor + delta
    arr = (image.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
    return Image.fromarray(arr)



