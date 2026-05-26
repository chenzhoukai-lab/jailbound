from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .config import Config


class SerialState:
    process_index = 0
    num_processes = 1
    is_main_process = True
    device = "cpu"

    def wait_for_everyone(self) -> None:
        return None


def get_accelerator() -> Any:
    try:
        from accelerate import Accelerator
    except ImportError:
        return SerialState()
    return Accelerator()


def runtime_device(cfg: Config, accelerator: Any) -> str:
    if accelerator.num_processes > 1 and str(cfg.device).startswith("cuda"):
        return str(accelerator.device)
    return cfg.device


def shard_items(items: list[Any], accelerator: Any) -> list[Any]:
    return items[accelerator.process_index :: accelerator.num_processes]


def reset_shard_dir(path: Path, accelerator: Any) -> None:
    if accelerator.is_main_process:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    path.mkdir(parents=True, exist_ok=True)

