from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _unwrap_state_dict(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(raw)}")

    for key in ("model_state_dict", "model_state", "state_dict", "model"):
        if key in raw and isinstance(raw[key], dict):
            return raw[key]
    return raw


def _strip_prefix(state_dict: dict[str, Any], prefix: str) -> dict[str, Any]:
    plen = len(prefix)
    return {k[plen:] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


def load_checkpoint(module: nn.Module, checkpoint_path: str | Path, strict: bool = False) -> None:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        raw = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        # torch>=2.6 uses weights_only=True by default; old checkpoints can require full unpickling.
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state = _strip_prefix(_unwrap_state_dict(raw), "module.")
    module.load_state_dict(state, strict=strict)
