from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

from ..paths import POINTNET2_MODELS_ROOT, POINTNET2_REPO_ROOT
from .checkpoint import load_checkpoint

PointNet2Variant = Literal["ssg", "msg"]

_DEFAULT_CHECKPOINTS = {
    ("ssg", False): POINTNET2_REPO_ROOT
    / "log"
    / "classification"
    / "pointnet2_ssg_wo_normals"
    / "checkpoints"
    / "best_model.pth",
    ("msg", True): POINTNET2_REPO_ROOT
    / "log"
    / "classification"
    / "pointnet2_msg_normals"
    / "checkpoints"
    / "best_model.pth",
}


def _ensure_pointnet_paths() -> None:
    for path in (POINTNET2_REPO_ROOT, POINTNET2_MODELS_ROOT):
        p = str(path)
        if p not in sys.path:
            sys.path.insert(0, p)


def _resolve_model_module(variant: PointNet2Variant):
    _ensure_pointnet_paths()
    if variant == "ssg":
        return importlib.import_module("pointnet2_cls_ssg")
    if variant == "msg":
        return importlib.import_module("pointnet2_cls_msg")
    raise ValueError(f"Unsupported PointNet++ variant: {variant}")


class PointNet2Backbone(nn.Module):
    def __init__(
        self,
        *,
        variant: PointNet2Variant = "ssg",
        use_normals: bool = False,
        num_classes: int = 40,
        feature_dim: int = 1024,
        checkpoint_path: Path | str | None = None,
        use_default_checkpoint: bool = True,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        model_module = _resolve_model_module(variant)
        self.variant = variant
        self.use_normals = use_normals
        self.model = model_module.get_model(num_class=num_classes, normal_channel=use_normals)
        self.proj = nn.Identity() if feature_dim == 1024 else nn.Linear(1024, feature_dim)

        ckpt = Path(checkpoint_path) if checkpoint_path is not None else None
        if ckpt is None and use_default_checkpoint:
            ckpt = _DEFAULT_CHECKPOINTS.get((variant, use_normals))
        if ckpt is not None and ckpt.exists():
            load_checkpoint(self.model, ckpt, strict=False)

        if freeze:
            for p in self.model.parameters():
                p.requires_grad = False

    def _prepare_points(self, pcd: torch.Tensor) -> torch.Tensor:
        if pcd.dim() != 3:
            raise ValueError(f"`pcd` must be [B,N,C] or [B,C,N], got shape={tuple(pcd.shape)}")

        if pcd.shape[1] in (3, 6):
            channels = pcd.shape[1]
            points = pcd
        elif pcd.shape[2] in (3, 6):
            channels = pcd.shape[2]
            points = pcd.transpose(1, 2).contiguous()
        else:
            raise ValueError(
                "`pcd` must have 3 or 6 channels. Expected [B,N,3/6] or [B,3/6,N], "
                f"got shape={tuple(pcd.shape)}"
            )

        expected = 6 if self.use_normals else 3
        if channels != expected:
            raise ValueError(
                f"PointNet2Backbone(use_normals={self.use_normals}) expects {expected} channels, got {channels}"
            )
        return points

    def forward(self, pcd: torch.Tensor) -> torch.Tensor:
        points = self._prepare_points(pcd)
        points = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)

        # PointNet++ set-abstraction indexing is significantly more stable in FP32.
        # Keep this path out of autocast to avoid occasional CUDA index asserts.
        if points.device.type == "cuda":
            with torch.autocast(device_type="cuda", enabled=False):
                _, global_token = self.model(points.float())
        else:
            _, global_token = self.model(points.float())

        feat = global_token.squeeze(-1).contiguous()
        return self.proj(feat)
