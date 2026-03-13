from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .backbones import PointNet2Backbone, SparseConvNetBackbone


class CADReconDualBackbone(nn.Module):
    def __init__(
        self,
        *,
        pointnet_kwargs: dict[str, Any] | None = None,
        sparseconv_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.pointnet = PointNet2Backbone(**(pointnet_kwargs or {}))
        self.sparseconv = SparseConvNetBackbone(**(sparseconv_kwargs or {}))

    def forward(self, *, pcd: torch.Tensor, voxel: torch.Tensor) -> dict[str, torch.Tensor]:
        pcd_feature = self.pointnet(pcd)
        voxel_feature = self.sparseconv(voxel)
        fused_feature = torch.cat([pcd_feature, voxel_feature], dim=1)
        return {
            "pcd_feature": pcd_feature,
            "voxel_feature": voxel_feature,
            "fused_feature": fused_feature,
        }
