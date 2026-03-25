from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from ..paths import SPARSECONVNET_REPO_ROOT
from .checkpoint import load_checkpoint


def _import_sparseconvnet():
    try:
        import sparseconvnet as scn  # type: ignore

        return scn
    except Exception:
        repo_path = str(SPARSECONVNET_REPO_ROOT)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        import sparseconvnet as scn  # type: ignore

        return scn


class _DenseVoxelFallbackBackbone(nn.Module):
    def __init__(self, *, in_channels: int, base_channels: int, feature_dim: int) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(c1),
            nn.ReLU(inplace=True),
            nn.Conv3d(c1, c2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(c2),
            nn.ReLU(inplace=True),
            nn.Conv3d(c2, c3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(c3),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Identity() if feature_dim == c3 else nn.Linear(c3, feature_dim)

    def forward(self, voxel: torch.Tensor) -> torch.Tensor:
        dense = self.encoder(voxel)
        cx = int(dense.shape[2] // 2)
        cy = int(dense.shape[3] // 2)
        cz = int(dense.shape[4] // 2)
        center_token = dense[:, :, cx, cy, cz]
        return self.proj(center_token)


class SparseConvNetBackbone(nn.Module):
    def __init__(
        self,
        *,
        voxel_resolution: int = 64,
        in_channels: int = 1,
        base_channels: int = 32,
        feature_dim: int = 256,
        reps: int = 1,
        unet_planes: Sequence[int] = (32, 64, 96, 128),
        occupancy_threshold: float = 0.0,
        checkpoint_path: Path | str | None = None,
        allow_fallback: bool = False,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        self.voxel_resolution = voxel_resolution
        self.in_channels = in_channels
        self.occupancy_threshold = occupancy_threshold
        self.backend_name = "sparseconvnet"
        self.fallback_reason: str | None = None

        if len(unet_planes) == 0:
            raise ValueError("`unet_planes` must contain at least one entry.")
        if unet_planes[0] != base_channels:
            raise ValueError("`unet_planes[0]` must match `base_channels`.")

        self.fallback_model: nn.Module | None = None
        try:
            scn = _import_sparseconvnet()
            if not hasattr(scn, "SCN") or not hasattr(scn.SCN, "Metadata_3"):
                raise RuntimeError("SparseConvNet extension is not built correctly (missing Metadata_3).")

            spatial_size = torch.LongTensor([voxel_resolution] * 3)
            self.input_layer = scn.InputLayer(3, spatial_size, mode=4)
            self.sparse_model = (
                scn.Sequential()
                .add(scn.SubmanifoldConvolution(3, in_channels, base_channels, 3, False))
                .add(scn.UNet(3, reps, list(unet_planes), residual_blocks=False, downsample=[2, 2]))
                .add(scn.BatchNormReLU(base_channels))
            )
            self.output_layer = scn.OutputLayer(3)
            self.proj = nn.Identity() if feature_dim == base_channels else nn.Linear(base_channels, feature_dim)
        except Exception as exc:
            if not allow_fallback:
                raise RuntimeError(
                    "Failed to initialize SparseConvNet. Build GPU extension first:\n"
                    "powershell -ExecutionPolicy Bypass -File cad_recon_network/scripts/install_backbones.ps1"
                ) from exc
            self.backend_name = "fallback_dense3d"
            self.fallback_reason = str(exc)
            self.fallback_model = _DenseVoxelFallbackBackbone(
                in_channels=in_channels,
                base_channels=base_channels,
                feature_dim=feature_dim,
            )

        if checkpoint_path is not None:
            load_checkpoint(self, checkpoint_path, strict=False)

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def _prepare_voxel(self, voxel: torch.Tensor) -> torch.Tensor:
        # Supports [B, R, R, R] (assumed 1 channel) or [B, C, R, R, R].
        if voxel.dim() == 4:
            voxel = voxel.unsqueeze(1)
        elif voxel.dim() != 5:
            raise ValueError(f"`voxel` must be [B,R,R,R] or [B,C,R,R,R], got shape={tuple(voxel.shape)}")

        if voxel.size(1) != self.in_channels:
            raise ValueError(f"Expected voxel channels={self.in_channels}, got {voxel.size(1)}")

        if voxel.shape[2:] != (self.voxel_resolution, self.voxel_resolution, self.voxel_resolution):
            raise ValueError(
                f"Expected voxel size {(self.voxel_resolution,) * 3}, got {tuple(voxel.shape[2:])}"
            )
        return voxel

    def _dense_to_sparse(self, voxel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # voxel: [B, C, X, Y, Z]
        batch_size, _, _, _, _ = voxel.shape
        active = voxel.abs().amax(dim=1) > self.occupancy_threshold
        raw_idx = active.nonzero(as_tuple=False)  # [N, 4] => b,x,y,z

        coords = []
        feats = []
        anchor_rows: list[int] = []

        if raw_idx.numel() > 0:
            xyzb = torch.stack(
                [raw_idx[:, 1], raw_idx[:, 2], raw_idx[:, 3], raw_idx[:, 0]],
                dim=1,
            )
            values = voxel[
                raw_idx[:, 0],
                :,
                raw_idx[:, 1],
                raw_idx[:, 2],
                raw_idx[:, 3],
            ]
            coords.append(xyzb)
            feats.append(values)

        occupied: list[set[tuple[int, int, int]]] = [set() for _ in range(batch_size)]
        if raw_idx.numel() > 0:
            for b, x, y, z in raw_idx.cpu().tolist():
                occupied[int(b)].add((int(x), int(y), int(z)))

        running_count = int(sum(int(c.shape[0]) for c in coords))
        vmax = self.voxel_resolution - 1
        center = self.voxel_resolution // 2
        candidates = [
            (0, 0, 0),
            (0, 0, vmax),
            (0, vmax, 0),
            (vmax, 0, 0),
            (vmax, vmax, vmax),
            (vmax, vmax, 0),
            (vmax, 0, vmax),
            (0, vmax, vmax),
            (center, center, center),
        ]
        for b in range(batch_size):
            anchor_xyz = None
            occ = occupied[b]
            for cand in candidates:
                if cand not in occ:
                    anchor_xyz = cand
                    break
            if anchor_xyz is None:
                anchor_xyz = (center, center, center)

            ax, ay, az = anchor_xyz
            coords.append(torch.tensor([[ax, ay, az, b]], dtype=torch.long, device=voxel.device))
            feats.append(torch.zeros((1, self.in_channels), dtype=voxel.dtype, device=voxel.device))
            anchor_rows.append(running_count)
            running_count += 1

        all_coords = torch.cat(coords, dim=0).to(dtype=torch.long, device="cpu")
        all_feats = torch.cat(feats, dim=0).to(dtype=torch.float32)
        anchor_index = torch.tensor(anchor_rows, dtype=torch.long, device=all_feats.device)
        return all_coords, all_feats, anchor_index

    def forward(self, voxel: torch.Tensor) -> torch.Tensor:
        voxel = self._prepare_voxel(voxel)
        if self.backend_name == "fallback_dense3d":
            return self.fallback_model(voxel)  # type: ignore[misc]

        batch_size = int(voxel.shape[0])
        coords, feats, anchor_rows = self._dense_to_sparse(voxel)
        sparse_input = self.input_layer([coords, feats, batch_size])
        sparse_feat = self.sparse_model(sparse_input)
        per_input_feat = self.output_layer(sparse_feat)
        anchor_rows = anchor_rows.to(device=per_input_feat.device, dtype=torch.long)
        anchor_feat = per_input_feat.index_select(0, anchor_rows)
        return self.proj(anchor_feat)
