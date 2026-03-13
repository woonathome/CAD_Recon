from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .brep_head import BRepHead
from .brep_scaling import denormalize_brep_features, normalize_brep_features_for_training
from .dual_backbone import CADReconDualBackbone


def _build_1d_mask(counts: torch.Tensor, max_items: int) -> torch.Tensor:
    idx = torch.arange(max_items, device=counts.device).unsqueeze(0)
    return idx < counts.unsqueeze(1)


def _masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    beta: float = 1.0,
    dim_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    loss_mask = mask
    while loss_mask.ndim < loss.ndim:
        loss_mask = loss_mask.unsqueeze(-1)
    loss_mask = loss_mask.expand_as(loss).to(loss.dtype)

    if dim_mask is not None:
        dim_mask = dim_mask.to(device=loss.device, dtype=loss.dtype)
        if loss.ndim < 1 or dim_mask.ndim != 1 or loss.shape[-1] != dim_mask.shape[0]:
            raise ValueError(
                f"Invalid dim_mask shape for loss tensor: loss={tuple(loss.shape)}, dim_mask={tuple(dim_mask.shape)}"
            )
        view_shape = [1] * (loss.ndim - 1) + [dim_mask.shape[0]]
        loss_mask = loss_mask * dim_mask.view(*view_shape)

    denom = loss_mask.sum().clamp_min(1.0)
    return (loss * loss_mask).sum() / denom


def _masked_cross_entropy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 3 or target.ndim != 2 or mask.ndim != 2:
        raise ValueError(
            f"Expected logits[B,N,C], target[B,N], mask[B,N]. Got {tuple(logits.shape)}, {tuple(target.shape)}, {tuple(mask.shape)}"
        )
    if logits.shape[:2] != target.shape or target.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: logits={tuple(logits.shape)}, target={tuple(target.shape)}, mask={tuple(mask.shape)}"
        )

    valid = mask
    if not bool(valid.any()):
        return logits.sum() * 0.0

    selected_target = target[valid]
    num_classes = int(logits.shape[-1])
    min_idx = int(selected_target.min().item())
    max_idx = int(selected_target.max().item())
    if min_idx < 0 or max_idx >= num_classes:
        raise ValueError(
            f"Class index out of range for cross entropy. target min/max=({min_idx}, {max_idx}), num_classes={num_classes}"
        )
    return F.cross_entropy(logits[valid], selected_target, reduction="mean")


def _masked_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or target.ndim != 2 or mask.ndim != 2:
        raise ValueError(
            f"Expected logits[B,N], target[B,N], mask[B,N]. Got {tuple(logits.shape)}, {tuple(target.shape)}, {tuple(mask.shape)}"
        )
    if logits.shape != target.shape or target.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: logits={tuple(logits.shape)}, target={tuple(target.shape)}, mask={tuple(mask.shape)}"
        )

    valid = mask
    if not bool(valid.any()):
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[valid], target[valid], reduction="mean")


class CADReconBRepModel(nn.Module):
    """
    End-to-end model:
    pcd/voxel -> dual backbones -> attention-based B-rep head -> B-rep targets.

    Coordinate-only B-rep scaling is applied in compute_loss(), right before
    feature loss calculation, using max_mesh_dist.
    """

    DEFAULT_LOSS_WEIGHTS = {
        "v_feat": 1.0,
        "e_feat": 1.0,
        "f_feat": 1.0,
        "e_type": 0.5,
        "f_type": 0.5,
        "e_ori": 0.25,
        "f_ori": 0.25,
        "adj_ev": 0.5,
        "adj_fe": 0.5,
        "counts": 0.2,
    }

    E_TYPE_INDEX = 0
    E_ORI_INDEX = 11
    F_TYPE_INDEX = 0
    F_ORI_INDEX = 13

    def __init__(
        self,
        *,
        pointnet_kwargs: dict[str, Any] | None = None,
        sparseconv_kwargs: dict[str, Any] | None = None,
        head_kwargs: dict[str, Any] | None = None,
        regression_target_clip: float = 200.0,
    ) -> None:
        super().__init__()
        pointnet_kwargs = dict(pointnet_kwargs or {})
        sparseconv_kwargs = dict(sparseconv_kwargs or {})
        head_kwargs = dict(head_kwargs or {})

        self.backbone = CADReconDualBackbone(
            pointnet_kwargs=pointnet_kwargs,
            sparseconv_kwargs=sparseconv_kwargs,
        )
        self.regression_target_clip = float(regression_target_clip)

        pcd_feature_dim = int(pointnet_kwargs.get("feature_dim", 1024))
        voxel_feature_dim = int(sparseconv_kwargs.get("feature_dim", 256))
        fused_feature_dim = pcd_feature_dim + voxel_feature_dim

        forced_dims = {
            "pcd_feature_dim": pcd_feature_dim,
            "voxel_feature_dim": voxel_feature_dim,
            "fused_feature_dim": fused_feature_dim,
        }
        for key, val in forced_dims.items():
            if key in head_kwargs and int(head_kwargs[key]) != val:
                raise ValueError(
                    f"head_kwargs[{key}]={head_kwargs[key]} does not match backbone output dim {val}."
                )
            head_kwargs[key] = val

        self.head = BRepHead(**head_kwargs)

        e_dim = int(self.head.edge_feat_head.out_features)
        f_dim = int(self.head.face_feat_head.out_features)
        e_reg_dim_mask = torch.ones(e_dim, dtype=torch.float32)
        f_reg_dim_mask = torch.ones(f_dim, dtype=torch.float32)
        if self.E_TYPE_INDEX < e_dim:
            e_reg_dim_mask[self.E_TYPE_INDEX] = 0.0
        if self.E_ORI_INDEX < e_dim:
            e_reg_dim_mask[self.E_ORI_INDEX] = 0.0
        if self.F_TYPE_INDEX < f_dim:
            f_reg_dim_mask[self.F_TYPE_INDEX] = 0.0
        if self.F_ORI_INDEX < f_dim:
            f_reg_dim_mask[self.F_ORI_INDEX] = 0.0
        self.register_buffer("e_feat_reg_dim_mask", e_reg_dim_mask, persistent=False)
        self.register_buffer("f_feat_reg_dim_mask", f_reg_dim_mask, persistent=False)

    def forward(self, *, pcd: torch.Tensor, voxel: torch.Tensor) -> dict[str, torch.Tensor]:
        backbone_out = self.backbone(pcd=pcd, voxel=voxel)
        head_out = self.head(
            pcd_feature=backbone_out["pcd_feature"],
            voxel_feature=backbone_out["voxel_feature"],
            fused_feature=backbone_out["fused_feature"],
        )
        return {**backbone_out, **head_out}

    def compute_loss(
        self,
        predictions: dict[str, torch.Tensor],
        batch: dict[str, Any],
        *,
        normalize_targets: bool = True,
        loss_weights: dict[str, float] | None = None,
        beta: float = 1.0,
        eps: float = 1e-8,
    ) -> dict[str, torch.Tensor]:
        device = predictions["v_feat"].device
        loss_dtype = torch.float32

        pred_v = predictions["v_feat"].to(dtype=loss_dtype)
        pred_e = predictions["e_feat"].to(dtype=loss_dtype)
        pred_f = predictions["f_feat"].to(dtype=loss_dtype)
        pred_e_type_logits = predictions["e_type_logits"].to(dtype=loss_dtype)
        pred_f_type_logits = predictions["f_type_logits"].to(dtype=loss_dtype)
        pred_e_ori_logits = predictions["e_ori_logits"].to(dtype=loss_dtype)
        pred_f_ori_logits = predictions["f_ori_logits"].to(dtype=loss_dtype)
        pred_adj_ev = predictions["adj_ev"].to(dtype=loss_dtype)
        pred_adj_fe = predictions["adj_fe"].to(dtype=loss_dtype)
        pred_counts = predictions["counts"].to(dtype=loss_dtype)

        target_v = batch["v_feat"].to(device=device, dtype=loss_dtype)
        target_e = batch["e_feat"].to(device=device, dtype=loss_dtype)
        target_f = batch["f_feat"].to(device=device, dtype=loss_dtype)
        target_adj_ev = batch["adj_ev"].to(device=device, dtype=loss_dtype)
        target_adj_fe = batch["adj_fe"].to(device=device, dtype=loss_dtype)
        target_counts_raw = batch["counts"].to(device=device, dtype=torch.long)

        if normalize_targets:
            if "max_mesh_dist" not in batch:
                raise KeyError("batch must contain `max_mesh_dist` when normalize_targets=True.")
            target_v, target_e, target_f = normalize_brep_features_for_training(
                v_feat=target_v,
                e_feat=target_e,
                f_feat=target_f,
                max_mesh_dist=batch["max_mesh_dist"],
                eps=eps,
            )

        max_v = int(pred_v.shape[1])
        max_e = int(pred_e.shape[1])
        max_f = int(pred_f.shape[1])

        if target_v.shape[1] < max_v or target_e.shape[1] < max_e or target_f.shape[1] < max_f:
            raise ValueError(
                "Target tensor capacities are smaller than model head capacities. "
                f"target_v={tuple(target_v.shape)}, target_e={tuple(target_e.shape)}, target_f={tuple(target_f.shape)}, "
                f"required=({max_v}, {max_e}, {max_f})."
            )

        target_v = target_v[:, :max_v, :]
        target_e = target_e[:, :max_e, :]
        target_f = target_f[:, :max_f, :]
        target_adj_ev = target_adj_ev[:, :max_e, :max_v]
        target_adj_fe = target_adj_fe[:, :max_f, :max_e]

        target_e_type = target_e[:, :, self.E_TYPE_INDEX].round().to(dtype=torch.long)
        target_f_type = target_f[:, :, self.F_TYPE_INDEX].round().to(dtype=torch.long)
        target_e_ori = target_e[:, :, self.E_ORI_INDEX].clamp(0.0, 1.0)
        target_f_ori = target_f[:, :, self.F_ORI_INDEX].clamp(0.0, 1.0)

        target_e_type = target_e_type.clamp(0, self.head.num_curve_types - 1)
        target_f_type = target_f_type.clamp(0, self.head.num_surface_types - 1)

        if self.regression_target_clip > 0:
            clip_v = self.regression_target_clip
            target_v = target_v.clamp(-clip_v, clip_v)
            target_e = target_e.clamp(-clip_v, clip_v)
            target_f = target_f.clamp(-clip_v, clip_v)

        num_v = target_counts_raw[:, 0].clamp_min(0).clamp_max(max_v)
        num_e = target_counts_raw[:, 1].clamp_min(0).clamp_max(max_e)
        num_f = target_counts_raw[:, 2].clamp_min(0).clamp_max(max_f)

        v_mask = _build_1d_mask(num_v, max_v)
        e_mask = _build_1d_mask(num_e, max_e)
        f_mask = _build_1d_mask(num_f, max_f)
        ev_mask = e_mask.unsqueeze(-1) & v_mask.unsqueeze(1)
        fe_mask = f_mask.unsqueeze(-1) & e_mask.unsqueeze(1)

        loss_v = _masked_smooth_l1(pred_v, target_v, v_mask, beta=beta)
        loss_e = _masked_smooth_l1(
            pred_e,
            target_e,
            e_mask,
            beta=beta,
            dim_mask=self.e_feat_reg_dim_mask,
        )
        loss_f = _masked_smooth_l1(
            pred_f,
            target_f,
            f_mask,
            beta=beta,
            dim_mask=self.f_feat_reg_dim_mask,
        )
        loss_e_type = _masked_cross_entropy(pred_e_type_logits, target_e_type, e_mask)
        loss_f_type = _masked_cross_entropy(pred_f_type_logits, target_f_type, f_mask)
        loss_e_ori = _masked_bce_with_logits(pred_e_ori_logits, target_e_ori, e_mask)
        loss_f_ori = _masked_bce_with_logits(pred_f_ori_logits, target_f_ori, f_mask)
        loss_adj_ev = _masked_smooth_l1(pred_adj_ev, target_adj_ev, ev_mask, beta=beta)
        loss_adj_fe = _masked_smooth_l1(pred_adj_fe, target_adj_fe, fe_mask, beta=beta)

        target_counts = torch.stack([num_v, num_e, num_f], dim=1).to(dtype=loss_dtype)
        max_counts = self.head.max_counts.to(device=device, dtype=loss_dtype).unsqueeze(0)
        pred_counts_norm = pred_counts / (max_counts + eps)
        target_counts_norm = target_counts / (max_counts + eps)
        loss_counts = F.smooth_l1_loss(pred_counts_norm, target_counts_norm, beta=beta)

        weights = dict(self.DEFAULT_LOSS_WEIGHTS)
        if loss_weights is not None:
            weights.update(loss_weights)

        total = (
            weights["v_feat"] * loss_v
            + weights["e_feat"] * loss_e
            + weights["f_feat"] * loss_f
            + weights["e_type"] * loss_e_type
            + weights["f_type"] * loss_f_type
            + weights["e_ori"] * loss_e_ori
            + weights["f_ori"] * loss_f_ori
            + weights["adj_ev"] * loss_adj_ev
            + weights["adj_fe"] * loss_adj_fe
            + weights["counts"] * loss_counts
        )

        return {
            "total": total,
            "v_feat": loss_v,
            "e_feat": loss_e,
            "f_feat": loss_f,
            "e_type": loss_e_type,
            "f_type": loss_f_type,
            "e_ori": loss_e_ori,
            "f_ori": loss_f_ori,
            "adj_ev": loss_adj_ev,
            "adj_fe": loss_adj_fe,
            "counts": loss_counts,
        }

    def forward_with_loss(
        self,
        batch: dict[str, Any],
        *,
        normalize_targets: bool = True,
        loss_weights: dict[str, float] | None = None,
        beta: float = 1.0,
        eps: float = 1e-8,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        predictions = self(
            pcd=batch["pcd"],
            voxel=batch["voxel"],
        )
        loss_dict = self.compute_loss(
            predictions,
            batch,
            normalize_targets=normalize_targets,
            loss_weights=loss_weights,
            beta=beta,
            eps=eps,
        )
        return predictions, loss_dict

    def denormalize_prediction_features(
        self,
        predictions: dict[str, torch.Tensor],
        *,
        max_mesh_dist: torch.Tensor | float,
        eps: float = 1e-8,
    ) -> dict[str, torch.Tensor]:
        v_denorm, e_denorm, f_denorm = denormalize_brep_features(
            v_feat=predictions["v_feat"],
            e_feat=predictions["e_feat"],
            f_feat=predictions["f_feat"],
            max_mesh_dist=max_mesh_dist,
            eps=eps,
        )
        out = dict(predictions)
        out["v_feat"] = v_denorm
        out["e_feat"] = e_denorm
        out["f_feat"] = f_denorm
        return out
