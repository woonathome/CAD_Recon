from __future__ import annotations

from typing import Tuple

import torch

# v_feat: [x, y, z]
V_COORD_SLICES = ((0, 3),)

# e_feat coordinate fields from cad_recon_lib/dataset.py:
# [3:9] start/end xyz, [13:43] bspline poles, [43:73] sampled points.
E_COORD_SLICES = ((3, 9), (13, 43), (43, 73))

# f_feat coordinate fields from cad_recon_lib/dataset.py:
# [5:8] mid xyz, [15:90] bspline poles, [90:93] local origin, [99:174] sampled grid points.
F_COORD_SLICES = ((5, 8), (15, 90), (90, 93), (99, 174))

_V_MIN_DIM = max(hi for _, hi in V_COORD_SLICES)
_E_MIN_DIM = max(hi for _, hi in E_COORD_SLICES)
_F_MIN_DIM = max(hi for _, hi in F_COORD_SLICES)


def _as_dist_vector(
    max_mesh_dist: torch.Tensor | float,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    eps: float = 1e-8,
) -> torch.Tensor:
    dist = torch.as_tensor(max_mesh_dist, device=device, dtype=dtype)
    if dist.ndim == 0:
        dist = dist.repeat(batch_size)
    elif dist.ndim == 1 and dist.shape[0] == batch_size:
        pass
    else:
        raise ValueError(
            f"`max_mesh_dist` must be scalar or [B]. Got shape={tuple(dist.shape)} for B={batch_size}."
        )
    return dist.clamp_min(eps)


def _apply_scale_by_slices(
    feat: torch.Tensor,
    *,
    dist: torch.Tensor,
    slices: Tuple[Tuple[int, int], ...],
    inverse: bool,
) -> torch.Tensor:
    out = feat.clone()
    scale = dist[:, None, None]
    for lo, hi in slices:
        if inverse:
            out[..., lo:hi] = out[..., lo:hi] * scale
        else:
            out[..., lo:hi] = out[..., lo:hi] / scale
    return out


def normalize_brep_features_for_training(
    *,
    v_feat: torch.Tensor,
    e_feat: torch.Tensor,
    f_feat: torch.Tensor,
    max_mesh_dist: torch.Tensor | float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize only coordinate components by max_mesh_dist.
    Non-coordinate components are kept unchanged.
    """
    if v_feat.ndim != 3 or e_feat.ndim != 3 or f_feat.ndim != 3:
        raise ValueError(
            "Expected batched feature tensors: v_feat[B,V,3], e_feat[B,E,73], f_feat[B,F,174]. "
            f"Got shapes v={tuple(v_feat.shape)}, e={tuple(e_feat.shape)}, f={tuple(f_feat.shape)}."
        )
    if v_feat.shape[-1] < _V_MIN_DIM or e_feat.shape[-1] < _E_MIN_DIM or f_feat.shape[-1] < _F_MIN_DIM:
        raise ValueError(
            "Feature dimension is smaller than required coordinate slice indices. "
            f"Need v>={_V_MIN_DIM}, e>={_E_MIN_DIM}, f>={_F_MIN_DIM}. "
            f"Got v={v_feat.shape[-1]}, e={e_feat.shape[-1]}, f={f_feat.shape[-1]}."
        )

    batch_size = int(v_feat.shape[0])
    dist = _as_dist_vector(
        max_mesh_dist,
        batch_size=batch_size,
        device=v_feat.device,
        dtype=v_feat.dtype,
        eps=eps,
    )
    v_norm = _apply_scale_by_slices(v_feat, dist=dist, slices=V_COORD_SLICES, inverse=False)
    e_norm = _apply_scale_by_slices(e_feat, dist=dist, slices=E_COORD_SLICES, inverse=False)
    f_norm = _apply_scale_by_slices(f_feat, dist=dist, slices=F_COORD_SLICES, inverse=False)
    return v_norm, e_norm, f_norm


def denormalize_brep_features(
    *,
    v_feat: torch.Tensor,
    e_feat: torch.Tensor,
    f_feat: torch.Tensor,
    max_mesh_dist: torch.Tensor | float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Inverse transform for normalize_brep_features_for_training.
    """
    if v_feat.ndim != 3 or e_feat.ndim != 3 or f_feat.ndim != 3:
        raise ValueError(
            "Expected batched feature tensors: v_feat[B,V,3], e_feat[B,E,73], f_feat[B,F,174]. "
            f"Got shapes v={tuple(v_feat.shape)}, e={tuple(e_feat.shape)}, f={tuple(f_feat.shape)}."
        )
    if v_feat.shape[-1] < _V_MIN_DIM or e_feat.shape[-1] < _E_MIN_DIM or f_feat.shape[-1] < _F_MIN_DIM:
        raise ValueError(
            "Feature dimension is smaller than required coordinate slice indices. "
            f"Need v>={_V_MIN_DIM}, e>={_E_MIN_DIM}, f>={_F_MIN_DIM}. "
            f"Got v={v_feat.shape[-1]}, e={e_feat.shape[-1]}, f={f_feat.shape[-1]}."
        )

    batch_size = int(v_feat.shape[0])
    dist = _as_dist_vector(
        max_mesh_dist,
        batch_size=batch_size,
        device=v_feat.device,
        dtype=v_feat.dtype,
        eps=eps,
    )
    v_denorm = _apply_scale_by_slices(v_feat, dist=dist, slices=V_COORD_SLICES, inverse=True)
    e_denorm = _apply_scale_by_slices(e_feat, dist=dist, slices=E_COORD_SLICES, inverse=True)
    f_denorm = _apply_scale_by_slices(f_feat, dist=dist, slices=F_COORD_SLICES, inverse=True)
    return v_denorm, e_denorm, f_denorm
