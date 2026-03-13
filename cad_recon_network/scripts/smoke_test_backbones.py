from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cad_recon_network import CADReconDualBackbone
from cad_recon_network.backbones import PointNet2Backbone, SparseConvNetBackbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for PointNet++ + SparseConvNet backbones.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--voxel-res", type=int, default=64)
    parser.add_argument("--skip-sparse", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Activate a CUDA-enabled conda env and retry.")

    pcd = torch.randn(args.batch_size, args.num_points, 3, device=device)
    voxel = torch.zeros(args.batch_size, 1, args.voxel_res, args.voxel_res, args.voxel_res, device=device)
    voxel[:, :, 10:20, 10:20, 10:20] = 1.0

    pointnet = PointNet2Backbone(variant="ssg", use_normals=False, feature_dim=1024).to(device).eval()
    with torch.no_grad():
        p_feat = pointnet(pcd)
    print("pointnet feature:", tuple(p_feat.shape), "device:", p_feat.device)

    if args.skip_sparse:
        return

    sparse = SparseConvNetBackbone(
        voxel_resolution=args.voxel_res,
        in_channels=1,
        feature_dim=256,
        allow_fallback=False,
    ).to(device).eval()
    with torch.no_grad():
        v_feat = sparse(voxel)
    print("sparseconv feature:", tuple(v_feat.shape), "device:", v_feat.device)

    model = CADReconDualBackbone(
        pointnet_kwargs={"variant": "ssg", "use_normals": False, "feature_dim": 1024},
        sparseconv_kwargs={
            "voxel_resolution": args.voxel_res,
            "in_channels": 1,
            "feature_dim": 256,
            "allow_fallback": False,
        },
    ).to(device).eval()
    with torch.no_grad():
        out = model(pcd=pcd, voxel=voxel)
    print("fused feature:", tuple(out["fused_feature"].shape), "device:", out["fused_feature"].device)


if __name__ == "__main__":
    main()
