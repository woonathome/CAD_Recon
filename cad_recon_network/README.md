# cad_recon_network

`pcd -> PointNet++ backbone`  
`voxel -> SparseConvNet backbone`  
`concat(pcd_feature, voxel_feature) -> downstream B-rep head`

This folder is prepared for your multimodal CAD reconstruction pipeline based on:

- `cad_recon_lib` dataset outputs:
  - `pcd`: `[B, N, 3]`
  - `voxel`: `[B, 1, R, R, R]`
  - `B-rep targets`: `v_feat/e_feat/f_feat/adj_ev/adj_fe/counts`
- third-party backbones:
  - `Pointnet_Pointnet2_pytorch`
  - `SparseConvNet`

## Structure

- `backbones/pointnet2_backbone.py`
- `backbones/sparseconvnet_backbone.py`
- `dual_backbone.py`
- `scripts/install_backbones.ps1`
- `scripts/smoke_test_backbones.py`
- `third_party/Pointnet_Pointnet2_pytorch`
- `third_party/SparseConvNet`
- `weights/`

## Build (GPU)

Activate your CUDA conda env first (for your case: `cadgen`).

```powershell
powershell -ExecutionPolicy Bypass -File cad_recon_network/scripts/install_backbones.ps1 -PythonExe "python"
```

If your GPU arch list must be fixed manually:

```powershell
powershell -ExecutionPolicy Bypass -File cad_recon_network/scripts/install_backbones.ps1 -PythonExe "python" -TorchCudaArchList "8.9"
```

## Smoke Test (GPU)

```powershell
python cad_recon_network/scripts/smoke_test_backbones.py --device cuda
```

## Minimal Use

```python
import torch
from cad_recon_network import CADReconDualBackbone

model = CADReconDualBackbone(
    pointnet_kwargs={"variant": "ssg", "use_normals": False, "feature_dim": 1024},
    sparseconv_kwargs={"voxel_resolution": 64, "in_channels": 1, "feature_dim": 256, "allow_fallback": False},
).cuda()

pcd = torch.randn(2, 2048, 3, device="cuda")
voxel = torch.zeros(2, 1, 64, 64, 64, device="cuda")
out = model(pcd=pcd, voxel=voxel)

print(out["pcd_feature"].shape)    # [2, 1024]
print(out["voxel_feature"].shape)  # [2, 256]
print(out["fused_feature"].shape)  # [2, 1280]
```

## End-to-End B-rep Model (forward + loss)

```python
import torch
from cad_recon_network import CADReconBRepModel

model = CADReconBRepModel(
    pointnet_kwargs={"variant": "ssg", "use_normals": False, "feature_dim": 1024},
    sparseconv_kwargs={"voxel_resolution": 64, "in_channels": 1, "feature_dim": 256, "allow_fallback": False},
    head_kwargs={
        "max_vertices": 4000,
        "max_edges": 2000,
        "max_faces": 1000,
        "hidden_dim": 256,
        "num_decoder_layers": 2,
    },
).cuda()

batch = {
    "pcd": torch.randn(2, 2048, 3, device="cuda"),
    "voxel": torch.zeros(2, 1, 64, 64, 64, device="cuda"),
    "v_feat": torch.zeros(2, 4000, 3, device="cuda"),
    "e_feat": torch.zeros(2, 2000, 73, device="cuda"),
    "f_feat": torch.zeros(2, 1000, 174, device="cuda"),
    "adj_ev": torch.zeros(2, 2000, 4000, device="cuda"),
    "adj_fe": torch.zeros(2, 1000, 2000, device="cuda"),
    "counts": torch.tensor([[120, 260, 140], [90, 180, 110]], device="cuda"),
    "max_mesh_dist": torch.tensor([0.85, 1.10], device="cuda"),  # used for coordinate-only GT scaling
}

pred, loss_dict = model.forward_with_loss(batch, normalize_targets=True)
print(loss_dict["total"].item())
```

`normalize_targets=True` applies coordinate-only normalization to GT `v/e/f_feat`
using `max_mesh_dist` right before computing losses.

## Train

```powershell
python cad_recon_network/scripts/train_brep_model.py `
  --base-dir "D:\ABC_dataset\obj_chunks" `
  --output-dir "cad_recon_network/weights/brep_runs/exp1" `
  --device cuda `
  --epochs 50 `
  --batch-size 1 `
  --num-workers 4 `
  --max-view-retry 8 `
  --regression-target-clip 200
```

Notes:
- `batch-size=1` is supported by default because PointNet BatchNorm is kept in eval mode during training.
- Checkpoints are saved to `last.pt` (periodic) and `best.pt` (best validation loss).
- `e_feat/f_feat` losses are split: `type` and `orientation` use classification losses, remaining indices use regression loss.
- Epoch-wise term losses are saved to `epoch_losses.csv` (train/val columns for each loss term).
- Camera-miss cases (`pcd hit=0`) are retried and then fallback to mesh point sampling, so DataLoader no longer crashes on empty PCD.
