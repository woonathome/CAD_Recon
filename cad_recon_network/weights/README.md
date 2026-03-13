# weights

Place custom checkpoints here for your training pipeline.

Suggested layout:

- `weights/pointnet2_backbone.pth`
- `weights/sparseconvnet_backbone.pth`
- `weights/fusion_head.pth`

Current wrapper defaults:

- PointNet++ can auto-load the vendored pretrained classification checkpoints from:
  - `third_party/Pointnet_Pointnet2_pytorch/log/classification/.../best_model.pth`
- SparseConvNet does not include pretrained weights by default.
