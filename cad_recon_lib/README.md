# cad_recon_lib

Library-ized modules extracted from `3(0)_Dataset (PCD-voxel-BRep)_v4.ipynb`.

## Modules

- `dataset.py`
  - `ABCMultiModalDataset`: loads OBJ/STEP pairs and returns:
    - simulated point cloud (`pcd`)
    - voxel tensor (`voxel`)
    - B-rep feature tensors (`v_feat`, `e_feat`, `f_feat`, `adj_ev`, `adj_fe`, `counts`)
- `visualization.py`
  - `visualize_multimodal_sample`: reproduces the notebook's sample visualization (STEP vs OBJ/PCD vs voxel)
- `reconstruction.py`
  - `build_reconstruction_geometries`: builds original/reconstructed geometry objects
  - `visualize_brep_reconstruction_comparison`: opens comparison viewer
  - `ReconstructionOptions`: runtime options (including fast-mode parameters)

## Quick Start

```python
from cad_recon_lib import (
    ABCMultiModalDataset,
    ReconstructionOptions,
    visualize_multimodal_sample,
    visualize_brep_reconstruction_comparison,
)

dataset = ABCMultiModalDataset(
    base_dir="./abc_dataset_filtered-1",
    pcd_num_points=2048,
    voxel_res=64,
)

sample = dataset[0]
visualize_multimodal_sample(dataset, sample=sample)

opts = ReconstructionOptions(fast_vis_mode=True)
visualize_brep_reconstruction_comparison(sample, options=opts)
```
