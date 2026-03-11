from pathlib import Path
import random

import numpy as np
import open3d as o3d

from .occ_visualization import build_step_mesh_and_wireframe


def _to_numpy(arr):
    if hasattr(arr, "detach"):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def visualize_multimodal_sample(dataset, sample=None, index=None, left_shift=-2.0, right_shift=2.0):
    """
    Visualize a random (or selected) sample:
    - STEP B-rep (primitive colored faces + black wireframe)
    - OBJ mesh + simulated PCD
    - voxel grid
    """
    if sample is None:
        sample = dataset[index] if index is not None else dataset[random.randint(0, len(dataset) - 1)]

    obj_path = Path(sample["obj_path"])
    step_path = Path(sample["step_path"])
    max_dist = float(sample["max_mesh_dist"])

    print(f"Visualizing Model ID: {sample['model_id']}")

    # A) OBJ + PCD
    orig_mesh = o3d.io.read_triangle_mesh(str(obj_path))
    orig_mesh.translate(-orig_mesh.get_center())
    orig_mesh.scale(1.0 / max_dist, center=(0, 0, 0))
    orig_mesh.paint_uniform_color([0.7, 0.7, 0.7])
    orig_mesh.translate([left_shift, 0, 0])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(_to_numpy(sample["pcd"]))
    pcd.paint_uniform_color([1, 0, 0])
    pcd.translate([left_shift, 0, 0])

    # B) Voxel points
    vox_matrix = _to_numpy(sample["voxel"]).squeeze()
    indices = np.argwhere(vox_matrix == 1.0)
    v_pts = (indices / (dataset.voxel_res - 1) - 0.5) * 2.0
    voxel_pcd = o3d.geometry.PointCloud()
    voxel_pcd.points = o3d.utility.Vector3dVector(v_pts)
    voxel_pcd.paint_uniform_color([0, 1, 0])
    voxel_pcd.translate([right_shift, 0, 0])

    # C) STEP B-rep (mesh + wireframe)
    step_mesh, step_wireframe = build_step_mesh_and_wireframe(step_path)
    step_center = step_mesh.get_center()
    step_mesh.translate(-step_center)
    step_mesh.scale(1.0 / max_dist, center=(0, 0, 0))
    step_wireframe.translate(-step_center)
    step_wireframe.scale(1.0 / max_dist, center=(0, 0, 0))

    print("Rendered:")
    print(" - LEFT  : STEP B-Rep (primitive color + black wireframe)")
    print(" - MIDDLE: OBJ mesh (gray) + simulated PCD (red)")
    print(" - RIGHT : voxel grid (green)")

    o3d.visualization.draw_geometries(
        [orig_mesh, pcd, voxel_pcd, step_mesh, step_wireframe],
        window_name=f"Comprehensive CAD Data Check: {sample['model_id']}",
        width=1500,
        height=1000,
        mesh_show_back_face=True,
    )
