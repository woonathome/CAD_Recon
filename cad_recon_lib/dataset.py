import os
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.BRepTools import breptools
from OCC.Core.GeomAbs import (
    GeomAbs_BSplineCurve,
    GeomAbs_BSplineSurface,
    GeomAbs_Circle,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Ellipse,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_Torus,
)
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_FORWARD, TopAbs_REVERSED, TopAbs_VERTEX
from OCC.Core.TopExp import TopExp_Explorer, topexp
from OCC.Core.TopTools import TopTools_IndexedMapOfShape
import OCC.Core.gp as gp


class ABCMultiModalDataset(Dataset):
    """ABC dataset loader that returns PCD, voxel, and B-rep feature tensors."""

    def __init__(
        self,
        base_dir,
        pcd_num_points=2048,
        voxel_res=64,
        max_vertices=4000,
        max_edges=2000,
        max_faces=1000,
        max_view_retry=8,
    ):
        self.base_dir = Path(base_dir)
        self.pcd_num_points = pcd_num_points
        self.voxel_res = voxel_res
        self.max_vertices = max_vertices
        self.max_edges = max_edges
        self.max_faces = max_faces
        self.max_view_retry = int(max_view_retry)
        self.device = o3d.core.Device("CPU:0")

        self.obj_files = []
        print("OBJ file scanning...")
        for chunk_folder in sorted(os.listdir(self.base_dir)):
            chunk_path = self.base_dir / chunk_folder
            if chunk_path.is_dir():
                for model_id in os.listdir(chunk_path):
                    model_path = chunk_path / model_id
                    objs = list(model_path.glob("*.obj"))
                    if objs:
                        self.obj_files.append(objs[0])
        print(f"{len(self.obj_files)} OBJ files loaded.")

    def __len__(self):
        return len(self.obj_files)

    def _extract_complex_brep_target(self, step_path):
        """Extract B-rep targets based on ISO-10303-42 conventions."""
        reader = STEPControl_Reader()
        if reader.ReadFile(str(step_path)) != 1:
            return None

        reader.TransferRoots()
        shape = reader.OneShape()

        # 1) Build unique topology maps (V/E/F)
        map_V = TopTools_IndexedMapOfShape()
        map_E = TopTools_IndexedMapOfShape()
        map_F = TopTools_IndexedMapOfShape()

        topexp.MapShapes(shape, TopAbs_VERTEX, map_V)
        topexp.MapShapes(shape, TopAbs_EDGE, map_E)
        topexp.MapShapes(shape, TopAbs_FACE, map_F)

        num_V, num_E, num_F = map_V.Size(), map_E.Size(), map_F.Size()

        # 2) Initialize feature tensors
        V_feat = np.zeros((self.max_vertices, 3))
        E_feat = np.zeros((self.max_edges, 73))
        F_feat = np.zeros((self.max_faces, 174))

        # 3) Initialize topology adjacency matrices
        adj_EV = np.zeros((self.max_edges, self.max_vertices))
        adj_FE = np.zeros((self.max_faces, self.max_edges))

        # 4) Extract vertex geometry
        for i in range(1, num_V + 1):
            if i > self.max_vertices:
                break
            v_shape = map_V.FindKey(i)
            pnt = BRep_Tool.Pnt(v_shape)
            V_feat[i - 1] = [pnt.X(), pnt.Y(), pnt.Z()]

        # 5) Extract edge geometry, trimming, and E-V adjacency
        for i in range(1, num_E + 1):
            if i > self.max_edges:
                break
            e_shape = map_E.FindKey(i)

            v_first, v_last = topexp.FirstVertex(e_shape), topexp.LastVertex(e_shape)
            if not v_first.IsNull():
                idx_v1 = map_V.FindIndex(v_first)
                if idx_v1 <= self.max_vertices:
                    adj_EV[i - 1, idx_v1 - 1] = 1.0
            if not v_last.IsNull():
                idx_v2 = map_V.FindIndex(v_last)
                if idx_v2 <= self.max_vertices:
                    adj_EV[i - 1, idx_v2 - 1] = -1.0

            try:
                adaptor = BRepAdaptor_Curve(e_shape)
                tmin, tmax = adaptor.FirstParameter(), adaptor.LastParameter()
                ctype = adaptor.GetType()

                p_start, p_end = adaptor.Value(tmin), adaptor.Value(tmax)
                ori = 0.0 if e_shape.Orientation() == TopAbs_FORWARD.value else 1.0

                E_feat[i - 1, 0:9] = [
                    float(ctype),
                    tmin,
                    tmax,
                    p_start.X(),
                    p_start.Y(),
                    p_start.Z(),
                    p_end.X(),
                    p_end.Y(),
                    p_end.Z(),
                ]
                E_feat[i - 1, 11] = ori

                if ctype == GeomAbs_Circle:
                    E_feat[i - 1, 9] = adaptor.Circle().Radius()
                elif ctype == GeomAbs_Ellipse:
                    E_feat[i - 1, 9] = adaptor.Ellipse().MajorRadius()
                    E_feat[i - 1, 10] = adaptor.Ellipse().MinorRadius()
                elif ctype == GeomAbs_BSplineCurve:
                    b_curve = adaptor.BSpline()
                    E_feat[i - 1, 9] = float(b_curve.Degree())
                    nb_poles = b_curve.NbPoles()
                    E_feat[i - 1, 12] = float(nb_poles)
                    p_indices = np.linspace(1, nb_poles, min(10, nb_poles))
                    pole_idx = 13
                    for p_i in p_indices:
                        pole = b_curve.Pole(int(round(p_i)))
                        E_feat[i - 1, pole_idx : pole_idx + 3] = [pole.X(), pole.Y(), pole.Z()]
                        pole_idx += 3

                pts_idx = 43
                for t in np.linspace(tmin, tmax, 10):
                    pt = adaptor.Value(t)
                    E_feat[i - 1, pts_idx : pts_idx + 3] = [pt.X(), pt.Y(), pt.Z()]
                    pts_idx += 3

            except Exception as e:
                print(e)

        # 6) Extract face geometry, trimming, and F-E adjacency
        for i in range(1, num_F + 1):
            if i > self.max_faces:
                break
            f_shape = map_F.FindKey(i)

            exp_e = TopExp_Explorer(f_shape, TopAbs_EDGE)
            while exp_e.More():
                e_in_f = exp_e.Current()
                idx_e = map_E.FindIndex(e_in_f)
                if 0 < idx_e <= self.max_edges:
                    e_ori = 1.0 if e_in_f.Orientation() == TopAbs_FORWARD.value else -1.0
                    adj_FE[i - 1, idx_e - 1] = e_ori
                exp_e.Next()

            try:
                adaptor = BRepAdaptor_Surface(f_shape)
                umin, umax, vmin, vmax = breptools.UVBounds(f_shape)
                stype = adaptor.GetType()

                umid, vmid = (umin + umax) / 2.0, (vmin + vmax) / 2.0
                p_mid = adaptor.Value(umid, vmid)

                loc_x, loc_y, loc_z = p_mid.X(), p_mid.Y(), p_mid.Z()
                z_dir = [0, 0, 1]
                x_dir = [1, 0, 0]

                pos = None
                if stype == GeomAbs_Plane:
                    pos = adaptor.Plane().Position()
                elif stype == GeomAbs_Cylinder:
                    pos = adaptor.Cylinder().Position()
                elif stype == GeomAbs_Cone:
                    pos = adaptor.Cone().Position()
                elif stype == GeomAbs_Sphere:
                    pos = adaptor.Sphere().Position()
                elif stype == GeomAbs_Torus:
                    pos = adaptor.Torus().Position()

                if pos is not None:
                    loc = pos.Location()
                    z_axis = pos.Direction()
                    x_axis = pos.XDirection()
                    loc_x, loc_y, loc_z = loc.X(), loc.Y(), loc.Z()
                    z_dir = [z_axis.X(), z_axis.Y(), z_axis.Z()]
                    x_dir = [x_axis.X(), x_axis.Y(), x_axis.Z()]

                P = gp.gp_Pnt()
                V1U, V1V = gp.gp_Vec(), gp.gp_Vec()
                adaptor.D1(umid, vmid, P, V1U, V1V)
                normal = V1U.Crossed(V1V)
                if normal.Magnitude() > 1e-10:
                    normal.Normalize()
                else:
                    normal = gp.gp_Vec(0, 0, 1)
                nx, ny, nz = normal.X(), normal.Y(), normal.Z()

                ori = 0.0 if f_shape.Orientation() == TopAbs_FORWARD.value else 1.0

                F_feat[i - 1, 0:8] = [
                    float(stype),
                    umin,
                    umax,
                    vmin,
                    vmax,
                    p_mid.X(),
                    p_mid.Y(),
                    p_mid.Z(),
                ]
                F_feat[i - 1, 8:11] = [nx, ny, nz]
                F_feat[i - 1, 13] = ori

                if stype == GeomAbs_Cylinder:
                    F_feat[i - 1, 11] = adaptor.Cylinder().Radius()
                elif stype == GeomAbs_Sphere:
                    F_feat[i - 1, 11] = adaptor.Sphere().Radius()
                elif stype == GeomAbs_Cone:
                    F_feat[i - 1, 11] = adaptor.Cone().RefRadius()
                    F_feat[i - 1, 12] = adaptor.Cone().SemiAngle()
                elif stype == GeomAbs_Torus:
                    F_feat[i - 1, 11] = adaptor.Torus().MajorRadius()
                    F_feat[i - 1, 12] = adaptor.Torus().MinorRadius()
                elif stype == GeomAbs_BSplineSurface:
                    bspl = adaptor.BSpline()
                    F_feat[i - 1, 11] = float(bspl.UDegree())
                    F_feat[i - 1, 12] = float(bspl.VDegree())
                    nb_u, nb_v = bspl.NbUPoles(), bspl.NbVPoles()
                    F_feat[i - 1, 14] = float(nb_u * nb_v)

                    u_indices = np.linspace(1, nb_u, min(5, nb_u))
                    v_indices = np.linspace(1, nb_v, min(5, nb_v))
                    pole_idx = 15
                    for u in u_indices:
                        for v in v_indices:
                            pole = bspl.Pole(int(round(u)), int(round(v)))
                            F_feat[i - 1, pole_idx : pole_idx + 3] = [pole.X(), pole.Y(), pole.Z()]
                            pole_idx += 3

                F_feat[i - 1, 90:93] = [loc_x, loc_y, loc_z]
                F_feat[i - 1, 93:96] = z_dir
                F_feat[i - 1, 96:99] = x_dir

                grid_idx = 99
                for u in np.linspace(umin, umax, 5):
                    for v in np.linspace(vmin, vmax, 5):
                        pt = adaptor.Value(u, v)
                        F_feat[i - 1, grid_idx : grid_idx + 3] = [pt.X(), pt.Y(), pt.Z()]
                        grid_idx += 3

            except Exception as e:
                print(e)

        return {
            "v_feat": torch.tensor(V_feat).float(),
            "e_feat": torch.tensor(E_feat).float(),
            "f_feat": torch.tensor(F_feat).float(),
            "adj_ev": torch.tensor(adj_EV).float(),
            "adj_fe": torch.tensor(adj_FE).float(),
            "counts": torch.tensor([num_V, num_E, num_F]).int(),
        }

    def _simulate_stereo_depth(self, mesh, max_mesh_dist):
        dist_min = max_mesh_dist * 2.5
        dist_max = max_mesh_dist * 4.0

        phi, theta = np.random.uniform(0, 2 * np.pi), np.arccos(np.random.uniform(-1, 1))
        radius = np.random.uniform(dist_min, dist_max)

        cam_pos = np.array(
            [
                radius * np.sin(theta) * np.cos(phi),
                radius * np.sin(theta) * np.sin(phi),
                radius * np.cos(theta),
            ]
        )

        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh, device=self.device))

        rays = scene.create_rays_pinhole(
            fov_deg=69,
            center=[0, 0, 0],
            eye=cam_pos,
            up=[0, 1, 0],
            width_px=320,
            height_px=240,
        )
        ans = scene.cast_rays(rays)
        t_hit = ans["t_hit"].numpy()
        hit_mask, rays_arr = np.isfinite(t_hit), rays.numpy()
        pts = rays_arr[hit_mask][:, :3] + t_hit[hit_mask][:, np.newaxis] * rays_arr[hit_mask][:, 3:]

        z_depths = np.dot(pts - cam_pos, -cam_pos / np.linalg.norm(cam_pos))
        noise = np.random.normal(0, 1, size=pts.shape) * ((z_depths**2 * 0.08) / (800 * 50))[:, np.newaxis]
        return pts + noise

    def __getitem__(self, idx):
        obj_path = self.obj_files[idx]
        step_path = obj_path.with_suffix(".step")

        mesh = o3d.io.read_triangle_mesh(str(obj_path))
        mesh.translate(-mesh.get_center())
        max_mesh_dist = np.max(np.linalg.norm(np.asarray(mesh.vertices), axis=1))
        if not np.isfinite(max_mesh_dist) or max_mesh_dist < 1e-8:
            max_mesh_dist = 1.0

        raw_sim_points = np.empty((0, 3), dtype=np.float32)
        for _ in range(max(1, self.max_view_retry)):
            raw_sim_points = self._simulate_stereo_depth(mesh, max_mesh_dist)
            if raw_sim_points.shape[0] > 0:
                break

        # Fallback path for rare camera-miss cases.
        if raw_sim_points.shape[0] == 0:
            sampled = mesh.sample_points_uniformly(number_of_points=max(self.pcd_num_points, 2048))
            raw_sim_points = np.asarray(sampled.points, dtype=np.float32)

        if raw_sim_points.shape[0] == 0:
            raw_sim_points = np.zeros((self.pcd_num_points, 3), dtype=np.float32)

        pcd_pts = raw_sim_points / (max_mesh_dist + 1e-8)

        if len(pcd_pts) >= self.pcd_num_points:
            pcd_pts = pcd_pts[np.random.choice(len(pcd_pts), self.pcd_num_points, replace=False)]
        else:
            pcd_pts = pcd_pts[np.random.choice(len(pcd_pts), self.pcd_num_points, replace=True)]

        voxel_pts = (pcd_pts * 0.5) + 0.5
        voxel_indices = np.clip((voxel_pts * (self.voxel_res - 1)).astype(int), 0, self.voxel_res - 1)
        voxel_matrix = np.zeros((self.voxel_res, self.voxel_res, self.voxel_res), dtype=np.float32)
        voxel_matrix[voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]] = 1.0

        brep_target = self._extract_complex_brep_target(step_path)
        if brep_target is None:
            brep_target = {
                "v_feat": torch.zeros((self.max_vertices, 3)),
                "e_feat": torch.zeros((self.max_edges, 73)),
                "f_feat": torch.zeros((self.max_faces, 174)),
                "adj_ev": torch.zeros((self.max_edges, self.max_vertices)),
                "adj_fe": torch.zeros((self.max_faces, self.max_edges)),
                "counts": torch.zeros(3).int(),
            }

        return {
            "pcd": torch.from_numpy(pcd_pts).float(),
            "voxel": torch.from_numpy(voxel_matrix).float().unsqueeze(0),
            **brep_target,
            "model_id": obj_path.parent.name,
            "max_mesh_dist": max_mesh_dist,
            "obj_path": obj_path,
            "step_path": step_path,
        }
