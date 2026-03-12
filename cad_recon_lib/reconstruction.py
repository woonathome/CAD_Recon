import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, snap, unary_union

import OCC.Core.gp as gp
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.Geom import (
    Geom_ConicalSurface,
    Geom_CylindricalSurface,
    Geom_Plane,
    Geom_SphericalSurface,
    Geom_ToroidalSurface,
)
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCC.Core.GeomAbs import (
    GeomAbs_BezierSurface,
    GeomAbs_BSplineSurface,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_SurfaceOfExtrusion,
    GeomAbs_SurfaceOfRevolution,
    GeomAbs_Torus,
)
from OCC.Core.gp import gp_Pnt
from OCC.Core.TopAbs import TopAbs_IN, TopAbs_OUT, TopAbs_SOLID
from OCC.Core.TopExp import TopExp_Explorer

from .constants import COLOR_MAP, DEFAULT_COLOR
from .occ_visualization import build_step_mesh_and_wireframe, load_step_shape


@dataclass
class ReconstructionOptions:
    mesh_deflection: float = 0.05
    edge_deflection: float = 0.01
    offset_x: float = 2.5

    fast_vis_mode: bool = True
    fast_normal_vertex_limit: int = 5000
    fast_probe_target: int = 96
    fast_base_steps: int = 72
    fast_grid_min: int = 48
    fast_grid_max: int = 140
    use_closed_solid_prior: bool = True
    solid_probe_samples: int = 32
    solid_probe_offset_ratio: float = 1e-3
    solid_probe_boundary_ratio: float = 3.0

    window_width: int = 1600
    window_height: int = 900
    background_color: tuple = (1.0, 1.0, 1.0)
    show_wireframe: bool = False
    show_back_face: bool = True
    window_name: str = "Tensor Verification: Original (Left) vs Extracted (Right)"


def _to_numpy(arr):
    if hasattr(arr, "detach"):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _normalize_pts(pts, orig_center, max_dist):
    return (pts - orig_center) / max_dist


def _state_is_inside(state):
    return state == TopAbs_IN


def _state_is_outside(state):
    return state == TopAbs_OUT


def build_reconstruction_geometries(
    sample,
    options: Optional[ReconstructionOptions] = None,
    color_map=None,
    default_color=None,
):
    """
    Build geometry objects for:
    - original STEP mesh + wireframe
    - reconstructed mesh + wireframe from extracted B-rep tensors
    """
    options = ReconstructionOptions() if options is None else options
    color_map = COLOR_MAP if color_map is None else color_map
    default_color = DEFAULT_COLOR if default_color is None else default_color

    step_path = str(sample["step_path"])
    max_dist = float(sample["max_mesh_dist"])

    v_feat = _to_numpy(sample["v_feat"])
    e_feat = _to_numpy(sample["e_feat"])
    f_feat = _to_numpy(sample["f_feat"])
    adj_FE = _to_numpy(sample["adj_fe"])
    adj_EV = _to_numpy(sample["adj_ev"])

    num_E = min(int(sample["counts"][1]), e_feat.shape[0])
    num_F = min(int(sample["counts"][2]), f_feat.shape[0])

    # 1) Original STEP mesh + wireframe
    orig_mesh, orig_wireframe = build_step_mesh_and_wireframe(
        step_path,
        mesh_deflection=options.mesh_deflection,
        edge_deflection=options.edge_deflection,
        color_map=color_map,
        default_color=default_color,
    )
    orig_center = orig_mesh.get_center()
    orig_mesh.translate(-orig_center)
    orig_mesh.scale(1.0 / max_dist, center=(0, 0, 0))
    orig_wireframe.translate(-orig_center)
    orig_wireframe.scale(1.0 / max_dist, center=(0, 0, 0))

    # Optional 3D solid prior:
    # Dataset is filtered to single watertight body, so use it to disambiguate
    # periodic-surface hole fill/complement candidates.
    solid_classifier = None
    if options.use_closed_solid_prior:
        try:
            solid_shape_root = load_step_shape(step_path)
            solid_exp = TopExp_Explorer(solid_shape_root, TopAbs_SOLID)
            if solid_exp.More():
                solid_shape = solid_exp.Current()
                solid_classifier = BRepClass3d_SolidClassifier(solid_shape)
        except Exception:
            solid_classifier = None

    # 2) Reconstructed edge wireframe
    offset = np.array([options.offset_x, 0.0, 0.0], dtype=np.float64)
    recon_e_pts, recon_e_lines = [], []
    for i in range(num_E):
        pts_10 = e_feat[i, 43:73].reshape(10, 3)
        if np.all(pts_10 == 0):
            continue

        start_idx = len(recon_e_pts)
        recon_e_pts.extend(pts_10)
        for j in range(9):
            recon_e_lines.append([start_idx + j, start_idx + j + 1])

    recon_wireframe = o3d.geometry.LineSet()
    if recon_e_pts:
        recon_wireframe.points = o3d.utility.Vector3dVector(
            _normalize_pts(np.array(recon_e_pts), orig_center, max_dist)
        )
        recon_wireframe.lines = o3d.utility.Vector2iVector(np.array(recon_e_lines))
        recon_wireframe.paint_uniform_color([0, 0, 0])
        recon_wireframe.translate(offset)

    # 3) Reconstructed face mesh
    recon_mesh = o3d.geometry.TriangleMesh()
    for i in range(num_F):
        umin, umax, vmin, vmax = f_feat[i, 1:5]
        if abs(umax - umin) < 1e-5 or abs(vmax - vmin) < 1e-5 or abs(umax) > 1e6:
            continue

        stype = int(f_feat[i, 0])
        p1, p2 = float(f_feat[i, 11]), float(f_feat[i, 12])

        matched_enum = None
        for k in color_map.keys():
            if int(k) == stype:
                matched_enum = k
                break

        face_color = [0.0, 0.0, 0.0]
        for k, v in color_map.items():
            if int(k) == stype:
                face_color = v
                break

        is_reversed = float(f_feat[i, 13]) == 1.0

        # Freeform surfaces
        if matched_enum in [
            GeomAbs_BezierSurface,
            GeomAbs_BSplineSurface,
            GeomAbs_SurfaceOfRevolution,
            GeomAbs_SurfaceOfExtrusion,
        ]:
            grid_pts = f_feat[i, 99:174].reshape(25, 3)
            if not np.all(grid_pts == 0):
                f_tris = []
                for r in range(4):
                    for c in range(4):
                        idx1 = r * 5 + c
                        idx2 = idx1 + 5
                        idx3 = idx1 + 1
                        idx4 = idx2 + 1
                        if is_reversed:
                            f_tris.extend([[idx1, idx2, idx3], [idx2, idx4, idx3]])
                        else:
                            f_tris.extend([[idx1, idx3, idx2], [idx2, idx3, idx4]])

                tmp_mesh = o3d.geometry.TriangleMesh()
                tmp_mesh.vertices = o3d.utility.Vector3dVector(
                    _normalize_pts(grid_pts, orig_center, max_dist)
                )
                tmp_mesh.triangles = o3d.utility.Vector3iVector(np.array(f_tris))
                tmp_mesh.paint_uniform_color(face_color)
                tmp_mesh.translate(offset)
                if (not options.fast_vis_mode) or (len(grid_pts) <= options.fast_normal_vertex_limit):
                    tmp_mesh.compute_vertex_normals()
                recon_mesh += tmp_mesh
            continue

        # Primitive analytic surfaces
        loc = gp.gp_Pnt(float(f_feat[i, 90]), float(f_feat[i, 91]), float(f_feat[i, 92]))
        z_vec = gp.gp_Vec(float(f_feat[i, 93]), float(f_feat[i, 94]), float(f_feat[i, 95]))
        x_vec = gp.gp_Vec(float(f_feat[i, 96]), float(f_feat[i, 97]), float(f_feat[i, 98]))

        if z_vec.Magnitude() > 1e-5 and x_vec.Magnitude() > 1e-5:
            z_dir = gp.gp_Dir(z_vec)
            y_vec = z_vec.Crossed(x_vec)
            if y_vec.Magnitude() > 1e-5:
                ax3 = gp.gp_Ax3(loc, z_dir, gp.gp_Dir(y_vec.Crossed(z_vec)))
            else:
                ax3 = gp.gp_Ax3(loc, z_dir)
        else:
            ax3 = gp.gp_Ax3(loc, gp.gp_Dir(0, 0, 1))

        geom_surf = None
        if matched_enum == GeomAbs_Plane:
            geom_surf = Geom_Plane(ax3)
        elif matched_enum == GeomAbs_Cylinder and p1 >= 0:
            geom_surf = Geom_CylindricalSurface(ax3, p1)
        elif matched_enum == GeomAbs_Cone and p1 >= 0:
            geom_surf = Geom_ConicalSurface(ax3, p2, p1)
        elif matched_enum == GeomAbs_Sphere and p1 >= 0:
            geom_surf = Geom_SphericalSurface(ax3, p1)
        elif matched_enum == GeomAbs_Torus and p1 >= 0:
            geom_surf = Geom_ToroidalSurface(ax3, p1, p2)

        if geom_surf is None:
            continue

        connected_edges = np.where(adj_FE[i] != 0)[0]

        periodic_u = matched_enum in [GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere, GeomAbs_Torus]
        period_u = 2.0 * math.pi
        raw_u_span = max(abs(umax - umin), 1e-6)
        u_eval_span = period_u if periodic_u and raw_u_span > period_u * 0.9 else raw_u_span

        def unwrap_u(u_val, ref_u):
            if not periodic_u:
                return float(u_val)
            out = float(u_val)
            ref = float(ref_u)
            while out - ref > math.pi:
                out -= period_u
            while ref - out > math.pi:
                out += period_u
            return out

        def wrap_u_eval(u_val):
            if not periodic_u:
                return float(u_val)
            out = float(u_val)
            span = max(u_eval_span, 1e-8)
            while out < umin:
                out += span
            while out > umax:
                out -= span
            return out

        # 3-1) Vertex anchors in UV
        anchor_ref_u = float((umin + umax) * 0.5)
        vertex_uv_map = {}
        for e_idx in connected_edges:
            v_indices = np.where(adj_EV[e_idx] != 0)[0]
            for v_idx in v_indices:
                if v_idx in vertex_uv_map:
                    continue
                v_pt = v_feat[v_idx, 0:3]
                if np.all(v_pt == 0):
                    continue
                proj = GeomAPI_ProjectPointOnSurf(
                    gp_Pnt(float(v_pt[0]), float(v_pt[1]), float(v_pt[2])),
                    geom_surf,
                )
                if proj.NbPoints() > 0:
                    u, v = proj.LowerDistanceParameters()
                    u = unwrap_u(u, anchor_ref_u)
                    vertex_uv_map[v_idx] = (float(u), float(v))

        if vertex_uv_map:
            anchor_ref_u = float(np.median([uv[0] for uv in vertex_uv_map.values()]))

        # 3-2) Edge samples in UV
        edge_lines = []
        edge_records = []
        for e_idx in connected_edges:
            pts_10 = e_feat[e_idx, 43:73].reshape(10, 3)
            pts_valid = [pt for pt in pts_10 if not np.all(pt == 0)]
            if len(pts_valid) < 2:
                continue
            if adj_FE[i, e_idx] < 0.0:
                pts_valid = pts_valid[::-1]

            v_pos = np.where(adj_EV[e_idx] > 0)[0]
            v_neg = np.where(adj_EV[e_idx] < 0)[0]
            start_v = int(v_pos[0]) if len(v_pos) > 0 else None
            end_v = int(v_neg[0]) if len(v_neg) > 0 else None
            if start_v is None and end_v is not None and len(v_neg) == 1 and len(v_pos) == 0:
                start_v = end_v
            elif end_v is None and start_v is not None and len(v_pos) == 1 and len(v_neg) == 0:
                end_v = start_v
            if start_v is not None and end_v is not None and adj_FE[i, e_idx] < 0.0:
                start_v, end_v = end_v, start_v

            uv_line = []
            prev_u = None
            for p3 in pts_valid:
                proj = GeomAPI_ProjectPointOnSurf(
                    gp_Pnt(float(p3[0]), float(p3[1]), float(p3[2])),
                    geom_surf,
                )
                if proj.NbPoints() <= 0:
                    continue
                u, v = proj.LowerDistanceParameters()
                if prev_u is not None:
                    u = unwrap_u(u, prev_u)
                elif start_v is not None and start_v in vertex_uv_map:
                    u = unwrap_u(u, vertex_uv_map[start_v][0])
                else:
                    u = unwrap_u(u, anchor_ref_u)
                uv_line.append([float(u), float(v)])
                prev_u = float(u)

            if len(uv_line) < 2:
                continue
            if start_v is not None and start_v in vertex_uv_map:
                uv_line[0] = [float(vertex_uv_map[start_v][0]), float(vertex_uv_map[start_v][1])]
            if end_v is not None and end_v in vertex_uv_map:
                uv_line[-1] = [float(vertex_uv_map[end_v][0]), float(vertex_uv_map[end_v][1])]
            if start_v is not None and end_v is not None and start_v == end_v:
                if math.hypot(uv_line[0][0] - uv_line[-1][0], uv_line[0][1] - uv_line[-1][1]) > 1e-8:
                    uv_line.append(uv_line[0][:])

            cleaned = [uv_line[0]]
            for uv in uv_line[1:]:
                if math.hypot(uv[0] - cleaned[-1][0], uv[1] - cleaned[-1][1]) > 1e-8:
                    cleaned.append(uv)
            if len(cleaned) < 2:
                continue

            edge_lines.append(cleaned)
            edge_records.append({"start_v": start_v, "end_v": end_v, "uv": cleaned})

        if periodic_u and edge_lines:
            all_u = [pt[0] for line in edge_lines for pt in line]
            work_u_center = float(np.median(all_u)) if all_u else float((umin + umax) * 0.5)
            for line in edge_lines:
                for pt in line:
                    pt[0] = unwrap_u(pt[0], work_u_center)
            for rec in edge_records:
                for pt in rec["uv"]:
                    pt[0] = unwrap_u(pt[0], work_u_center)
            for v_idx in list(vertex_uv_map.keys()):
                u_v, v_v = vertex_uv_map[v_idx]
                vertex_uv_map[v_idx] = (unwrap_u(u_v, work_u_center), float(v_v))
            work_umin = work_u_center - 0.5 * u_eval_span
            work_umax = work_u_center + 0.5 * u_eval_span
        else:
            work_umin, work_umax = float(umin), float(umax)
            work_u_center = float((work_umin + work_umax) * 0.5)

        bbox_poly = Polygon([(work_umin, vmin), (work_umax, vmin), (work_umax, vmax), (work_umin, vmax)])
        valid_polygon = bbox_poly

        uv_span = max(abs(work_umax - work_umin), abs(vmax - vmin), 1e-6)
        chain_tol = max(uv_span * 2e-3, 5e-5)
        snap_tol = max(uv_span * 5e-4, 1e-6)
        area_tol = max(bbox_poly.area * 1e-8, 1e-12)

        probe_eps = max(chain_tol * 1.5, uv_span * 3e-3)
        probe_pairs = []
        for line in edge_lines:
            seg_count = max(0, len(line) - 1)
            seg_stride = max(1, seg_count // 12) if options.fast_vis_mode else 1
            for idx_seg in range(0, seg_count, seg_stride):
                p0 = line[idx_seg]
                p1 = line[idx_seg + 1]
                dx = float(p1[0] - p0[0])
                dy = float(p1[1] - p0[1])
                seg_len = math.hypot(dx, dy)
                if seg_len <= 1e-12:
                    continue
                nx = -dy / seg_len
                ny = dx / seg_len
                if is_reversed:
                    nx, ny = -nx, -ny
                mx = 0.5 * (float(p0[0]) + float(p1[0]))
                my = 0.5 * (float(p0[1]) + float(p1[1]))
                probe_in = Point(mx + probe_eps * nx, my + probe_eps * ny)
                probe_out = Point(mx - probe_eps * nx, my - probe_eps * ny)
                probe_pairs.append((probe_in, probe_out))
        if options.fast_vis_mode and len(probe_pairs) > options.fast_probe_target:
            pick_stride = max(1, len(probe_pairs) // options.fast_probe_target)
            probe_pairs = probe_pairs[::pick_stride]

        support_points = []
        for line in edge_lines:
            support_stride = max(1, len(line) // 3) if options.fast_vis_mode else max(1, len(line) // 8)
            for uv in line[::support_stride]:
                support_points.append(Point(float(uv[0]), float(uv[1])))

        seed_point = None
        seed_xyz = f_feat[i, 5:8]
        if not np.all(seed_xyz == 0):
            proj_seed = GeomAPI_ProjectPointOnSurf(
                gp_Pnt(float(seed_xyz[0]), float(seed_xyz[1]), float(seed_xyz[2])),
                geom_surf,
            )
            if proj_seed.NbPoints() > 0:
                su, sv = proj_seed.LowerDistanceParameters()
                su = unwrap_u(su, work_u_center)
                seed_point = Point(float(su), float(sv))

        # Solid prior score for single watertight body datasets:
        # points on the true shell should have one side inside and one side outside.
        def solid_prior_score(cand_clip):
            if (
                (not options.use_closed_solid_prior)
                or (solid_classifier is None)
                or (not periodic_u)
                or cand_clip.is_empty
            ):
                return 0.0

            min_u, min_v, max_u, max_v = cand_clip.bounds
            du = max(abs(max_u - min_u) * 1e-4, 1e-5)
            dv = max(abs(max_v - min_v) * 1e-4, 1e-5)
            offset_len = max(max_dist * options.solid_probe_offset_ratio, 1e-5)
            boundary_guard = chain_tol * options.solid_probe_boundary_ratio

            target = max(8, int(options.solid_probe_samples))
            grid_side = max(3, int(math.sqrt(target)) + 1)
            u_grid = np.linspace(min_u, max_u, grid_side)
            v_grid = np.linspace(min_v, max_v, grid_side)

            good = 0.0
            bad = 0.0
            count = 0

            for u in u_grid:
                for v in v_grid:
                    if count >= target:
                        break
                    p_uv = Point(float(u), float(v))
                    if not cand_clip.covers(p_uv):
                        continue
                    if cand_clip.boundary.distance(p_uv) <= boundary_guard:
                        continue

                    u_eval = wrap_u_eval(u)
                    v_eval = float(v)
                    p = geom_surf.Value(float(u_eval), v_eval)

                    u_eval_du = wrap_u_eval(u + du)
                    v_eval_dv = v_eval + dv if (v_eval + dv) <= vmax else v_eval - dv
                    v_eval_dv = min(float(vmax), max(float(vmin), float(v_eval_dv)))
                    p_u = geom_surf.Value(float(u_eval_du), v_eval)
                    p_v = geom_surf.Value(float(u_eval), float(v_eval_dv))

                    t_u = np.array([p_u.X() - p.X(), p_u.Y() - p.Y(), p_u.Z() - p.Z()], dtype=np.float64)
                    t_v = np.array([p_v.X() - p.X(), p_v.Y() - p.Y(), p_v.Z() - p.Z()], dtype=np.float64)
                    n = np.cross(t_u, t_v)
                    n_norm = np.linalg.norm(n)
                    if n_norm <= 1e-12:
                        continue
                    n /= n_norm
                    if is_reversed:
                        n = -n

                    p_plus = gp_Pnt(
                        float(p.X() + n[0] * offset_len),
                        float(p.Y() + n[1] * offset_len),
                        float(p.Z() + n[2] * offset_len),
                    )
                    p_minus = gp_Pnt(
                        float(p.X() - n[0] * offset_len),
                        float(p.Y() - n[1] * offset_len),
                        float(p.Z() - n[2] * offset_len),
                    )

                    solid_classifier.Perform(p_plus, 1e-6)
                    state_plus = solid_classifier.State()
                    solid_classifier.Perform(p_minus, 1e-6)
                    state_minus = solid_classifier.State()

                    plus_in = _state_is_inside(state_plus)
                    minus_in = _state_is_inside(state_minus)
                    plus_out = _state_is_outside(state_plus)
                    minus_out = _state_is_outside(state_minus)

                    if (plus_in and minus_out) or (minus_in and plus_out):
                        good += 1.0
                    elif plus_out and minus_out:
                        bad += 1.0
                    else:
                        bad += 0.2
                    count += 1
                if count >= target:
                    break

            if count == 0:
                return 0.0
            return (good - 0.5 * bad) / count

        if edge_lines:
            try:
                loops_uv = []
                if edge_records:
                    unused = set(range(len(edge_records)))
                    max_steps = len(edge_records) + 4
                    for idx in list(unused):
                        rec = edge_records[idx]
                        if rec["start_v"] is not None and rec["end_v"] is not None and rec["start_v"] == rec["end_v"]:
                            if len(rec["uv"]) >= 4:
                                loops_uv.append([p[:] for p in rec["uv"]])
                            unused.remove(idx)

                    while unused:
                        seed_idx = unused.pop()
                        seed = edge_records[seed_idx]
                        if seed["start_v"] is None or seed["end_v"] is None:
                            continue

                        loop_pts = [p[:] for p in seed["uv"]]
                        start_v = seed["start_v"]
                        cur_v = seed["end_v"]
                        steps = 0
                        closed = False

                        while steps < max_steps:
                            if cur_v == start_v:
                                closed = True
                                break
                            fwd = [idx for idx in unused if edge_records[idx]["start_v"] == cur_v]
                            rev = [idx for idx in unused if edge_records[idx]["end_v"] == cur_v]
                            if not fwd and not rev:
                                break
                            reverse_pick = False
                            if fwd:
                                next_idx = min(
                                    fwd,
                                    key=lambda idx: math.hypot(
                                        loop_pts[-1][0] - edge_records[idx]["uv"][0][0],
                                        loop_pts[-1][1] - edge_records[idx]["uv"][0][1],
                                    ),
                                )
                            else:
                                next_idx = min(
                                    rev,
                                    key=lambda idx: math.hypot(
                                        loop_pts[-1][0] - edge_records[idx]["uv"][-1][0],
                                        loop_pts[-1][1] - edge_records[idx]["uv"][-1][1],
                                    ),
                                )
                                reverse_pick = True

                            unused.remove(next_idx)
                            seg_rec = edge_records[next_idx]
                            seg_uv = [p[:] for p in seg_rec["uv"]]
                            seg_end = seg_rec["end_v"]
                            if reverse_pick:
                                seg_uv = seg_uv[::-1]
                                seg_end = seg_rec["start_v"]
                            if math.hypot(loop_pts[-1][0] - seg_uv[0][0], loop_pts[-1][1] - seg_uv[0][1]) > chain_tol:
                                loop_pts.append(seg_uv[0][:])
                            loop_pts.extend([p[:] for p in seg_uv[1:]])
                            cur_v = seg_end
                            steps += 1

                        if len(loop_pts) >= 3:
                            if math.hypot(loop_pts[0][0] - loop_pts[-1][0], loop_pts[0][1] - loop_pts[-1][1]) <= chain_tol:
                                loop_pts[-1] = loop_pts[0][:]
                                closed = True
                            elif closed:
                                loop_pts.append(loop_pts[0][:])
                            if closed and len(loop_pts) >= 4:
                                loops_uv.append(loop_pts)

                loop_polys = []
                for loop in loops_uv:
                    poly = Polygon(loop).buffer(0)
                    if poly.is_empty:
                        continue
                    if poly.geom_type == "Polygon":
                        if poly.area > area_tol:
                            loop_polys.append(poly)
                    else:
                        for g in getattr(poly, "geoms", []):
                            if g.geom_type == "Polygon" and g.area > area_tol:
                                loop_polys.append(g)

                if not loop_polys:
                    line_geoms = [LineString(line) for line in edge_lines if len(line) >= 2]
                    merged = unary_union(line_geoms)
                    merged = snap(merged, merged, snap_tol)
                    for p in polygonize(merged):
                        q = p.buffer(0)
                        if not q.is_empty and q.area > area_tol:
                            loop_polys.append(q)

                candidates = []
                if loop_polys:
                    parity_shape = None
                    for poly in loop_polys:
                        clipped = poly.intersection(bbox_poly).buffer(0)
                        if clipped.is_empty or clipped.area <= area_tol:
                            continue
                        parity_shape = (
                            clipped
                            if parity_shape is None
                            else parity_shape.symmetric_difference(clipped).buffer(0)
                        )
                    if parity_shape is not None and not parity_shape.is_empty and parity_shape.area > area_tol:
                        candidates.append(parity_shape)
                        if periodic_u:
                            comp = bbox_poly.difference(parity_shape).buffer(0)
                            if not comp.is_empty and comp.area > area_tol:
                                candidates.append(comp)

                    if periodic_u:
                        # For periodic cylindrical-like faces, interior loops are usually holes.
                        # Build an explicit outer-shell candidate as bbox minus interior loops.
                        interior_holes = []
                        boundary_touch_tol = max(chain_tol * 1.5, uv_span * 1e-3)
                        for poly in loop_polys:
                            clipped = poly.intersection(bbox_poly).buffer(0)
                            if clipped.is_empty or clipped.area <= area_tol:
                                continue
                            if clipped.area >= bbox_poly.area * 0.98:
                                continue
                            if clipped.boundary.distance(bbox_poly.boundary) > boundary_touch_tol:
                                interior_holes.append(clipped)
                        if interior_holes:
                            shell_candidate = bbox_poly.difference(unary_union(interior_holes)).buffer(0)
                            if not shell_candidate.is_empty and shell_candidate.area > area_tol:
                                candidates.append(shell_candidate)
                if not candidates:
                    candidates = [bbox_poly]

                def score_candidate(cand):
                    if cand is None or cand.is_empty:
                        return -1e18
                    cand_clip = cand.intersection(bbox_poly).buffer(0)
                    if cand_clip.is_empty or cand_clip.area <= area_tol:
                        return -1e18
                    score = 0.0

                    edge_hits = 0
                    edge_total = 0
                    for p_uv in support_points:
                        if cand_clip.boundary.distance(p_uv) <= chain_tol * 1.5:
                            edge_hits += 1
                        edge_total += 1
                    if edge_total > 0:
                        score += 2.0 * (edge_hits / edge_total)

                    if probe_pairs:
                        probe_ok = 0.0
                        for pin, pout in probe_pairs:
                            in_hit = cand_clip.covers(pin)
                            out_hit = not cand_clip.covers(pout)
                            if in_hit and out_hit:
                                probe_ok += 1.0
                            elif in_hit or out_hit:
                                probe_ok += 0.5
                        score += 6.0 * (probe_ok / max(len(probe_pairs), 1))

                    if periodic_u and seed_point is not None:
                        seed_dist = cand_clip.boundary.distance(seed_point)
                        if seed_dist > chain_tol:
                            score += 2.5 if cand_clip.covers(seed_point) else -2.5

                    # Strong disambiguation for curved periodic faces with holes.
                    score += 8.0 * solid_prior_score(cand_clip)

                    fill_ratio = cand_clip.area / max(bbox_poly.area, 1e-12)
                    if fill_ratio < 0.01 or fill_ratio > 0.99:
                        score -= 0.5
                    if periodic_u:
                        # Bias away from degenerate tiny strips on periodic side faces.
                        score += 1.2 * fill_ratio
                    return score

                scored = sorted(((score_candidate(c), c) for c in candidates), key=lambda x: x[0], reverse=True)
                best_poly = scored[0][1] if scored else bbox_poly

                if periodic_u and len(scored) > 1:
                    best_score, best_geom = scored[0]
                    best_clip = best_geom.intersection(bbox_poly).buffer(0)
                    best_area = best_clip.area if not best_clip.is_empty else 0.0
                    best_fill = best_area / max(bbox_poly.area, 1e-12)

                    alt_score, alt_geom = max(
                        scored,
                        key=lambda x: (x[1].intersection(bbox_poly).buffer(0).area if not x[1].is_empty else 0.0),
                    )
                    alt_clip = alt_geom.intersection(bbox_poly).buffer(0)
                    alt_area = alt_clip.area if not alt_clip.is_empty else 0.0

                    if (
                        best_fill < 0.35
                        and alt_area > best_area * 1.5
                        and alt_score >= (best_score - 1.25)
                    ):
                        best_poly = alt_geom

                best_poly = best_poly.intersection(bbox_poly).buffer(0)
                if best_poly.is_empty or best_poly.area <= area_tol:
                    best_poly = bbox_poly
                valid_polygon = best_poly
            except Exception as e:
                print(f"Topology Error: {e}")
                valid_polygon = bbox_poly

        u_span = max(abs(work_umax - work_umin), 1e-6)
        v_span = max(abs(vmax - vmin), 1e-6)
        aspect = max(u_span / v_span, 1e-6)
        if options.fast_vis_mode:
            base_steps = options.fast_base_steps
            u_steps = int(np.clip(base_steps * math.sqrt(aspect), options.fast_grid_min, options.fast_grid_max))
            v_steps = int(np.clip(base_steps / math.sqrt(aspect), options.fast_grid_min, options.fast_grid_max))
        else:
            base_steps = 120
            u_steps = int(np.clip(base_steps * math.sqrt(aspect), 80, 220))
            v_steps = int(np.clip(base_steps / math.sqrt(aspect), 80, 220))

        mask_tol = max(max(u_span, v_span) * 1e-4, 1e-6)
        u_vals = np.linspace(work_umin, work_umax, u_steps)
        v_vals = np.linspace(vmin, vmax, v_steps)

        f_verts, f_tris = [], []
        uv_idx_map = {}
        next_idx = 0
        for ru, u in enumerate(u_vals):
            for rv, v in enumerate(v_vals):
                uv_point = Point(float(u), float(v))
                if not (valid_polygon.covers(uv_point) or valid_polygon.boundary.distance(uv_point) <= mask_tol):
                    continue
                u_eval = wrap_u_eval(u)
                p3 = geom_surf.Value(float(u_eval), float(v))
                f_verts.append([p3.X(), p3.Y(), p3.Z()])
                uv_idx_map[(ru, rv)] = next_idx
                next_idx += 1

        for ru in range(u_steps - 1):
            for rv in range(v_steps - 1):
                k00 = (ru, rv)
                k10 = (ru + 1, rv)
                k01 = (ru, rv + 1)
                k11 = (ru + 1, rv + 1)
                if not (k00 in uv_idx_map and k10 in uv_idx_map and k01 in uv_idx_map and k11 in uv_idx_map):
                    continue
                i00 = uv_idx_map[k00]
                i10 = uv_idx_map[k10]
                i01 = uv_idx_map[k01]
                i11 = uv_idx_map[k11]
                if is_reversed:
                    f_tris.extend([[i00, i10, i01], [i10, i11, i01]])
                else:
                    f_tris.extend([[i00, i01, i10], [i10, i01, i11]])

        if len(f_verts) >= 3 and len(f_tris) > 0:
            tmp_mesh = o3d.geometry.TriangleMesh()
            tmp_mesh.vertices = o3d.utility.Vector3dVector(
                _normalize_pts(np.array(f_verts), orig_center, max_dist)
            )
            tmp_mesh.triangles = o3d.utility.Vector3iVector(np.array(f_tris))
            tmp_mesh.paint_uniform_color(face_color)
            tmp_mesh.translate(offset)
            if (not options.fast_vis_mode) or (len(f_verts) <= options.fast_normal_vertex_limit):
                tmp_mesh.compute_vertex_normals()
            recon_mesh += tmp_mesh

    return {
        "orig_mesh": orig_mesh,
        "orig_wireframe": orig_wireframe,
        "recon_mesh": recon_mesh,
        "recon_wireframe": recon_wireframe,
    }


def visualize_brep_reconstruction_comparison(
    sample,
    options: Optional[ReconstructionOptions] = None,
    color_map=None,
    default_color=None,
    enforce_closed_solid_prior: bool = True,
):
    options = ReconstructionOptions() if options is None else options
    options.use_closed_solid_prior = enforce_closed_solid_prior
    geom = build_reconstruction_geometries(
        sample=sample,
        options=options,
        color_map=color_map,
        default_color=default_color,
    )

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=options.window_name,
        width=options.window_width,
        height=options.window_height,
    )
    opt = vis.get_render_option()
    opt.background_color = np.asarray(options.background_color)
    opt.mesh_show_wireframe = options.show_wireframe
    opt.mesh_show_back_face = options.show_back_face

    vis.add_geometry(geom["orig_mesh"])
    vis.add_geometry(geom["orig_wireframe"])
    vis.add_geometry(geom["recon_mesh"])
    vis.add_geometry(geom["recon_wireframe"])
    vis.run()
    vis.destroy_window()
    return geom
