import numpy as np
import open3d as o3d

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.GCPnts import GCPnts_QuasiUniformDeflection
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopLoc import TopLoc_Location

from .constants import COLOR_MAP, DEFAULT_COLOR


def load_step_shape(step_path):
    reader = STEPControl_Reader()
    reader.ReadFile(str(step_path))
    reader.TransferRoots()
    return reader.OneShape()


def build_step_mesh_and_wireframe(
    step_path,
    mesh_deflection=0.05,
    edge_deflection=0.01,
    color_map=None,
    default_color=None,
):
    """Build colored STEP face mesh and black edge wireframe."""
    color_map = COLOR_MAP if color_map is None else color_map
    default_color = DEFAULT_COLOR if default_color is None else default_color

    shape = load_step_shape(step_path)
    BRepMesh_IncrementalMesh(shape, mesh_deflection)

    step_mesh = o3d.geometry.TriangleMesh()
    explorer_face = TopExp_Explorer(shape, TopAbs_FACE)
    while explorer_face.More():
        face = explorer_face.Current()
        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, loc)
        if triangulation:
            adaptor = BRepAdaptor_Surface(face)
            surf_type = adaptor.GetType()
            face_color = color_map.get(surf_type, default_color)

            verts = []
            for i in range(1, triangulation.NbNodes() + 1):
                p = triangulation.Node(i).Transformed(loc.Transformation())
                verts.append([p.X(), p.Y(), p.Z()])

            tris = []
            is_reversed = face.Orientation() == TopAbs_REVERSED
            for i in range(1, triangulation.NbTriangles() + 1):
                t = triangulation.Triangle(i)
                n1, n2, n3 = t.Get()
                if is_reversed:
                    tris.append([n1 - 1, n3 - 1, n2 - 1])
                else:
                    tris.append([n1 - 1, n2 - 1, n3 - 1])

            face_mesh = o3d.geometry.TriangleMesh()
            face_mesh.vertices = o3d.utility.Vector3dVector(np.array(verts))
            face_mesh.triangles = o3d.utility.Vector3iVector(np.array(tris))
            face_mesh.paint_uniform_color(face_color)
            step_mesh += face_mesh
        explorer_face.Next()

    edge_points, edge_lines = [], []
    point_idx = 0
    explorer_edge = TopExp_Explorer(shape, TopAbs_EDGE)
    while explorer_edge.More():
        edge = explorer_edge.Current()
        adaptor = BRepAdaptor_Curve(edge)
        discretizer = GCPnts_QuasiUniformDeflection(adaptor, edge_deflection)
        if discretizer.IsDone():
            pts = [
                [discretizer.Value(i).X(), discretizer.Value(i).Y(), discretizer.Value(i).Z()]
                for i in range(1, discretizer.NbPoints() + 1)
            ]
            if len(pts) > 1:
                edge_points.extend(pts)
                for i in range(len(pts) - 1):
                    edge_lines.append([point_idx + i, point_idx + i + 1])
                point_idx += len(pts)
        explorer_edge.Next()

    wireframe = o3d.geometry.LineSet()
    if edge_points:
        wireframe.points = o3d.utility.Vector3dVector(np.array(edge_points))
    else:
        wireframe.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
    if edge_lines:
        wireframe.lines = o3d.utility.Vector2iVector(np.array(edge_lines))
    else:
        wireframe.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
    wireframe.paint_uniform_color([0, 0, 0])
    return step_mesh, wireframe


def normalize_geometry(geometry, center, max_dist):
    geometry.translate(-center)
    geometry.scale(1.0 / max_dist, center=(0, 0, 0))
    return geometry
