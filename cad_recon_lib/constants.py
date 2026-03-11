from OCC.Core.GeomAbs import (
    GeomAbs_Plane,
    GeomAbs_Cylinder,
    GeomAbs_Cone,
    GeomAbs_Sphere,
    GeomAbs_Torus,
    GeomAbs_BezierSurface,
    GeomAbs_BSplineSurface,
    GeomAbs_SurfaceOfRevolution,
    GeomAbs_SurfaceOfExtrusion,
    GeomAbs_OffsetSurface,
    GeomAbs_OtherSurface,
)


COLOR_MAP = {
    GeomAbs_Plane: [0.8, 0.8, 0.8],
    GeomAbs_Cylinder: [0.2, 0.6, 1.0],
    GeomAbs_Cone: [1.0, 0.8, 0.0],
    GeomAbs_Sphere: [0.2, 0.8, 0.2],
    GeomAbs_Torus: [1.0, 0.5, 0.0],
    GeomAbs_BSplineSurface: [1.0, 0.2, 0.2],
    GeomAbs_SurfaceOfRevolution: [0.6, 0.2, 0.8],
    GeomAbs_SurfaceOfExtrusion: [0.4, 0.8, 0.8],
    GeomAbs_BezierSurface: [1.0, 0.4, 0.7],
    GeomAbs_OffsetSurface: [0.5, 0.5, 0.0],
    GeomAbs_OtherSurface: [0.3, 0.3, 0.3],
}

DEFAULT_COLOR = [1.0, 1.0, 1.0]
