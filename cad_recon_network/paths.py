from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
THIRD_PARTY_ROOT = PACKAGE_ROOT / "third_party"

POINTNET2_REPO_ROOT = THIRD_PARTY_ROOT / "Pointnet_Pointnet2_pytorch"
POINTNET2_MODELS_ROOT = POINTNET2_REPO_ROOT / "models"

SPARSECONVNET_REPO_ROOT = THIRD_PARTY_ROOT / "SparseConvNet"
