param(
    [string]$PythonExe = "python",
    [string]$TorchCudaArchList = ""
)

$ErrorActionPreference = "Stop"
$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$networkRoot = Resolve-Path (Join-Path $scriptRoot "..")
$sparseRoot = Join-Path $networkRoot "third_party\SparseConvNet"
$pointnetRoot = Join-Path $networkRoot "third_party\Pointnet_Pointnet2_pytorch"

if (-not (Test-Path $pointnetRoot)) {
    throw "PointNet++ repo not found: $pointnetRoot"
}
if (-not (Test-Path $sparseRoot)) {
    throw "SparseConvNet repo not found: $sparseRoot"
}

Write-Host "Using python: $PythonExe"
& $PythonExe -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'is_available', torch.cuda.is_available())"

$env:FORCE_CUDA = "1"
$env:SCN_USE_CUDA = "1"
if ($TorchCudaArchList -ne "") {
    $env:TORCH_CUDA_ARCH_LIST = $TorchCudaArchList
    Write-Host "TORCH_CUDA_ARCH_LIST=$TorchCudaArchList"
}

Write-Host ""
Write-Host "[1/2] PointNet++"
Write-Host "PointNet++ from yanx27 repo is pure PyTorch and runs on GPU when tensors/model are on CUDA."

Write-Host ""
Write-Host "[2/2] SparseConvNet (CUDA extension build)"
Push-Location $sparseRoot
try {
    & $PythonExe setup.py build_ext --inplace
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "Done. Run smoke test:"
Write-Host "  $PythonExe cad_recon_network/scripts/smoke_test_backbones.py --device cuda"
