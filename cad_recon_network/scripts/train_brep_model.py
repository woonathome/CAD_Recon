from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cad_recon_lib.dataset import ABCMultiModalDataset
from cad_recon_network import CADReconBRepModel

_TENSOR_KEYS = ("pcd", "voxel", "v_feat", "e_feat", "f_feat", "adj_ev", "adj_fe", "counts", "max_mesh_dist")
_LOSS_TERMS = (
    "total",
    "v_feat",
    "e_feat",
    "f_feat",
    "e_type",
    "f_type",
    "e_ori",
    "f_ori",
    "adj_ev",
    "adj_fe",
    "counts",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CADReconBRepModel end-to-end.")
    parser.add_argument("--base-dir", type=str, required=True, help="Root directory passed to ABCMultiModalDataset.")
    parser.add_argument("--output-dir", type=str, default="cad_recon_network/weights/brep_runs/exp1")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint (.pt) to resume from.")

    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fail-on-nonfinite",
        action="store_true",
        help="Raise an error immediately when non-finite loss is detected.",
    )

    parser.add_argument("--amp", dest="amp", action="store_true", help="Enable mixed precision training.")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable mixed precision training.")
    parser.set_defaults(amp=True)

    parser.add_argument(
        "--freeze-pointnet-bn",
        dest="freeze_pointnet_bn",
        action="store_true",
        help="Keep PointNet BatchNorm layers in eval mode during training (recommended for batch_size=1).",
    )
    parser.add_argument(
        "--train-pointnet-bn",
        dest="freeze_pointnet_bn",
        action="store_false",
        help="Train PointNet BatchNorm layers.",
    )
    parser.set_defaults(freeze_pointnet_bn=True)

    parser.add_argument("--pcd-num-points", type=int, default=2048)
    parser.add_argument("--voxel-res", type=int, default=64)
    parser.add_argument("--max-vertices", type=int, default=4000)
    parser.add_argument("--max-edges", type=int, default=2000)
    parser.add_argument("--max-faces", type=int, default=1000)
    parser.add_argument("--max-view-retry", type=int, default=8)
    parser.add_argument(
        "--regression-target-clip",
        type=float,
        default=200.0,
        help="Absolute clip value applied to v/e/f regression targets before SmoothL1.",
    )

    parser.add_argument("--pointnet-variant", type=str, default="ssg", choices=["ssg", "msg"])
    parser.add_argument("--pointnet-feature-dim", type=int, default=1024)
    parser.add_argument("--sparse-feature-dim", type=int, default=256)
    parser.add_argument("--head-hidden-dim", type=int, default=256)
    parser.add_argument("--head-layers", type=int, default=2)
    parser.add_argument("--head-latents", type=int, default=64)
    parser.add_argument("--head-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-curve-types", type=int, default=9)
    parser.add_argument("--num-surface-types", type=int, default=11)

    parser.add_argument("--w-v-feat", type=float, default=1.0)
    parser.add_argument("--w-e-feat", type=float, default=1.0)
    parser.add_argument("--w-f-feat", type=float, default=1.0)
    parser.add_argument("--w-e-type", type=float, default=0.5)
    parser.add_argument("--w-f-type", type=float, default=0.5)
    parser.add_argument("--w-e-ori", type=float, default=0.25)
    parser.add_argument("--w-f-ori", type=float, default=0.25)
    parser.add_argument("--w-adj-ev", type=float, default=0.5)
    parser.add_argument("--w-adj-fe", type=float, default=0.5)
    parser.add_argument("--w-counts", type=float, default=0.2)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cad_collate_fn(items: list[dict[str, Any]]) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    for key in ("pcd", "voxel", "v_feat", "e_feat", "f_feat", "adj_ev", "adj_fe", "counts"):
        batch[key] = torch.stack([item[key] for item in items], dim=0)

    batch["max_mesh_dist"] = torch.tensor([float(item["max_mesh_dist"]) for item in items], dtype=torch.float32)
    batch["model_id"] = [item["model_id"] for item in items]
    batch["obj_path"] = [str(item["obj_path"]) for item in items]
    batch["step_path"] = [str(item["step_path"]) for item in items]
    return batch


def create_loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader | None]:
    dataset = ABCMultiModalDataset(
        base_dir=args.base_dir,
        pcd_num_points=args.pcd_num_points,
        voxel_res=args.voxel_res,
        max_vertices=args.max_vertices,
        max_edges=args.max_edges,
        max_faces=args.max_faces,
        max_view_retry=args.max_view_retry,
    )

    val_loader: DataLoader | None = None
    if args.val_ratio > 0.0 and len(dataset) > 1:
        val_size = max(1, int(round(len(dataset) * args.val_ratio)))
        train_size = len(dataset) - val_size
        if train_size <= 0:
            train_size = len(dataset) - 1
            val_size = 1

        generator = torch.Generator().manual_seed(args.seed)
        train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)
    else:
        train_ds = dataset
        val_ds = None

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=cad_collate_fn,
        persistent_workers=args.num_workers > 0,
    )
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=cad_collate_fn,
            persistent_workers=args.num_workers > 0,
        )
    return train_loader, val_loader


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if key in _TENSOR_KEYS and isinstance(value, torch.Tensor):
            moved[key] = value.to(device=device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def freeze_pointnet_batchnorm(model: CADReconBRepModel) -> None:
    for module in model.backbone.pointnet.model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def to_float_dict(loss_dict: dict[str, torch.Tensor]) -> dict[str, float]:
    return {k: float(v.detach().cpu()) for k, v in loss_dict.items()}


def mean_loss(loss_accum: list[dict[str, float]]) -> dict[str, float]:
    if not loss_accum:
        return {}
    keys = loss_accum[0].keys()
    return {k: float(np.mean([entry[k] for entry in loss_accum])) for k in keys}


def write_epoch_csv_row(
    csv_path: Path,
    *,
    epoch: int,
    elapsed_sec: float,
    best_val_loss: float,
    train_loss: dict[str, float],
    val_loss: dict[str, float] | None,
) -> None:
    fieldnames = ["epoch", "elapsed_sec", "best_val_loss"]
    fieldnames.extend([f"train_{k}" for k in _LOSS_TERMS])
    fieldnames.extend([f"val_{k}" for k in _LOSS_TERMS])

    row: dict[str, Any] = {
        "epoch": epoch,
        "elapsed_sec": elapsed_sec,
        "best_val_loss": best_val_loss,
    }
    for k in _LOSS_TERMS:
        row[f"train_{k}"] = train_loss.get(k, float("nan"))
        row[f"val_{k}"] = (val_loss.get(k, float("nan")) if val_loss is not None else "")

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train_one_epoch(
    model: CADReconBRepModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    use_amp: bool,
    loss_weights: dict[str, float],
    grad_clip: float,
    log_every: int,
    freeze_pointnet_bn: bool,
    fail_on_nonfinite: bool,
) -> dict[str, float]:
    model.train()
    if freeze_pointnet_bn:
        freeze_pointnet_batchnorm(model)

    losses: list[dict[str, float]] = []
    skipped_nonfinite = 0
    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
            _, loss_dict = model.forward_with_loss(
                batch,
                normalize_targets=True,
                loss_weights=loss_weights,
            )
            total_loss = loss_dict["total"]

        if not torch.isfinite(total_loss):
            skipped_nonfinite += 1
            model_id = batch.get("model_id", ["<unknown>"])[0]
            msg = (
                f"[train][warn] non-finite loss at step {step}, model_id={model_id}, "
                f"loss={float(total_loss.detach().cpu())}. Batch skipped."
            )
            if fail_on_nonfinite:
                raise RuntimeError(msg)
            print(msg)
            optimizer.zero_grad(set_to_none=True)
            continue

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        current = to_float_dict(loss_dict)
        losses.append(current)
        if step % log_every == 0:
            print(
                f"[train] step {step:05d}/{len(loader):05d} "
                f"total={current['total']:.6f} v={current['v_feat']:.6f} e={current['e_feat']:.6f} "
                f"f={current['f_feat']:.6f} e_type={current['e_type']:.6f} f_type={current['f_type']:.6f} "
                f"e_ori={current['e_ori']:.6f} f_ori={current['f_ori']:.6f} "
                f"adj_ev={current['adj_ev']:.6f} adj_fe={current['adj_fe']:.6f} counts={current['counts']:.6f}"
            )
    if skipped_nonfinite > 0:
        print(f"[train] skipped non-finite batches: {skipped_nonfinite}")
    return mean_loss(losses)


@torch.no_grad()
def evaluate(
    model: CADReconBRepModel,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    loss_weights: dict[str, float],
    fail_on_nonfinite: bool,
) -> dict[str, float]:
    model.eval()
    losses: list[dict[str, float]] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
            _, loss_dict = model.forward_with_loss(
                batch,
                normalize_targets=True,
                loss_weights=loss_weights,
            )
        total = loss_dict["total"]
        if not torch.isfinite(total):
            model_id = batch.get("model_id", ["<unknown>"])[0]
            msg = (
                f"[val][warn] non-finite loss, model_id={model_id}, "
                f"loss={float(total.detach().cpu())}. Batch ignored."
            )
            if fail_on_nonfinite:
                raise RuntimeError(msg)
            print(msg)
            continue
        losses.append(to_float_dict(loss_dict))
    return mean_loss(losses)


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: CADReconBRepModel,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler | None,
    best_val_loss: float,
    args: argparse.Namespace,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val_loss,
        "args": vars(args),
    }
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use --device cpu or activate CUDA-enabled environment.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    train_loader, val_loader = create_loaders(args)
    print(f"train batches: {len(train_loader)}")
    if val_loader is not None:
        print(f"val batches:   {len(val_loader)}")
    else:
        print("val batches:   0 (validation disabled)")

    model = CADReconBRepModel(
        pointnet_kwargs={
            "variant": args.pointnet_variant,
            "use_normals": False,
            "feature_dim": args.pointnet_feature_dim,
        },
        sparseconv_kwargs={
            "voxel_resolution": args.voxel_res,
            "in_channels": 1,
            "feature_dim": args.sparse_feature_dim,
            "allow_fallback": False,
        },
        head_kwargs={
            "num_curve_types": args.num_curve_types,
            "num_surface_types": args.num_surface_types,
            "max_vertices": args.max_vertices,
            "max_edges": args.max_edges,
            "max_faces": args.max_faces,
            "hidden_dim": args.head_hidden_dim,
            "num_decoder_layers": args.head_layers,
            "num_latents": args.head_latents,
            "num_heads": args.head_heads,
            "dropout": args.dropout,
        },
        regression_target_clip=args.regression_target_clip,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler: torch.amp.GradScaler | None = torch.amp.GradScaler("cuda", enabled=True) if use_amp else None

    loss_weights = {
        "v_feat": args.w_v_feat,
        "e_feat": args.w_e_feat,
        "f_feat": args.w_f_feat,
        "e_type": args.w_e_type,
        "f_type": args.w_f_type,
        "e_ori": args.w_e_ori,
        "f_ori": args.w_f_ori,
        "adj_ev": args.w_adj_ev,
        "adj_fe": args.w_adj_fe,
        "counts": args.w_counts,
    }

    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if scaler is not None and ckpt.get("scaler_state") is not None:
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        print(f"Resumed from {ckpt_path} at epoch {start_epoch}.")

    metrics_path = output_dir / "metrics.jsonl"
    metrics_csv_path = output_dir / "epoch_losses.csv"
    for epoch in range(start_epoch, args.epochs + 1):
        tic = time.time()
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            use_amp=use_amp,
            loss_weights=loss_weights,
            grad_clip=args.grad_clip,
            log_every=args.log_every,
            freeze_pointnet_bn=args.freeze_pointnet_bn,
            fail_on_nonfinite=args.fail_on_nonfinite,
        )

        val_loss: dict[str, float] | None = None
        if val_loader is not None:
            val_loss = evaluate(
                model=model,
                loader=val_loader,
                device=device,
                use_amp=use_amp,
                loss_weights=loss_weights,
                fail_on_nonfinite=args.fail_on_nonfinite,
            )

        elapsed = time.time() - tic
        train_total = train_loss.get("total", float("nan"))
        msg = f"[epoch {epoch:03d}] train_total={train_total:.6f}"
        if val_loss is not None:
            val_total = val_loss.get("total", float("nan"))
            msg += f" val_total={val_total:.6f}"
            if val_total < best_val_loss:
                best_val_loss = val_total
                save_checkpoint(
                    output_dir / "best.pt",
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    best_val_loss=best_val_loss,
                    args=args,
                )
                msg += " (best)"
        msg += f" time={elapsed:.1f}s"
        print(msg)

        if epoch % args.save_every == 0:
            save_checkpoint(
                output_dir / "last.pt",
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                best_val_loss=best_val_loss,
                args=args,
            )

        record = {
            "epoch": epoch,
            "train": train_loss,
            "val": val_loss,
            "best_val_loss": best_val_loss,
            "elapsed_sec": elapsed,
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        write_epoch_csv_row(
            metrics_csv_path,
            epoch=epoch,
            elapsed_sec=elapsed,
            best_val_loss=best_val_loss,
            train_loss=train_loss,
            val_loss=val_loss,
        )

    print(f"Training finished. Metrics: {metrics_path} / {metrics_csv_path}")


if __name__ == "__main__":
    main()
