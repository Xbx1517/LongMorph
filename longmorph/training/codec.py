"""Train a dual-stream cycle codec with reconstruction and derivative objectives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from longmorph.models.codec import DualStreamCodec, split_components
from .objectives import diagnostics, diff1, diff2, event_targets
from .splits import seed_all, subject_disjoint_split


def reconstruction_loss(model, x, weights):
    """Normalized raw/coarse/detail, first-derivative and event-curvature losses."""
    y, coarse, detail, _, _ = model(x)
    target_coarse, target_detail = split_components(x)
    raw = F.smooth_l1_loss(y, x)
    coarse_loss = F.smooth_l1_loss(coarse, target_coarse)
    detail_scale = target_detail.flatten(1).std(1).view(-1, 1, 1).detach().clamp_min(1e-4)
    detail_loss = F.smooth_l1_loss(detail / detail_scale, target_detail / detail_scale)
    td1, pd1 = diff1(x), diff1(y)
    d1_scale = td1.flatten(1).std(1).view(-1, 1, 1).detach().clamp_min(1e-4)
    d1_loss = F.smooth_l1_loss(pd1 / d1_scale, td1 / d1_scale)
    _, mask = event_targets(x)
    td2, pd2 = diff2(x), diff2(y)
    d2_scale = td2.flatten(1).std(1).view(-1, 1, 1).detach().clamp_min(1e-5)
    event = (mask * F.smooth_l1_loss(pd2 / d2_scale, td2 / d2_scale, reduction="none")).sum() / (
        mask.sum() + 1e-6
    )
    terms = dict(raw=raw, coarse=coarse_loss, detail=detail_loss, d1=d1_loss, event_d2=event)
    total = sum(getattr(weights, name + "_weight") * value for name, value in terms.items())
    return total, {"total": total.item(), **{name: value.item() for name, value in terms.items()}}


def split_rows(rows, groups_file, validation_fraction, seed):
    if groups_file:
        return subject_disjoint_split(
            rows, Path(groups_file).read_text().splitlines(), validation_fraction, seed
        )
    order = np.random.default_rng(seed).permutation(len(rows))
    split = min(len(rows) - 1, max(1, int((1 - validation_fraction) * len(rows))))
    return (
        rows[order[:split]].reshape(-1, rows.shape[-1]),
        rows[order[split:]].reshape(-1, rows.shape[-1]),
        order[:split].astype(str),
        order[split:].astype(str),
    )


def train(args):
    seed_all(args.seed)
    with np.load(args.cycles_npz, allow_pickle=False) as archive:
        rows = np.asarray(archive[args.cycles_key], np.float32)
    if (
        rows.ndim not in (2, 3)
        or rows.shape[-1] != 256
        or len(rows) < 2
        or not np.isfinite(rows).all()
    ):
        raise ValueError("Provide at least two finite owners with cycles of length 256")
    train_rows, validation, train_groups, valid_groups = split_rows(
        rows, args.groups_file, args.validation_fraction, args.seed
    )
    device = torch.device(args.device)
    model = DualStreamCodec(args.coarse_channels, args.detail_channels, args.detail_fullres).to(
        device
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_rows[:, None])),
        batch_size=args.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.lr * 0.1
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    for name in ("models", "metrics", "data"):
        (out / name).mkdir()
    best_loss = float("inf")
    history = []
    for epoch in range(args.epochs):
        model.train()
        sums = {}
        for (batch,) in loader:
            loss, terms = reconstruction_loss(model, batch.to(device), args)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite codec objective")
            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite codec gradient")
            optimizer.step()
            for name, value in terms.items():
                sums[name] = sums.get(name, 0.0) + value
        row = dict(epoch=epoch + 1, **{name: value / len(loader) for name, value in sums.items()})
        if args.checkpoint_selection == "validation":
            model.eval()
            valid_total = 0.0
            with torch.no_grad():
                for start in range(0, len(validation), args.batch_size):
                    batch = torch.from_numpy(validation[start : start + args.batch_size, None]).to(
                        device
                    )
                    loss, _ = reconstruction_loss(model, batch, args)
                    valid_total += float(loss) * len(batch)
            row["validation_total"] = valid_total / len(validation)
        selection_loss = row[
            "total" if args.checkpoint_selection == "train" else "validation_total"
        ]
        if selection_loss < best_loss:
            best_loss = selection_loss
            torch.save(
                dict(
                    model=model.state_dict(),
                    args=vars(args),
                    best_epoch=epoch + 1,
                    best_loss=best_loss,
                    checkpoint_selection=args.checkpoint_selection,
                ),
                out / "models/dualstream_codec.pt",
            )
        history.append(row)
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print(json.dumps(row), flush=True)
    checkpoint = torch.load(
        out / "models/dualstream_codec.pt", map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()
    recon, coarse, detail = [], [], []
    with torch.no_grad():
        for start in range(0, len(validation), args.batch_size):
            y, c, d, _, _ = model(
                torch.from_numpy(validation[start : start + args.batch_size, None]).to(device)
            )
            recon.append(y[:, 0].cpu().numpy())
            coarse.append(c[:, 0].cpu().numpy())
            detail.append(d[:, 0].cpu().numpy())
    recon, coarse, detail = map(np.concatenate, (recon, coarse, detail))
    metrics = diagnostics(validation, recon, args.seed)
    metrics.update(
        train_cycles=len(train_rows),
        validation_cycles=len(validation),
        split_mode="subject_disjoint" if args.groups_file else "sequence_random",
        training_groups=train_groups.tolist(),
        validation_groups=valid_groups.tolist(),
        best_epoch=checkpoint["best_epoch"],
        best_selection_loss=best_loss,
    )
    (out / "metrics/metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out / "metrics/history.json").write_text(json.dumps(history, indent=2) + "\n")
    np.savez_compressed(
        out / "data/reconstructions.npz",
        real=validation,
        reconstruction=recon,
        coarse=coarse,
        detail=detail,
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles-npz", required=True)
    parser.add_argument("--cycles-key", default="x")
    parser.add_argument("--groups-file", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--coarse-channels", type=int, default=4)
    parser.add_argument("--detail-channels", type=int, default=4)
    parser.add_argument("--detail-fullres", action="store_true")
    parser.add_argument("--raw-weight", type=float, default=1.0)
    parser.add_argument("--coarse-weight", type=float, default=1.0)
    parser.add_argument("--detail-weight", type=float, default=0.5)
    parser.add_argument("--d1-weight", type=float, default=0.3)
    parser.add_argument("--event-d2-weight", type=float, default=0.1)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--checkpoint-selection", choices=("train", "validation"), default="train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.lr <= 0
        or not 0 < args.validation_fraction < 1
    ):
        parser.error(
            "epochs/batch-size/lr must be positive and validation-fraction must be in (0,1)"
        )
    if (
        min(
            args.raw_weight,
            args.coarse_weight,
            args.detail_weight,
            args.d1_weight,
            args.event_d2_weight,
        )
        < 0
    ):
        parser.error("objective weights must be nonnegative")
    print(json.dumps(train(args), indent=2))


if __name__ == "__main__":
    main()
