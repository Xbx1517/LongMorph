"""Train coarse and coarse-conditioned detail block flows with velocity coupling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from longmorph.data.cycles import rhythm_features, shared_cycles
from longmorph.models.codec import load_generator_codec
from longmorph.models.flow import BlockVelocity, CycleCouplingPredictor
from .splits import seed_all


def normalize_pairs(prefix, target):
    n = len(prefix)
    joined = np.concatenate([prefix.reshape(n, -1), target.reshape(n, -1)])
    mean = joined.mean(0).astype(np.float32)
    scale = joined.std(0).astype(np.float32) + 1e-4
    return (
        ((prefix.reshape(n, -1) - mean) / scale).reshape(prefix.shape).astype(np.float32),
        ((target.reshape(n, -1) - mean) / scale).reshape(target.shape).astype(np.float32),
        mean,
        scale,
    )


def gaussian_source(target, rank, rng):
    """Low-rank Gaussian fitted to complete normalized three-cycle blocks."""
    flat = target.reshape(len(target), -1)
    mean = flat.mean(0, keepdims=True)
    centered = flat - mean
    rank = max(1, min(rank, len(flat) - 1, flat.shape[1]))
    _, singular_values, vectors = torch.pca_lowrank(
        torch.from_numpy(centered), q=rank, center=False, niter=4
    )
    basis = vectors.numpy().astype(np.float32)
    scale = (singular_values.numpy() / np.sqrt(max(len(flat) - 1, 1))).astype(np.float32)
    sample = (
        mean
        + rng.normal(size=(len(target), len(scale))).astype(np.float32) @ (basis * scale[None]).T
    )
    return sample.reshape(target.shape).astype(np.float32), mean[0].astype(np.float32), basis, scale


def block_dataset(cycles, conditions, durations, codec, batch_size, device, sample_rate=500.0):
    if (
        cycles.shape[1:] != (24, 256)
        or conditions.shape != (len(cycles), 24, 14)
        or durations.shape != (len(cycles), 24)
    ):
        raise ValueError(
            "Flow training requires 24 cycles per sequence, 14 conditions and positive durations"
        )
    if not all(np.isfinite(x).all() for x in (cycles, conditions, durations)) or np.any(
        durations <= 0
    ):
        raise ValueError("Training arrays must be finite with positive durations")
    coarse, detail = [], []
    flat = cycles.reshape(-1, 256)
    with torch.no_grad():
        for start in range(0, len(flat), batch_size):
            _, _, _, c, d = codec(
                torch.from_numpy(flat[start : start + batch_size, None]).to(device)
            )
            coarse.append(c.cpu().numpy())
            detail.append(d.cpu().numpy())
    coarse = np.concatenate(coarse).reshape(len(cycles), 24, 4, 128)
    detail = np.concatenate(detail).reshape(len(cycles), 24, 2, 256)
    pc, pd, tc, td, rows = [], [], [], [], []
    for owner in range(len(cycles)):
        rhythm = rhythm_features(durations[owner], sample_rate)
        for start in range(0, 19, 3):
            pc.append(coarse[owner, start : start + 3])
            pd.append(detail[owner, start : start + 3])
            tc.append(coarse[owner, start + 3 : start + 6])
            td.append(detail[owner, start + 3 : start + 6])
            rows.append(np.r_[conditions[owner, start + 3 : start + 6].reshape(-1), rhythm])
    return tuple(np.asarray(x, np.float32) for x in (pc, pd, tc, td, rows))


def flow_objective(
    coarse_flow,
    detail_flow,
    coupling_predictor,
    batch,
    coupling_weight=0.1,
    detail_weight=1.0,
    times=None,
):
    source_c, source_d, target_c, target_d, prefix_c, prefix_d, conditions = batch
    t = torch.rand(len(source_c), device=source_c.device) if times is None else times
    state_c = source_c + t[:, None, None] * (target_c - source_c)
    state_d = source_d + t[:, None, None] * (target_d - source_d)
    velocity_c = coarse_flow(state_c, prefix_c, conditions, t)
    coarse_context = F.interpolate(target_c, size=256, mode="linear", align_corners=False)
    velocity_d = detail_flow(state_d, prefix_d, conditions, t, coarse_context)
    coarse_loss = F.mse_loss(velocity_c, target_c - source_c)
    detail_loss = F.mse_loss(velocity_d, target_d - source_d)
    coupling = velocity_c.new_zeros(())
    if coupling_predictor is not None:
        velocity_cycles = velocity_c.reshape(len(velocity_c), 3, 4, 128)
        state_cycles = state_c.reshape(len(state_c), 3, 4, 128)
        pair_state = state_cycles[:, :-1].reshape(-1, 4, 128)
        pair_velocity = velocity_cycles[:, 1:].reshape(-1, 4, 128)
        pair_t = t[:, None].expand(-1, 2).reshape(-1)
        coupling = F.mse_loss(coupling_predictor(pair_state, pair_t), pair_velocity)
    total = coarse_loss + coupling_weight * coupling + detail_weight * detail_loss
    return total, dict(
        total=float(total.detach()),
        coarse_cfm=float(coarse_loss.detach()),
        detail_cfm=float(detail_loss.detach()),
        coarse_coupling=float(coupling.detach()),
    )


def train(args):
    seed_all(args.seed)
    reference = Path(args.reference_run)
    cfg = json.loads((reference / "config/config.json").read_text())
    with np.load(
        reference / "data/train_segment_condition_sequences.npz", allow_pickle=False
    ) as sequences:
        conditions = np.asarray(sequences["conditions"], np.float32)
        durations = np.asarray(sequences["durations"], np.float32)
        if "cycles" in sequences:
            cycles = np.asarray(sequences["cycles"], np.float32)
        else:
            with np.load(reference / "data/train_ppg_segments.npz", allow_pickle=False) as data:
                waveforms = data["x"][:, :, 0]
            cycles = shared_cycles(
                waveforms, durations, cfg["seq_len"], cfg["beat_count"], cfg["cycle_len"]
            )
    device = torch.device(args.device)
    codec = load_generator_codec(args.codec, device)
    codec.eval()
    for parameter in codec.parameters():
        parameter.requires_grad_(False)
    pc, pd, tc, td, condition_rows = block_dataset(
        cycles, conditions, durations, codec, args.batch_size, device, cfg["sample_rate"]
    )
    pc, tc, coarse_mean, coarse_scale = normalize_pairs(pc, tc)
    pd, td, detail_mean, detail_scale = normalize_pairs(pd, td)
    condition_mean = condition_rows.mean(0)
    condition_scale = condition_rows.std(0) + 1e-4
    cn = ((condition_rows - condition_mean) / condition_scale).astype(np.float32)
    rng = np.random.default_rng(args.seed)
    sc, coarse_source_mean, coarse_source_basis, coarse_source_scale = gaussian_source(
        tc, args.source_pca_rank, rng
    )
    sd, detail_source_mean, detail_source_basis, detail_source_scale = gaussian_source(
        td, args.source_pca_rank, rng
    )
    stats = dict(
        coarse_mean=coarse_mean,
        coarse_scale=coarse_scale,
        detail_mean=detail_mean,
        detail_scale=detail_scale,
        cond_mean=condition_mean,
        cond_scale=condition_scale,
        coarse_source_mean=coarse_source_mean,
        coarse_source_basis=coarse_source_basis,
        coarse_source_scale=coarse_source_scale,
        detail_source_mean=detail_source_mean,
        detail_source_basis=detail_source_basis,
        detail_source_scale=detail_source_scale,
    )
    n = len(pc)
    arrays = (
        sc.reshape(n, 12, 128),
        sd.reshape(n, 6, 256),
        tc.reshape(n, 12, 128),
        td.reshape(n, 6, 256),
        pc.reshape(n, 12, 128),
        pd.reshape(n, 6, 256),
        cn,
    )
    loader = DataLoader(
        TensorDataset(*(torch.from_numpy(value) for value in arrays)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    coarse_flow = BlockVelocity(12, len(condition_mean)).to(device)
    detail_flow = BlockVelocity(6, len(condition_mean), 12).to(device)
    coupling = CycleCouplingPredictor(4).to(device) if args.coupling_weight > 0 else None
    trainable = list(coarse_flow.parameters()) + list(detail_flow.parameters())
    if coupling is not None:
        trainable += list(coupling.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.lr * 0.1
    )
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    (out / "models").mkdir()
    (out / "metrics").mkdir()
    history = []
    for epoch in range(args.epochs):
        sums = {}
        for batch in loader:
            batch = tuple(value.to(device) for value in batch)
            loss, terms = flow_objective(
                coarse_flow,
                detail_flow,
                coupling,
                batch,
                args.coupling_weight,
                args.detail_flow_weight,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite block Flow objective")
            optimizer.zero_grad()
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, 2)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite block Flow gradient")
            optimizer.step()
            for name, value in terms.items():
                sums[name] = sums.get(name, 0.0) + value
        scheduler.step()
        row = dict(epoch=epoch + 1, **{name: value / len(loader) for name, value in sums.items()})
        history.append(row)
        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            print(json.dumps(row), flush=True)
    source_metadata = dict(
        schema_version=5,
        mode="pca_gaussian",
        samples=n,
        residual_pca_convention="mean + normal @ (basis * scale).T",
    )
    payload = dict(
        coarse_flow=coarse_flow.state_dict(),
        detail_flow=detail_flow.state_dict(),
        stats=stats,
        cond_dim=len(condition_mean),
        args=vars(args),
        source_mode="pca_gaussian",
        source_metadata=source_metadata,
        training_metadata=dict(
            objective="L_cCFM + lambda_coup L_coup + lambda_d L_dCFM",
            lambda_coup=args.coupling_weight,
            lambda_d=args.detail_flow_weight,
        ),
    )
    if coupling is not None:
        payload["coupling_predictor"] = coupling.state_dict()
    torch.save(payload, out / "models/dual_block_flow.pt")
    (out / "metrics/history.json").write_text(json.dumps(history, indent=2) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", required=True)
    parser.add_argument("--codec", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--source-pca-rank", type=int, default=256)
    parser.add_argument("--coupling-weight", type=float, default=0.1)
    parser.add_argument("--detail-flow-weight", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.source_pca_rank) < 1 or args.lr <= 0:
        parser.error("epochs, batch-size, PCA rank and learning rate must be positive")
    if min(args.coupling_weight, args.detail_flow_weight) < 0:
        parser.error("objective weights must be nonnegative")
    train(args)


if __name__ == "__main__":
    main()
