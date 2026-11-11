#!/usr/bin/env python3
"""Autonomous stochastic 3->3 condition transition gate for LongMorph."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from longmorph.models.condition import ConditionTransition
from torch.utils.data import DataLoader, TensorDataset


def hrv(durations, sample_rate):
    rr = durations / sample_rate * 1000.0
    mean = rr.mean(1)
    sd = rr.std(1)
    diff = np.diff(rr, axis=1)
    rmssd = np.sqrt(np.mean(diff**2, axis=1))
    pnn = (np.abs(diff) > 50).mean(1)
    hr = 60000.0 / np.maximum(mean, 1)
    return np.stack([mean, sd, rmssd, pnn, hr], 1)


def mmd(x, y):
    z = np.concatenate([x, y])
    take = z[: min(len(z), 256)]
    d = ((take[:, None] - take[None]) ** 2).sum(2)
    gamma = 1 / (np.median(d[d > 0]) + 1e-8)
    kxx = np.exp(-gamma * ((x[:, None] - x[None]) ** 2).sum(2)).mean()
    kyy = np.exp(-gamma * ((y[:, None] - y[None]) ** 2).sum(2)).mean()
    kxy = np.exp(-gamma * ((x[:, None] - y[None]) ** 2).sum(2)).mean()
    return float(kxx + kyy - 2 * kxy)


def constrain_rollout_block(
    values, lower, upper, mode="none", enforce_positive_duration=False, min_duration_samples=1.0
):
    """Project one decoded transition proposal onto empirical support.

    The model and its checkpoint remain in the original normalized coordinate
    system.  Projection happens in physical feature units and the caller maps
    the result back to normalized coordinates before the next autoregressive
    step.  ``none`` with no duration enforcement is an exact no-op.
    """
    raw = np.asarray(values, dtype=np.float32)
    projected = raw.copy()
    lower = np.asarray(lower, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)
    lower_violation = np.isfinite(raw) & (raw < lower)
    upper_violation = np.isfinite(raw) & (raw > upper)
    nonfinite = ~np.isfinite(raw)
    if mode == "clip":
        midpoint = (lower + upper) * 0.5
        projected = np.where(np.isnan(projected), midpoint, projected)
        projected = np.where(np.isneginf(projected), lower, projected)
        projected = np.where(np.isposinf(projected), upper, projected)
        projected = np.minimum(np.maximum(projected, lower), upper)
    elif mode != "none":
        raise ValueError(f"unknown rollout support mode: {mode}")
    after_support = projected.copy()
    if enforce_positive_duration:
        invalid = ~np.isfinite(projected[..., -1]) | (projected[..., -1] < min_duration_samples)
        projected[..., -1] = np.where(invalid, min_duration_samples, projected[..., -1])
    else:
        invalid = np.zeros(projected.shape[:-1], dtype=bool)
    support_changed = ~np.isclose(after_support, raw, rtol=0.0, atol=0.0, equal_nan=True)
    total_changed = ~np.isclose(projected, raw, rtol=0.0, atol=0.0, equal_nan=True)
    return projected.astype(np.float32), {
        "lower_violation": lower_violation,
        "upper_violation": upper_violation,
        "nonfinite": nonfinite,
        "support_changed": support_changed,
        "duration_floor_applied": invalid,
        "total_changed": total_changed,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sequences-npz", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--rollouts-per-sequence", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--resume-model", default="")
    p.add_argument("--duration-centered-noise-scale", type=float, default=1.0)
    p.add_argument("--duration-common-noise-scale", type=float, default=1.0)
    p.add_argument(
        "--rollout-support-mode",
        choices=["none", "clip"],
        default="none",
        help="Opt-in projection of generated conditions/durations to empirical train quantiles",
    )
    p.add_argument("--support-quantile-low", type=float, default=0.005)
    p.add_argument("--support-quantile-high", type=float, default=0.995)
    p.add_argument(
        "--enforce-positive-duration",
        action="store_true",
        help="Floor generated duration values even when support clipping is disabled",
    )
    p.add_argument("--min-duration-samples", type=float, default=1.0)
    p.add_argument(
        "--normalization-scale-floor",
        type=float,
        default=0.0,
        help="0 preserves additive std+1e-4; positive values use max(train_std, floor) per feature",
    )
    a = p.parse_args()
    if not 0 <= a.support_quantile_low < a.support_quantile_high <= 1:
        raise ValueError("support quantiles must satisfy 0 <= low < high <= 1")
    if a.min_duration_samples <= 0:
        raise ValueError("--min-duration-samples must be positive")
    if a.normalization_scale_floor < 0:
        raise ValueError("--normalization-scale-floor must be non-negative")
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    out = Path(a.out_dir)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    (out / "data").mkdir(exist_ok=True)
    seq = np.load(a.sequences_npz)
    conditions = seq["conditions"].astype(np.float32)
    durations = seq["durations"].astype(np.float32)
    values = np.concatenate([conditions, durations[..., None]], 2)
    order = np.random.default_rng(a.seed).permutation(len(values))
    split = int(0.8 * len(values))
    train_rows, test_rows = values[order[:split]], values[order[split:]]
    train_flat = train_rows.reshape(-1, values.shape[-1])
    empirical_lower = np.quantile(train_flat, a.support_quantile_low, axis=0).astype(np.float32)
    empirical_upper = np.quantile(train_flat, a.support_quantile_high, axis=0).astype(np.float32)
    support_lower = empirical_lower.copy()
    support_upper = empirical_upper.copy()
    positive_duration_active = a.enforce_positive_duration or a.rollout_support_mode == "clip"
    projection_active = a.rollout_support_mode != "none" or positive_duration_active
    if a.rollout_support_mode == "clip":
        support_lower[-1] = max(float(support_lower[-1]), a.min_duration_samples)
        support_upper[-1] = max(float(support_upper[-1]), float(support_lower[-1]))
    center = train_flat.mean(0)
    raw_scale = train_flat.std(0)
    if a.normalization_scale_floor > 0:
        scale = np.maximum(raw_scale, a.normalization_scale_floor).astype(np.float32)
        normalization_mode = "floored_std"
    else:
        scale = (raw_scale + 1e-4).astype(np.float32)
        normalization_mode = "std_plus_epsilon"
    normalization_audit = {
        "mode": normalization_mode,
        "requested_scale_floor": a.normalization_scale_floor,
        "raw_scale_min": float(raw_scale.min()),
        "saved_scale_min": float(scale.min()),
        "floored_axes": (
            int((raw_scale < a.normalization_scale_floor).sum())
            if a.normalization_scale_floor > 0
            else 0
        ),
        "dimensions": int(len(scale)),
        "source": "current_training_split",
    }
    train_n = (train_rows - center) / scale
    past = []
    target = []
    for row in train_n:
        for start in range(0, row.shape[0] - 5):
            past.append(row[start : start + 3])
            target.append(row[start + 3 : start + 6] - row[start + 2 : start + 3])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(np.stack(past)), torch.from_numpy(np.stack(target))),
        batch_size=a.batch_size,
        shuffle=True,
    )
    device = torch.device(a.device)
    model = ConditionTransition(values.shape[-1]).to(device)
    history = []
    if a.resume_model:
        checkpoint = torch.load(a.resume_model, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        center = checkpoint["center"]
        scale = checkpoint["scale"]
        train_n = (train_rows - center) / scale
        checkpoint_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
        normalization_audit.update(
            mode="checkpoint_saved_scale",
            source="resume_checkpoint",
            saved_scale_min=float(np.min(scale)),
            checkpoint_scale_floor=(
                checkpoint_args.get("normalization_scale_floor")
                if isinstance(checkpoint_args, dict)
                else None
            ),
        )
        print("loaded condition transition", a.resume_model, flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, max(a.epochs, 1), eta_min=a.lr * 0.1
    )
    for epoch in range(a.epochs):
        vals = []
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            mean, logstd = model(x)
            inv = torch.exp(-2 * logstd)
            loss = (0.5 * (y - mean).square() * inv + logstd).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2)
            opt.step()
            vals.append(loss.item())
        scheduler.step()
        history.append(float(np.mean(vals)))
        if (epoch + 1) % 20 == 0:
            print("condition-transition", epoch + 1, history[-1], flush=True)
    feature_names = [f"condition_{i}" for i in range(conditions.shape[-1])] + ["duration_samples"]
    audit_counts = {
        key: np.zeros(values.shape[-1], dtype=np.int64)
        for key in [
            "lower_violation",
            "upper_violation",
            "nonfinite",
            "support_changed",
            "total_changed",
        ]
    }
    duration_floor_count = 0
    proposal_rows = 0
    generated = []
    generated_pre_projection = []
    with torch.no_grad():
        for real in test_rows:
            initial = (real[:3] - center) / scale
            for _ in range(a.rollouts_per_sequence):
                blocks = [initial.copy()]
                raw_blocks = [initial.copy()]
                while sum(len(b) for b in blocks) < real.shape[0]:
                    context = np.concatenate(blocks)[-3:]
                    mean, logstd = model(torch.from_numpy(context[None]).float().to(device))
                    eps = torch.randn_like(mean)
                    noise = torch.exp(logstd) * eps
                    duration_noise = noise[..., -1:]
                    common = duration_noise.mean(1, keepdim=True)
                    noise[..., -1:] = (
                        a.duration_common_noise_scale * common
                        + a.duration_centered_noise_scale * (duration_noise - common)
                    )
                    delta = mean + a.temperature * noise
                    nxt = context[-1:] + delta[0].cpu().numpy()
                    raw_nxt = nxt * scale + center
                    constrained, block_audit = constrain_rollout_block(
                        raw_nxt,
                        support_lower,
                        support_upper,
                        a.rollout_support_mode,
                        positive_duration_active,
                        a.min_duration_samples,
                    )
                    used = min(len(constrained), real.shape[0] - sum(len(b) for b in blocks))
                    for key in audit_counts:
                        audit_counts[key] += block_audit[key][:used].sum(0, dtype=np.int64)
                    duration_floor_count += int(block_audit["duration_floor_applied"][:used].sum())
                    proposal_rows += used
                    blocks.append((constrained - center) / scale if projection_active else nxt)
                    raw_blocks.append(nxt)
                gen = np.concatenate(blocks)[: real.shape[0]] * scale + center
                raw_gen = np.concatenate(raw_blocks)[: real.shape[0]] * scale + center
                generated.append(gen)
                generated_pre_projection.append(raw_gen)
    generated = np.stack(generated)
    generated_pre_projection = np.stack(generated_pre_projection)
    real_rep = np.repeat(test_rows, a.rollouts_per_sequence, axis=0)
    cond_real = real_rep[:, :, :-1]
    cond_fake = generated[:, :, :-1]
    hr_real = hrv(real_rep[:, :, -1], 500.0)
    hr_fake = hrv(generated[:, :, -1], 500.0)
    metrics = {
        "condition_flat_mmd": mmd(
            cond_real.reshape(len(cond_real), -1), cond_fake.reshape(len(cond_fake), -1)
        ),
        "condition_mean_abs_diff": float(
            np.mean(np.abs(cond_real.mean((0, 1)) - cond_fake.mean((0, 1))))
        ),
        "condition_std_abs_diff": float(
            np.mean(np.abs(cond_real.std((0, 1)) - cond_fake.std((0, 1))))
        ),
        "duration_positive_fraction": float((generated[:, :, -1] > 0).mean()),
        "duration_positive_fraction_before_constraint": float(
            (generated_pre_projection[:, :, -1] > 0).mean()
        ),
        "rollout_support_changed_fraction": float(
            audit_counts["total_changed"].sum() / max(1, proposal_rows * values.shape[-1])
        ),
        "mean_rr_ms_abs_diff": float(abs(hr_real[:, 0].mean() - hr_fake[:, 0].mean())),
        "sdnn_ms_abs_diff": float(abs(hr_real[:, 1].mean() - hr_fake[:, 1].mean())),
        "rmssd_ms_abs_diff": float(abs(hr_real[:, 2].mean() - hr_fake[:, 2].mean())),
        "pnn50_abs_diff": float(abs(hr_real[:, 3].mean() - hr_fake[:, 3].mean())),
        "mean_hr_bpm_abs_diff": float(abs(hr_real[:, 4].mean() - hr_fake[:, 4].mean())),
        "train_sequences": len(train_rows),
        "test_sequences": len(test_rows),
    }
    support_audit = {
        "enabled": a.rollout_support_mode != "none" or a.enforce_positive_duration,
        "support_mode": a.rollout_support_mode,
        "positive_duration_active": positive_duration_active,
        "support_quantiles": {"low": a.support_quantile_low, "high": a.support_quantile_high},
        "min_duration_samples": a.min_duration_samples,
        "feature_names": feature_names,
        "empirical_lower": dict(zip(feature_names, map(float, empirical_lower))),
        "empirical_upper": dict(zip(feature_names, map(float, empirical_upper))),
        "effective_lower": dict(zip(feature_names, map(float, support_lower))),
        "effective_upper": dict(zip(feature_names, map(float, support_upper))),
        "proposal_rows": int(proposal_rows),
        "proposal_values": int(proposal_rows * values.shape[-1]),
        "lower_violation_count": dict(
            zip(feature_names, map(int, audit_counts["lower_violation"]))
        ),
        "upper_violation_count": dict(
            zip(feature_names, map(int, audit_counts["upper_violation"]))
        ),
        "nonfinite_count": dict(zip(feature_names, map(int, audit_counts["nonfinite"]))),
        "support_changed_count": dict(
            zip(feature_names, map(int, audit_counts["support_changed"]))
        ),
        "total_changed_count": dict(zip(feature_names, map(int, audit_counts["total_changed"]))),
        "duration_floor_applied_count": int(duration_floor_count),
        "changed_value_fraction": float(
            audit_counts["total_changed"].sum() / max(1, proposal_rows * values.shape[-1])
        ),
        "duration_positive_fraction_before": float((generated_pre_projection[:, :, -1] > 0).mean()),
        "duration_positive_fraction_after": float((generated[:, :, -1] > 0).mean()),
        "normalization": normalization_audit,
    }
    (out / "metrics/metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "metrics/history.json").write_text(json.dumps(history))
    (out / "metrics/transition_support_audit.json").write_text(json.dumps(support_audit, indent=2))
    torch.save(
        {
            "model": model.state_dict(),
            "center": center,
            "scale": scale,
            "args": vars(a),
            "normalization": normalization_audit,
            "support_lower": support_lower,
            "support_upper": support_upper,
            "feature_names": np.asarray(feature_names),
        },
        out / "models/condition_transition.pt",
    )
    np.savez_compressed(
        out / "data/condition_rollouts.npz",
        real=real_rep,
        generated=generated,
        generated_pre_projection=generated_pre_projection,
        support_lower=support_lower,
        support_upper=support_upper,
        feature_names=np.asarray(feature_names),
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
