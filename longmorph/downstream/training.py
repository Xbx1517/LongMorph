"""Shared forecaster training with explicit task-specific objectives."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, TensorDataset

from .models import PatchTSTForecaster, TCNForecaster, ResidualTCNForecaster


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def spectral_log_power(signal: torch.Tensor) -> torch.Tensor:
    centered = signal.reshape(len(signal), -1)
    centered = centered - centered.mean(dim=-1, keepdim=True)
    window = torch.hann_window(
        centered.shape[-1],
        periodic=True,
        device=signal.device,
        dtype=signal.dtype,
    )
    power = torch.fft.rfft(centered * window, dim=-1).abs().square()
    return torch.log(power.clamp_min(1e-12))


def fixed_budget_loss(prediction: torch.Tensor, target: torch.Tensor, args) -> torch.Tensor:
    raw = F.smooth_l1_loss(prediction, target)
    d1 = F.l1_loss(torch.diff(prediction, dim=-1), torch.diff(target, dim=-1))
    d2 = F.l1_loss(torch.diff(prediction, n=2, dim=-1), torch.diff(target, n=2, dim=-1))
    spectrum = F.smooth_l1_loss(spectral_log_power(prediction), spectral_log_power(target))
    pred_flat = prediction.reshape(len(prediction), -1)
    target_flat = target.reshape(len(target), -1)
    pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
    corr = (pred_centered * target_centered).sum(dim=1) / (
        pred_centered.square().sum(dim=1).sqrt() * target_centered.square().sum(dim=1).sqrt()
    ).clamp_min(1e-8)
    corr_loss = (1.0 - corr.clamp(-1.0, 1.0)).mean()
    return (
        raw
        + args.loss_d1_weight * d1
        + args.loss_d2_weight * d2
        + args.loss_spectral_weight * spectrum
        + args.loss_correlation_weight * corr_loss
    )


def predict(
    model: torch.nn.Module,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            outputs.append(
                model(torch.from_numpy(x[start : start + batch_size]).to(device)).cpu().numpy()
            )
    return np.concatenate(outputs)


def phase_welch_log_power(signal: torch.Tensor) -> torch.Tensor:
    """Differentiable counterpart of the reported three-cycle Welch PSD.

    scipy.signal.welch uses 256-point Hann windows, 50% overlap and per-window
    constant detrending for the downstream metric.  Scale factors shared by
    prediction and target cancel in log-space, so only the aligned power shape
    is needed here.
    """
    flat = signal.reshape(len(signal), -1)
    segments = flat.unfold(-1, 256, 128)
    segments = segments - segments.mean(dim=-1, keepdim=True)
    window = torch.hann_window(256, periodic=True, device=signal.device, dtype=signal.dtype)
    power = torch.fft.rfft(segments * window, dim=-1).abs().square().mean(dim=-2)
    return torch.log(power.clamp_min(1e-12))


def per_sample_loss_components(
    prediction: torch.Tensor, target: torch.Tensor, args
) -> dict[str, torch.Tensor]:
    if args.loss_raw_kind == "l1":
        raw = F.l1_loss(prediction, target, reduction="none").mean((1, 2))
    else:
        raw = F.smooth_l1_loss(prediction, target, reduction="none").mean((1, 2))
    d1 = F.l1_loss(
        torch.diff(prediction, dim=-1), torch.diff(target, dim=-1), reduction="none"
    ).mean((1, 2))
    d2 = F.l1_loss(
        torch.diff(prediction, n=2, dim=-1),
        torch.diff(target, n=2, dim=-1),
        reduction="none",
    ).mean((1, 2))
    if args.spectral_loss_mode == "phase_welch_log_power":
        prediction_spectrum = phase_welch_log_power(prediction)
        target_spectrum = phase_welch_log_power(target)
    else:
        prediction_spectrum = torch.log1p(torch.abs(torch.fft.rfft(prediction, dim=-1)))
        target_spectrum = torch.log1p(torch.abs(torch.fft.rfft(target, dim=-1)))
    spectral = (
        F.smooth_l1_loss(
            prediction_spectrum,
            target_spectrum,
            reduction="none",
        )
        .reshape(len(prediction), -1)
        .mean(1)
    )
    prediction_flat = prediction.reshape(len(prediction), -1)
    target_flat = target.reshape(len(target), -1)
    prediction_centered = prediction_flat - prediction_flat.mean(1, keepdim=True)
    target_centered = target_flat - target_flat.mean(1, keepdim=True)
    correlation = (prediction_centered * target_centered).sum(1) / (
        prediction_centered.square().sum(1).sqrt() * target_centered.square().sum(1).sqrt()
    ).clamp_min(1e-8)
    correlation_loss = 1.0 - correlation.clamp(-1.0, 1.0)
    phase = torch.linspace(
        0.0,
        1.0,
        prediction.shape[-1],
        device=prediction.device,
        dtype=prediction.dtype,
    )
    prediction_scale = prediction.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4)
    target_scale = target.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-4)
    prediction_peak_phase = (
        torch.softmax(
            args.peak_softmax_temperature
            * (prediction - prediction.mean(dim=-1, keepdim=True))
            / prediction_scale,
            dim=-1,
        )
        * phase
    ).sum(dim=-1)
    target_peak_phase = (
        torch.softmax(
            args.peak_softmax_temperature
            * (target - target.mean(dim=-1, keepdim=True))
            / target_scale,
            dim=-1,
        )
        * phase
    ).sum(dim=-1)
    peak_phase = F.smooth_l1_loss(
        prediction_peak_phase,
        target_peak_phase,
        reduction="none",
    ).mean(dim=1)
    return {
        "raw": raw,
        "d1": d1,
        "d2": d2,
        "spectral": spectral,
        "correlation": correlation_loss,
        "peak_phase": peak_phase,
    }


def per_sample_loss(prediction: torch.Tensor, target: torch.Tensor, args) -> torch.Tensor:
    component = per_sample_loss_components(prediction, target, args)
    return (
        component["raw"]
        + args.loss_d1_weight * component["d1"]
        + args.loss_d2_weight * component["d2"]
        + args.loss_spectral_weight * component["spectral"]
        + args.loss_correlation_weight * component["correlation"]
        + args.loss_peak_weight * component["peak_phase"]
    )


def build_model(args):
    if args.model == "patchtst":
        return PatchTSTForecaster(
            input_cycles=args.input_cycles,
            target_cycles=args.target_cycles,
            patch_len=args.patch_len,
            stride=args.patch_stride,
            d_model=args.d_model,
            n_heads=args.n_heads,
            e_layers=args.e_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            revin=not args.disable_revin,
        )
    if args.model == "tcn":
        return TCNForecaster()
    return ResidualTCNForecaster(residual_scale=args.residual_scale)


def train_loader(x, y, weights, batch_size, epoch_size, seed):
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(weights))
    sampler = RandomSampler(
        dataset,
        replacement=True,
        num_samples=epoch_size,
        generator=torch.Generator().manual_seed(seed),
    )
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def eval_loss(model, x, y, batch_size, device, args):
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device)
            yb = torch.from_numpy(y[start : start + batch_size]).to(device)
            prediction = model(xb)
            loss = (
                fixed_budget_loss(prediction, yb, args)
                if args.protocol == "fixed_budget"
                else per_sample_loss(prediction, yb, args).mean()
            )
            values.append(float(loss.detach().cpu()))
    return float(np.mean(values))


def fit_model(
    train_x,
    train_y,
    validation_x,
    validation_y,
    args,
    seed,
    device,
    weights=None,
    real_data=None,
    max_epochs=None,
    fine_tune_epochs=0,
):
    seed_all(seed)
    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    floor = args.epoch_size_floor if args.protocol == "fixed_budget" else 2 * len(validation_x)
    epoch_size = args.epoch_size or min(max(floor, len(train_x)), args.epoch_size_cap)
    epoch_size = max(epoch_size, args.batch_size)
    weights = np.ones(len(train_x), np.float32) if weights is None else weights
    loader = train_loader(train_x, train_y, weights, args.batch_size, epoch_size, seed)
    best_state = None
    best_validation = float("inf")
    stale = 0
    epochs_run = 0
    for epoch in range(max_epochs or args.max_epochs):
        model.train()
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            optimizer.zero_grad()
            prediction = model(xb)
            if args.protocol == "fixed_budget":
                loss = fixed_budget_loss(prediction, yb, args)
            else:
                loss = (per_sample_loss(prediction, yb, args) * wb).sum() / wb.sum().clamp_min(1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        validation = eval_loss(model, validation_x, validation_y, args.batch_size, device, args)
        epochs_run = epoch + 1
        if validation < best_validation - args.min_delta:
            best_validation = validation
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No finite validation checkpoint selected")
    model.load_state_dict(best_state)
    best_stage = "mixed"
    if fine_tune_epochs > 0 and real_data is not None and len(real_data[0]):
        real_x, real_y = real_data
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr * args.fine_tune_lr_scale, weight_decay=args.weight_decay
        )
        real_epoch_size = min(max(len(real_x), args.batch_size), args.epoch_size_cap)
        real_loader = train_loader(
            real_x,
            real_y,
            np.ones(len(real_x), np.float32),
            args.batch_size,
            real_epoch_size,
            seed + 100_000,
        )
        for epoch in range(fine_tune_epochs):
            model.train()
            for xb, yb, _ in real_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = per_sample_loss(model(xb), yb, args).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            validation = eval_loss(model, validation_x, validation_y, args.batch_size, device, args)
            if validation < best_validation - args.min_delta:
                best_validation = validation
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_stage = f"real_finetune_{epoch+1}"
        model.load_state_dict(best_state)
    return model, {
        "best_val_loss": float(best_validation),
        "epochs": epochs_run,
        "epoch_size": epoch_size,
        "best_stage": best_stage,
    }
