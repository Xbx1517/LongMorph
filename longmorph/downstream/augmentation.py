"""Training-only filtering and validation-selected data augmentation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .metrics import metrics
from .pairs import real_pairs, synthetic_pairs
from .training import fit_model as fit_forecaster, predict
from .reporting import summarize_augmentation

FEATURE_NAMES = [
    "amplitude_mean",
    "amplitude_std",
    "signal_std_mean",
    "d1_rms_mean",
    "d2_rms_mean",
    "peak_phase_mean",
    "peak_phase_std",
    "area_mean",
    "area_std",
    "spectral_tail_mean",
    "spectral_tail_std",
    "adjacent_cycle_correlation",
]


def pair_features(x: np.ndarray, y: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    rows = []
    for start in range(0, len(x), batch_size):
        cycles = np.concatenate(
            [x[start : start + batch_size], y[start : start + batch_size]], axis=1
        ).astype(np.float64)
        low = np.quantile(cycles, 0.05, axis=2)
        high = np.quantile(cycles, 0.95, axis=2)
        amplitude = np.maximum(high - low, 1e-6)
        standard = cycles.std(2)
        d1 = np.sqrt(np.mean(np.diff(cycles, axis=2) ** 2, axis=2)) / amplitude
        d2 = np.sqrt(np.mean(np.diff(cycles, n=2, axis=2) ** 2, axis=2)) / amplitude
        peak_phase = cycles.argmax(2) / float(cycles.shape[2] - 1)
        normalized = (cycles - cycles.min(2, keepdims=True)) / np.maximum(
            cycles.max(2, keepdims=True) - cycles.min(2, keepdims=True),
            1e-6,
        )
        area = normalized.mean(2)
        spectrum = np.abs(np.fft.rfft(cycles - cycles.mean(2, keepdims=True), axis=2)) ** 2
        tail = spectrum[:, :, 8:].sum(2) / np.maximum(spectrum[:, :, 1:].sum(2), 1e-12)
        left = normalized[:, :-1] - normalized[:, :-1].mean(2, keepdims=True)
        right = normalized[:, 1:] - normalized[:, 1:].mean(2, keepdims=True)
        correlation = (left * right).sum(2) / np.maximum(
            np.sqrt((left**2).sum(2) * (right**2).sum(2)),
            1e-8,
        )
        rows.append(
            np.stack(
                [
                    amplitude.mean(1),
                    amplitude.std(1),
                    standard.mean(1),
                    d1.mean(1),
                    d2.mean(1),
                    peak_phase.mean(1),
                    peak_phase.std(1),
                    area.mean(1),
                    area.std(1),
                    tail.mean(1),
                    tail.std(1),
                    correlation.mean(1),
                ],
                axis=1,
            ).astype(np.float32)
        )
    return np.concatenate(rows)


def quality_filter(
    real_x,
    real_y,
    synthetic_x,
    synthetic_y,
    keep_fraction,
    minimum_pairs,
    feature_weights: np.ndarray,
):
    real_feature = pair_features(real_x, real_y)
    synthetic_feature = pair_features(synthetic_x, synthetic_y)
    center = np.median(real_feature, axis=0)
    scale = 1.4826 * np.median(np.abs(real_feature - center), axis=0)
    scale = np.maximum(scale, np.maximum(np.abs(center) * 0.02, 1e-5))
    real_z = np.clip(np.abs((real_feature - center) / scale), 0, 20)
    synthetic_z = np.clip(np.abs((synthetic_feature - center) / scale), 0, 20)
    weights = np.asarray(feature_weights, dtype=np.float32)
    if weights.shape != (len(FEATURE_NAMES),) or np.any(weights <= 0):
        raise ValueError(
            f"quality feature weights must contain {len(FEATURE_NAMES)} positive values"
        )
    real_score = np.sqrt(np.sum(real_z**2 * weights, axis=1) / weights.sum())
    synthetic_score = np.sqrt(np.sum(synthetic_z**2 * weights, axis=1) / weights.sum())
    count = max(minimum_pairs, int(round(keep_fraction * len(synthetic_score))))
    count = min(count, len(synthetic_score))
    keep = np.argsort(synthetic_score)[:count]
    audit = {
        "feature_names": FEATURE_NAMES,
        "feature_weights": weights.tolist(),
        "real_center": center.tolist(),
        "real_robust_scale": scale.tolist(),
        "real_score_quantiles": {
            str(q): float(np.quantile(real_score, q)) for q in (0.5, 0.9, 0.95, 0.99)
        },
        "synthetic_score_quantiles": {
            str(q): float(np.quantile(synthetic_score, q)) for q in (0.1, 0.5, 0.75, 0.9, 0.95)
        },
        "input_pairs": int(len(synthetic_score)),
        "kept_pairs": int(len(keep)),
        "keep_fraction": float(len(keep) / len(synthetic_score)),
        "selected_score_max": float(synthetic_score[keep].max()),
    }
    return synthetic_x[keep], synthetic_y[keep], audit


def subject_stratified_budget(x, y, subjects, fraction, seed):
    if fraction >= 1:
        return x, y, subjects
    rng = np.random.default_rng(seed)
    chosen = []
    for subject in np.unique(subjects):
        index = np.flatnonzero(subjects == subject)
        count = max(1, int(round(fraction * len(index))))
        chosen.extend(rng.choice(index, count, replace=False).tolist())
    chosen = np.asarray(sorted(chosen), dtype=int)
    return x[chosen], y[chosen], subjects[chosen]


def draw_synthetic(x, y, count, seed):
    if count <= 0:
        return x[:0], y[:0]
    rng = np.random.default_rng(seed)
    index = rng.choice(len(x), count, replace=count > len(x))
    return x[index], y[index]


def mixed_data(real_x, real_y, synthetic_x, synthetic_y, ratio, synthetic_weight, seed):
    count = int(round(ratio * len(real_x)))
    sx, sy = draw_synthetic(synthetic_x, synthetic_y, count, seed)
    if not len(sx):
        return real_x, real_y, np.ones(len(real_x), np.float32), 0
    x = np.concatenate([real_x, sx])
    y = np.concatenate([real_y, sy])
    weights = np.concatenate(
        [
            np.ones(len(real_x), np.float32),
            np.full(len(sx), synthetic_weight, np.float32),
        ]
    )
    return x, y, weights, len(sx)


def fit_model(
    train_x,
    train_y,
    train_weights,
    validation_x,
    validation_y,
    real_x,
    real_y,
    args,
    seed,
    max_epochs,
    fine_tune_epochs,
):
    return fit_forecaster(
        train_x,
        train_y,
        validation_x,
        validation_y,
        args,
        seed,
        torch.device(args.device),
        weights=train_weights,
        real_data=(real_x, real_y),
        max_epochs=max_epochs,
        fine_tune_epochs=fine_tune_epochs,
    )


def run(args):
    quality_keep_fractions = [float(value) for value in args.quality_keep_fractions.split(",")]
    if any(not 0 < value <= 1 for value in quality_keep_fractions):
        raise ValueError("quality keep fractions must be in (0,1]")
    quality_feature_weights = np.asarray(
        [float(value) for value in args.quality_feature_weights.split(",")],
        dtype=np.float32,
    )
    if len(quality_feature_weights) != len(FEATURE_NAMES) or np.any(quality_feature_weights <= 0):
        raise ValueError(f"--quality-feature-weights requires {len(FEATURE_NAMES)} positive values")
    real_fractions = [float(value) for value in args.real_fractions.split(",")]
    synthetic_ratios = [float(value) for value in args.synthetic_ratios.split(",")]
    synthetic_weights = [float(value) for value in args.synthetic_weights.split(",")]
    if any(not 0 < value <= 1 for value in real_fractions):
        raise ValueError("real fractions must be in (0,1]")
    if any(value <= 0 for value in synthetic_ratios + synthetic_weights):
        raise ValueError("synthetic ratios/weights must be positive")
    if args.loss_peak_weight < 0 or args.peak_softmax_temperature <= 0:
        raise ValueError("peak loss weight must be non-negative and temperature must be positive")
    evaluation_groups = [
        value.strip() for value in args.evaluation_groups.split(",") if value.strip()
    ]
    allowed_groups = {
        "real_only",
        "synthetic_only",
        "synthetic_only_unfiltered",
        "real_plus_synthetic_unfiltered",
        "real_plus_synthetic",
    }
    if not evaluation_groups or not set(evaluation_groups).issubset(allowed_groups):
        raise ValueError(f"evaluation groups must be selected from {sorted(allowed_groups)}")

    archive = np.load(args.prepared_npz, allow_pickle=False)
    data = {key: archive[key] for key in archive.files}
    subjects = sorted(map(int, np.unique(data["subject_index"])))
    if len(subjects) < 15:
        raise ValueError("Subject-fold augmentation requires at least 15 subjects")
    output = Path(args.output_dir)
    for folder in ("metrics", "splits", "figures", "configs"):
        (output / folder).mkdir(parents=True, exist_ok=True)

    splits = []
    selection_rows = []
    quality_audits = []
    window_rows = []
    seed_rows = []
    evaluation_seeds = [int(value) for value in args.seeds.split(",")]
    device = torch.device(args.device)
    for fold in range(args.folds):
        test_subjects = set(subjects[fold * 3 : (fold + 1) * 3])
        remaining = [subject for subject in subjects if subject not in test_subjects]
        validation_subjects = set(remaining[:2])
        train_subjects = set(remaining[2:])
        splits.append(
            {
                "fold": fold + 1,
                "train": sorted(train_subjects),
                "validation": sorted(validation_subjects),
                "test": sorted(test_subjects),
            }
        )
        full_x, full_y, _, full_subject = real_pairs(data, train_subjects, 3, 3)
        validation_x, validation_y, _, _ = real_pairs(data, validation_subjects, 3, 3)
        test_x, test_y, test_duration, test_subject = real_pairs(data, test_subjects, 3, 3)
        raw_sx, raw_sy = synthetic_pairs(
            str(Path(args.synthetic_root) / f"fold_{fold + 1:02d}.npz"),
            train_subjects,
            3,
            3,
        )
        for fraction in real_fractions:
            budget_x, budget_y, budget_subject = subject_stratified_budget(
                full_x,
                full_y,
                full_subject,
                fraction,
                10_000 + fold * 100 + int(round(fraction * 100)),
            )
            filtered_pools = {}
            for keep_fraction in quality_keep_fractions:
                filtered_sx, filtered_sy, audit = quality_filter(
                    budget_x,
                    budget_y,
                    raw_sx,
                    raw_sy,
                    keep_fraction,
                    args.quality_minimum_pairs,
                    quality_feature_weights,
                )
                filtered_pools[keep_fraction] = (filtered_sx, filtered_sy)
                audit.update(
                    {
                        "fold": fold + 1,
                        "real_fraction": fraction,
                        "real_budget_pairs": len(budget_x),
                    }
                )
                quality_audits.append(audit)

            candidates = [{"ratio": 0.0, "synthetic_weight": 0.0, "quality_keep_fraction": 0.0}]
            candidates.extend(
                {
                    "ratio": ratio,
                    "synthetic_weight": weight,
                    "quality_keep_fraction": keep_fraction,
                }
                for keep_fraction in quality_keep_fractions
                for ratio in synthetic_ratios
                for weight in synthetic_weights
            )
            best_candidate = None
            best_augmented_candidate = None
            for candidate_index, candidate in enumerate(candidates):
                if candidate["ratio"] > 0:
                    candidate_sx, candidate_sy = filtered_pools[candidate["quality_keep_fraction"]]
                else:
                    candidate_sx, candidate_sy = raw_sx[:0], raw_sy[:0]
                mixed_x, mixed_y, mixed_weight, synthetic_count = mixed_data(
                    budget_x,
                    budget_y,
                    candidate_sx,
                    candidate_sy,
                    candidate["ratio"],
                    candidate["synthetic_weight"],
                    20_000 + fold * 1000 + candidate_index,
                )
                _, info = fit_model(
                    mixed_x,
                    mixed_y,
                    mixed_weight,
                    validation_x,
                    validation_y,
                    budget_x,
                    budget_y,
                    args,
                    args.tuning_seed + fold * 100 + candidate_index,
                    args.tune_epochs,
                    args.fine_tune_epochs if candidate["ratio"] > 0 else 0,
                )
                row = {
                    "fold": fold + 1,
                    "real_fraction": fraction,
                    **candidate,
                    "synthetic_pairs": synthetic_count,
                    **info,
                }
                selection_rows.append(row)
                if best_candidate is None or row["best_val_loss"] < best_candidate["best_val_loss"]:
                    best_candidate = row
                if candidate["ratio"] > 0 and (
                    best_augmented_candidate is None
                    or row["best_val_loss"] < best_augmented_candidate["best_val_loss"]
                ):
                    best_augmented_candidate = row

            groups = {}
            real_weights = np.ones(len(budget_x), np.float32)
            if "real_only" in evaluation_groups:
                groups["real_only"] = (budget_x, budget_y, real_weights, 0, False, best_candidate)
            if best_candidate["ratio"] > 0:
                selected_sx, selected_sy = filtered_pools[best_candidate["quality_keep_fraction"]]
            else:
                selected_sx, selected_sy = raw_sx[:0], raw_sy[:0]
            if "synthetic_only" in evaluation_groups:
                if best_augmented_candidate is None:
                    raise RuntimeError("no non-zero synthetic candidate was evaluated")
                synthetic_sx, synthetic_sy = filtered_pools[
                    best_augmented_candidate["quality_keep_fraction"]
                ]
                groups["synthetic_only"] = (
                    synthetic_sx,
                    synthetic_sy,
                    np.ones(len(synthetic_sx), np.float32),
                    len(synthetic_sx),
                    False,
                    best_augmented_candidate,
                )
            if "synthetic_only_unfiltered" in evaluation_groups:
                groups["synthetic_only_unfiltered"] = (
                    raw_sx,
                    raw_sy,
                    np.ones(len(raw_sx), np.float32),
                    len(raw_sx),
                    False,
                    best_augmented_candidate or best_candidate,
                )
            for name, source_x, source_y in (
                ("real_plus_synthetic_unfiltered", raw_sx, raw_sy),
                ("real_plus_synthetic", selected_sx, selected_sy),
            ):
                if name not in evaluation_groups:
                    continue
                gx, gy, gw, synthetic_count = mixed_data(
                    budget_x,
                    budget_y,
                    source_x,
                    source_y,
                    best_candidate["ratio"],
                    best_candidate["synthetic_weight"],
                    30_000 + fold * 100 + (0 if name.endswith("unfiltered") else 1),
                )
                groups[name] = (
                    gx,
                    gy,
                    gw,
                    synthetic_count,
                    best_candidate["ratio"] > 0,
                    best_candidate,
                )

            for group, (
                group_x,
                group_y,
                group_weight,
                synthetic_count,
                use_finetune,
                group_candidate,
            ) in groups.items():
                for seed in evaluation_seeds:
                    model, info = fit_model(
                        group_x,
                        group_y,
                        group_weight,
                        validation_x,
                        validation_y,
                        budget_x,
                        budget_y,
                        args,
                        seed,
                        args.max_epochs,
                        args.fine_tune_epochs if use_finetune else 0,
                    )
                    prediction = predict(model, test_x, args.batch_size, device)
                    frame = metrics(prediction, test_y, test_duration, test_subject)
                    frame["fold"] = fold + 1
                    frame["seed"] = seed
                    frame["group"] = group
                    frame["real_fraction"] = fraction
                    window_rows.append(frame)
                    summary = frame.mean(numeric_only=True).to_dict()
                    summary.update(
                        {
                            "fold": fold + 1,
                            "seed": seed,
                            "group": group,
                            "real_fraction": fraction,
                            "real_train_pairs": len(budget_x),
                            "synthetic_train_pairs": synthetic_count,
                            "selected_ratio": group_candidate["ratio"],
                            "selected_synthetic_weight": group_candidate["synthetic_weight"],
                            "selected_quality_keep_fraction": group_candidate[
                                "quality_keep_fraction"
                            ],
                            **info,
                        }
                    )
                    seed_rows.append(summary)

    per_window = pd.concat(window_rows, ignore_index=True)
    per_window.to_csv(output / "metrics/per_window.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(output / "metrics/per_seed.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(output / "metrics/augmentation_selection.csv", index=False)
    (output / "metrics/quality_filter.json").write_text(json.dumps(quality_audits, indent=2))
    (output / "splits/subject_splits.json").write_text(json.dumps(splits, indent=2))
    (output / "configs/forecasting_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str)
    )

    summarize_augmentation(output, args.bootstrap_repeats)
    plot = per_window.groupby(["real_fraction", "group"]).waveform_mae.mean().unstack()
    plot.plot.bar()
    plt.ylabel("Real-test waveform MAE")
    plt.tight_layout()
    plt.savefig(output / "figures/forecasting_by_budget_and_group.png", dpi=180)
    plt.close()
    print(
        json.dumps(
            {
                "output": str(output),
                "model_runs": len(seed_rows),
                "selection_runs": len(selection_rows),
                "subjects": len(subjects),
            },
            indent=2,
        )
    )
