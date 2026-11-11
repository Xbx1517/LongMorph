"""Evaluate fixed-budget forecasting or validation-selected augmentation.

Fixed-budget forecasting uses five input cycles and one target cycle with a
PatchTST predictor. Augmentation uses three input and three target cycles with
TCN predictors. Dataset split, training objective and budget are explicit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .export import export_pairs, extract_sequences
from .metrics import metrics
from .pairs import (
    draw_indices,
    limit_windows,
    pairs_from_windows,
    read_sources,
    real_pairs,
    source_windows,
    split_sources,
    subject_splits,
    synthetic_pairs,
)
from .reporting import summarize_fixed_budget
from .training import fit_model, predict


def subject_fold_data(args):
    with np.load(args.prepared_npz, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    splits = subject_splits(data, args.folds)
    folds = []
    generators = [value.strip() for value in args.generators.split(",") if value.strip()]
    if not generators:
        raise ValueError("At least one generator name is required")
    for split in splits:
        fold = split["fold"]
        real = real_pairs(data, set(split["train"]))
        validation = limit_windows(
            *real_pairs(data, set(split["validation"])), args.max_validation_windows, 500_000 + fold
        )
        test = limit_windows(
            *real_pairs(data, set(split["test"])), args.max_test_windows, 600_000 + fold
        )
        synthetic = {
            name: synthetic_pairs(
                args.synthetic_root / name / f"fold_{fold:02d}.npz", set(split["train"])
            )
            for name in generators
        }
        folds.append(
            {
                "fold": fold,
                "real": real,
                "validation": validation,
                "test": test,
                "synthetic": synthetic,
                "seed_offset": fold * 1000,
                "split": split,
            }
        )
    return folds


def source_holdout_data(args):
    train_sources = read_sources(args.train_sources)
    test_sources = read_sources(args.test_sources)
    if set(train_sources) & set(test_sources):
        raise ValueError("Training and test source files overlap")
    train_set, validation_set = split_sources(
        train_sources, args.validation_fraction, args.validation_min_sources, args.split_seed
    )
    source_to_id = {
        source: index for index, source in enumerate(sorted(set(train_sources) | set(test_sources)))
    }
    with np.load(args.train_sequences, allow_pickle=False) as sequences:
        cycles = np.asarray(sequences["cycles"])
        durations = np.asarray(sequences["durations"])
    if len(cycles) != len(train_sources):
        raise ValueError("Training source count does not match training cycle sequences")
    rows = {}
    for name, source_set in [("real", train_set), ("validation", validation_set)]:
        mask = np.asarray([source in source_set for source in train_sources])
        windows = source_windows(
            cycles[mask],
            durations[mask],
            [source for source in train_sources if source in source_set],
            source_to_id,
            args.expected_cycles,
            args.real_window_stride,
        )
        rows[name] = pairs_from_windows(*windows)
    sequences, sequence_durations, segment_ids, counts = extract_sequences(
        args.test_segments,
        args.sample_rate,
        args.expected_cycles,
        expected_segments=len(test_sources),
    )
    windows = []
    window_durations = []
    subjects = []
    for row, duration, segment_id in zip(sequences, sequence_durations, segment_ids):
        if segment_id >= len(test_sources):
            raise ValueError("Detected test segment has no source metadata")
        source = test_sources[segment_id]
        for start in range(0, len(row) - 5, args.real_window_stride):
            windows.append(row[start : start + 6])
            window_durations.append(duration[start : start + 6])
            subjects.append(source_to_id[source])
    if not windows:
        raise ValueError("No six-cycle windows extracted from test segments")
    rows["test"] = pairs_from_windows(
        np.asarray(windows, np.float32),
        np.asarray(window_durations, np.float32),
        np.asarray(subjects, np.int32),
    )
    rows["validation"] = limit_windows(*rows["validation"], args.max_validation_windows, 500_001)
    rows["test"] = limit_windows(*rows["test"], args.max_test_windows, 600_001)
    sx, sy, generated_counts = export_pairs(
        args.generated_npz,
        args.sample_rate,
        args.expected_cycles,
        5,
        1,
        args.real_prefix_cycles,
        args.pair_stride,
    )
    split = {
        "train_sources": sorted(train_set),
        "validation_sources": sorted(validation_set),
        "test_sources": sorted(set(test_sources)),
        "test_detected_cycle_counts": counts.tolist(),
        "generated_detected_cycle_counts": generated_counts.tolist(),
    }
    return [
        {
            "fold": 1,
            **rows,
            "synthetic": {args.generators: (sx, sy)},
            "seed_offset": 0,
            "split": split,
        }
    ]


def fixed_training_groups(real_x, real_y, synthetic, budget, seed, seed_offset, groups):
    """Use equal pair counts across real, synthetic and half-mixed groups."""
    result = []
    if "real_only" in groups:
        index = draw_indices(len(real_x), budget, 100_000 + seed_offset + seed)
        result.append(("__shared__", "real_only", real_x[index], real_y[index], budget, 0, seed))
    for name, (sx, sy) in synthetic.items():
        if "synthetic_only" in groups:
            index = draw_indices(len(sx), budget, 200_000 + seed_offset + seed)
            result.append(
                (name, "synthetic_only", sx[index], sy[index], 0, budget, 10_000_000 + seed)
            )
        if "half_real_half_generated" in groups:
            real_count = budget // 2
            synthetic_count = budget - real_count
            ri = draw_indices(len(real_x), real_count, 300_000 + seed_offset + seed)
            si = draw_indices(len(sx), synthetic_count, 400_000 + seed_offset + seed)
            x = np.concatenate([real_x[ri], sx[si]]).astype(np.float32)
            y = np.concatenate([real_y[ri], sy[si]]).astype(np.float32)
            result.append(
                (
                    name,
                    "half_real_half_generated",
                    x,
                    y,
                    real_count,
                    synthetic_count,
                    10_000_000 + seed,
                )
            )
    return result


def run_fixed_budget(args):
    folds = subject_fold_data(args) if args.split == "subject_folds" else source_holdout_data(args)
    output = args.output_dir
    for folder in ("metrics", "splits", "configs"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in args.seeds.split(",")]
    groups = [value.strip() for value in args.groups.split(",") if value.strip()]
    allowed = {"real_only", "synthetic_only", "half_real_half_generated"}
    if not groups or not set(groups).issubset(allowed):
        raise ValueError(f"Groups must be selected from {sorted(allowed)}")
    per_window = []
    training_rows = []
    budgets = {}
    device = torch.device(args.device)
    for fold in folds:
        real_x, real_y, _, _ = fold["real"]
        vx, vy, _, _ = fold["validation"]
        tx, ty, td, ts = fold["test"]
        synthetic = fold["synthetic"]
        budget = args.train_budget or min(
            len(real_x), min(len(sx) for sx, _ in synthetic.values()), args.train_budget_cap
        )
        if budget < args.minimum_train_pairs:
            raise ValueError(f"Fold {fold['fold']} budget {budget} is below --minimum-train-pairs")
        budgets[fold["fold"]] = budget
        for seed in seeds:
            for generator, group, x, y, n_real, n_synthetic, model_seed in fixed_training_groups(
                real_x, real_y, synthetic, budget, seed, fold["seed_offset"], groups
            ):
                model, info = fit_model(x, y, vx, vy, args, model_seed, device)
                frame = metrics(predict(model, tx, args.batch_size, device), ty, td, ts)
                frame["generator"] = generator
                frame["group"] = group
                frame["fold"] = fold["fold"]
                frame["seed"] = seed
                per_window.append(frame)
                training_rows.append(
                    {
                        "generator": generator,
                        "group": group,
                        "fold": fold["fold"],
                        "seed": seed,
                        "train_pairs": len(x),
                        "real_train_pairs": n_real,
                        "synthetic_train_pairs": n_synthetic,
                        **info,
                    }
                )
    frame = pd.concat(per_window, ignore_index=True)
    frame.to_csv(output / "metrics/per_window.csv", index=False)
    pd.DataFrame(training_rows).to_csv(output / "metrics/training_runs.csv", index=False)
    paths = summarize_fixed_budget(frame, output)
    (output / "splits/data_splits.json").write_text(
        json.dumps([fold["split"] for fold in folds], indent=2), encoding="utf-8"
    )
    config = {**vars(args), "budgets_by_fold": budgets}
    (output / "configs/forecasting_config.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )
    print(
        json.dumps({"output": str(output), "training_runs": len(training_rows), **paths}, indent=2)
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", choices=("fixed_budget", "augmentation"), default="fixed_budget"
    )
    parser.add_argument(
        "--split", choices=("subject_folds", "source_holdout"), default="subject_folds"
    )
    parser.add_argument("--model", choices=("patchtst", "tcn", "residual_tcn"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prepared-npz", type=Path)
    parser.add_argument(
        "--synthetic-root",
        type=Path,
        help="Fixed-budget: GENERATOR/fold_XX.npz; augmentation: fold_XX.npz",
    )
    parser.add_argument("--generators", default="longmorph")
    parser.add_argument("--train-sequences", type=Path)
    parser.add_argument("--train-sources", type=Path)
    parser.add_argument("--test-segments", type=Path)
    parser.add_argument("--test-sources", type=Path)
    parser.add_argument("--generated-npz", type=Path)
    parser.add_argument("--seeds", default="7,17,27,37,47")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--input-cycles", type=int)
    parser.add_argument("--target-cycles", type=int)
    parser.add_argument("--sample-rate", type=float, default=500.0)
    parser.add_argument("--expected-cycles", type=int, default=24)
    parser.add_argument("--real-window-stride", type=int, default=1)
    parser.add_argument("--real-prefix-cycles", type=int, default=3)
    parser.add_argument("--pair-stride", type=int, default=1)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--validation-min-sources", type=int, default=20)
    parser.add_argument("--split-seed", type=int, default=1701)
    parser.add_argument("--groups", default="real_only,synthetic_only,half_real_half_generated")
    parser.add_argument("--train-budget", type=int, default=0)
    parser.add_argument("--train-budget-cap", type=int, default=512)
    parser.add_argument("--minimum-train-pairs", type=int)
    parser.add_argument("--max-validation-windows", type=int, default=0)
    parser.add_argument("--max-test-windows", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epoch-size", type=int, default=0)
    parser.add_argument("--epoch-size-floor", type=int, default=1024)
    parser.add_argument("--epoch-size-cap", type=int)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--patch-len", type=int, default=64)
    parser.add_argument("--patch-stride", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--e-layers", type=int, default=3)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--disable-revin", action="store_true")
    parser.add_argument("--loss-d1-weight", type=float, default=0.15)
    parser.add_argument("--loss-d2-weight", type=float, default=0.05)
    parser.add_argument("--loss-spectral-weight", type=float, default=0.02)
    parser.add_argument("--loss-correlation-weight", type=float, default=0.05)
    parser.add_argument("--residual-scale", type=float, default=0.5)
    parser.add_argument("--loss-raw-kind", choices=("l1", "smooth_l1"), default="l1")
    parser.add_argument("--loss-peak-weight", type=float, default=0.05)
    parser.add_argument("--peak-softmax-temperature", type=float, default=20.0)
    parser.add_argument(
        "--spectral-loss-mode",
        choices=("phase_welch_log_power", "cycle_log_amplitude"),
        default="phase_welch_log_power",
    )
    parser.add_argument(
        "--evaluation-groups",
        default="real_only,real_plus_synthetic_unfiltered,real_plus_synthetic",
    )
    parser.add_argument("--real-fractions", default="0.05,1.0")
    parser.add_argument("--synthetic-ratios", default="0.10,0.25,0.50")
    parser.add_argument("--synthetic-weights", default="0.25,0.50")
    parser.add_argument("--quality-keep-fractions", default="0.50,0.75")
    parser.add_argument("--quality-minimum-pairs", type=int, default=64)
    parser.add_argument("--quality-feature-weights", default="1,1,1,1.5,2,1,1,1,1,2,2,1.5")
    parser.add_argument("--tuning-seed", type=int, default=1701)
    parser.add_argument("--tune-epochs", type=int, default=45)
    parser.add_argument("--fine-tune-epochs", type=int, default=12)
    parser.add_argument("--fine-tune-lr-scale", type=float, default=0.20)
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    parser.add_argument("--device", default="cuda")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.split == "subject_folds" and (args.prepared_npz is None or args.synthetic_root is None):
        parser.error("--prepared-npz and --synthetic-root are required for subject_folds")
    if args.split == "source_holdout":
        required = (
            "train_sequences",
            "train_sources",
            "test_segments",
            "test_sources",
            "generated_npz",
        )
        if any(getattr(args, key) is None for key in required):
            parser.error(
                "source_holdout requires --train-sequences, --train-sources, --test-segments, --test-sources and --generated-npz"
            )
        if "," in args.generators:
            parser.error("source_holdout accepts one generator name")
    is_fixed = args.protocol == "fixed_budget"
    args.model = args.model or ("patchtst" if is_fixed else "residual_tcn")
    required_shape = (5, 1) if is_fixed else (3, 3)
    args.input_cycles = args.input_cycles or required_shape[0]
    args.target_cycles = args.target_cycles or required_shape[1]
    if (args.input_cycles, args.target_cycles) != required_shape:
        parser.error(f"{args.protocol} requires input/target cycles {required_shape}")
    if is_fixed and args.model != "patchtst":
        parser.error("fixed_budget uses the PatchTST five-to-one objective")
    if not is_fixed and (args.model == "patchtst" or args.split != "subject_folds"):
        parser.error("augmentation uses tcn/residual_tcn and subject_folds")
    args.minimum_train_pairs = args.minimum_train_pairs or (
        128 if args.split == "subject_folds" else 64
    )
    args.max_epochs = args.max_epochs or (80 if is_fixed else 100)
    args.patience = args.patience or (10 if is_fixed else 12)
    args.epoch_size_cap = args.epoch_size_cap or (4096 if is_fixed else 8192)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA unavailable; using CPU.", file=sys.stderr)
        args.device = "cpu"
    if is_fixed:
        run_fixed_budget(args)
    else:
        from .augmentation import run

        run(args)


if __name__ == "__main__":
    main()
