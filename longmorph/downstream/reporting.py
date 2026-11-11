"""Subject-level aggregation and paired forecasting comparisons."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import METRICS, LOWER_BETTER


def summarize_fixed_budget(
    per_window: pd.DataFrame,
    output_dir: Path,
) -> dict[str, str]:
    per_seed = per_window.groupby(["generator", "group", "fold", "seed"], as_index=False)[
        METRICS
    ].mean()
    per_subject = per_window.groupby(
        ["generator", "group", "fold", "seed", "subject_index"], as_index=False
    )[METRICS].mean()
    summary_rows = []
    for (model, group), frame in per_subject.groupby(["generator", "group"]):
        row: dict[str, object] = {
            "generator": model,
            "group": group,
            "subject_seed_count": int(len(frame)),
        }
        for metric in METRICS:
            row[metric] = float(frame[metric].mean())
            row[f"{metric}_std"] = float(frame[metric].std(ddof=1))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).sort_values(["group", "waveform_mae", "generator"])

    real = per_subject[per_subject["generator"] == "__shared__"].copy()
    real = real.rename(columns={metric: f"{metric}_real" for metric in METRICS})
    paired_rows = []
    for (model, group), frame in per_subject[per_subject["generator"] != "__shared__"].groupby(
        ["generator", "group"]
    ):
        merged = frame.merge(
            real[["fold", "seed", "subject_index"] + [f"{metric}_real" for metric in METRICS]],
            on=["fold", "seed", "subject_index"],
            how="inner",
        )
        if merged.empty:
            continue
        row = {
            "generator": model,
            "group": group,
            "paired_subject_seed_count": int(len(merged)),
        }
        for metric in METRICS:
            delta = merged[metric] - merged[f"{metric}_real"]
            mean = float(delta.mean())
            std = float(delta.std(ddof=1)) if len(delta) > 1 else 0.0
            ci = 1.96 * std / math.sqrt(max(len(delta), 1))
            improved = delta < 0 if metric in LOWER_BETTER else delta > 0
            row[f"{metric}_delta_vs_real_only"] = mean
            row[f"{metric}_ci95_low"] = mean - ci
            row[f"{metric}_ci95_high"] = mean + ci
            row[f"{metric}_improved_subject_seeds"] = int(improved.sum())
        paired_rows.append(row)
    paired = pd.DataFrame(paired_rows)
    if not paired.empty:
        paired = paired.sort_values(["group", "waveform_mae_delta_vs_real_only", "generator"])

    per_seed_path = output_dir / "metrics" / "per_seed.csv"
    per_subject_path = output_dir / "metrics" / "per_subject.csv"
    summary_path = output_dir / "metrics" / "summary_by_generator_group.csv"
    paired_path = output_dir / "metrics" / "paired_delta_vs_real_only.csv"
    per_seed.to_csv(per_seed_path, index=False)
    per_subject.to_csv(per_subject_path, index=False)
    summary.to_csv(summary_path, index=False)
    paired.to_csv(paired_path, index=False)

    markdown_path = output_dir / "metrics" / "forecasting_summary.md"
    write_markdown(summary, paired, markdown_path)
    return {
        "per_seed": str(per_seed_path),
        "per_subject": str(per_subject_path),
        "summary": str(summary_path),
        "paired_delta": str(paired_path),
        "markdown": str(markdown_path),
    }


def fmt(value: object) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return ""
    if abs(number) >= 100:
        return f"{number:.3f}"
    return f"{number:.6g}"


def write_markdown(summary, paired, output):
    columns = ["generator", "group", *METRICS]
    lines = [
        "# Forecasting results",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in summary.iterrows():
        lines.append("| " + " | ".join(fmt(row[c]) for c in columns) + " |")
    if not paired.empty:
        columns = ["generator", "group", *[f"{m}_delta_vs_real_only" for m in METRICS]]
        lines.extend(
            [
                "",
                "# Paired differences from real-only",
                "",
                "| " + " | ".join(columns) + " |",
                "| " + " | ".join(["---"] * len(columns)) + " |",
            ]
        )
        for _, row in paired.iterrows():
            lines.append("| " + " | ".join(fmt(row[c]) for c in columns) + " |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_augmentation(output, bootstrap_repeats=10_000):
    metrics_dir = Path(output) / "metrics"
    window = pd.read_csv(metrics_dir / "per_window.csv")
    subject = window.groupby(
        ["real_fraction", "fold", "seed", "group", "subject_index"],
        as_index=False,
    )[METRICS].mean()
    subject.to_csv(metrics_dir / "per_subject.csv", index=False)
    fold = subject.groupby(["real_fraction", "fold", "group"], as_index=False)[METRICS].mean()
    fold.to_csv(metrics_dir / "fold_summary.csv", index=False)
    overall = {}
    paired = {}
    dataset_group_comparisons = {}
    rng = np.random.default_rng(7)
    for fraction, fraction_frame in subject.groupby("real_fraction"):
        budget = f"{float(fraction):g}"
        overall[budget] = {
            group: {
                metric: {
                    "mean": float(frame[metric].mean()),
                    "std_across_subject_seed": float(frame[metric].std()),
                }
                for metric in METRICS
            }
            for group, frame in fraction_frame.groupby("group")
        }
        averaged = fraction_frame.groupby(["group", "subject_index"], as_index=False)[
            METRICS
        ].mean()
        target = averaged[averaged.group == "real_plus_synthetic"].sort_values("subject_index")
        paired[budget] = {}
        dataset_group_comparisons[budget] = {}
        available_groups = set(averaged.group.unique())
        for reference in ("real_only", "real_plus_synthetic_unfiltered") if len(target) else ():
            if reference not in available_groups:
                continue
            base = averaged[averaged.group == reference].sort_values("subject_index")
            paired[budget][reference] = {}
            if len(base) != len(target):
                raise ValueError(
                    f"Unmatched subjects for fraction={fraction}, reference={reference}"
                )
            for metric in METRICS:
                delta = target[metric].to_numpy() - base[metric].to_numpy()
                bootstrap = np.mean(
                    delta[rng.integers(0, len(delta), size=(bootstrap_repeats, len(delta)))],
                    axis=1,
                )
                improved = delta > 0 if metric == "pearson_r" else delta < 0
                paired[budget][reference][metric] = {
                    "paired_delta": float(delta.mean()),
                    "ci95_low": float(np.quantile(bootstrap, 0.025)),
                    "ci95_high": float(np.quantile(bootstrap, 0.975)),
                    "n_subjects": int(len(delta)),
                    "improved_subjects": int(improved.sum()),
                }
        if "real_only" in available_groups:
            base = averaged[averaged.group == "real_only"].sort_values("subject_index")
            for group in sorted(available_groups - {"real_only"}):
                candidate = averaged[averaged.group == group].sort_values("subject_index")
                if len(base) != len(candidate) or not np.array_equal(
                    base.subject_index.to_numpy(),
                    candidate.subject_index.to_numpy(),
                ):
                    raise ValueError(f"Unmatched subjects for fraction={fraction}, group={group}")
                dataset_group_comparisons[budget][group] = {}
                for metric in METRICS:
                    delta = candidate[metric].to_numpy() - base[metric].to_numpy()
                    bootstrap = np.mean(
                        delta[
                            rng.integers(
                                0,
                                len(delta),
                                size=(bootstrap_repeats, len(delta)),
                            )
                        ],
                        axis=1,
                    )
                    improved = delta > 0 if metric == "pearson_r" else delta < 0
                    dataset_group_comparisons[budget][group][metric] = {
                        "paired_delta_vs_real_only": float(delta.mean()),
                        "ci95_low": float(np.quantile(bootstrap, 0.025)),
                        "ci95_high": float(np.quantile(bootstrap, 0.975)),
                        "n_subjects": int(len(delta)),
                        "improved_subjects": int(improved.sum()),
                    }
    summary = {
        "dataset": "PPG-DaLiA",
        "task": "3-cycle to 3-cycle waveform forecasting",
        "protocol": "subject-disjoint, explicit real-data budgets, validation-selected augmentation",
        "groups_by_real_fraction": overall,
    }
    (metrics_dir / "overall_summary.json").write_text(json.dumps(summary, indent=2))
    (metrics_dir / "paired_comparison.json").write_text(
        json.dumps(
            {
                "bootstrap_unit": "held-out subject after averaging configured model seeds",
                "target_group": "quality-filtered real_plus_synthetic",
                "comparisons_by_real_fraction": paired,
            },
            indent=2,
        )
    )
    (metrics_dir / "dataset_group_comparison.json").write_text(
        json.dumps(
            {
                "reference_group": "real_only",
                "bootstrap_unit": "held-out subject after averaging configured model seeds",
                "comparisons_by_real_fraction": dataset_group_comparisons,
            },
            indent=2,
        )
    )
    print(json.dumps({"subject_seed_rows": len(subject), "paired": paired}, indent=2))


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate saved downstream forecasting metrics")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol", choices=("fixed_budget", "augmentation"), default="fixed_budget"
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=10_000)
    args = parser.parse_args()
    if args.protocol == "augmentation":
        summarize_augmentation(args.run_dir, args.bootstrap_repeats)
    else:
        frame = pd.read_csv(args.run_dir / "metrics/per_window.csv")
        print(json.dumps(summarize_fixed_budget(frame, args.run_dir), indent=2))


if __name__ == "__main__":
    main()
