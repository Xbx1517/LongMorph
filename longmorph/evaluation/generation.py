"""Evaluate generated future cycles against continuous paired references."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from . import metrics
from longmorph.selection.backend import sha


PRIMARY_FIELDS = (
    "shape_mmd",
    "hr_mae_bpm",
    "mean_future_internal_seam",
    "psd_log_mse",
    "cycle_acf_mse_lag20",
    "cycle_acf_mse_lag63",
    "hr_w1_bpm",
)


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def identifiers(data):
    cases = np.asarray(data["case_id"]).astype(str)
    subjects = np.asarray(data["subject_id"]).astype(str)
    if cases.ndim != 1 or subjects.shape != cases.shape or not len(cases):
        raise ValueError("case_id and canonical subject_id must be nonempty vectors")
    if len(set(cases.tolist())) != len(cases) or np.any(cases == "") or np.any(subjects == ""):
        raise ValueError("Empty or duplicate case IDs, or empty subject IDs")
    if not np.all(np.asarray(data.get("sample_rate", 500.0)) == 500.0):
        raise ValueError("Current online metrics require a 500Hz native-duration clock")
    return cases, subjects


def csv_write(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(clean(rows))


def summary(rows, fields):
    result = {}
    for field in fields:
        values = [
            float(row[field])
            for row in rows
            if row.get(field) is not None and np.isfinite(row[field])
        ]
        result[field + "_mean"] = float(np.mean(values)) if values else None
        result[field + "_sample_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        result[field + "_finite_seed_count"] = len(values)
    return result


def evaluate(
    generated_paths,
    references,
    output_dir,
    seeds,
    horizon=None,
    requested_slots=None,
    common_reference_horizon=None,
    method="longmorph",
):
    if len(generated_paths) != len(seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Provide exactly one generated file per distinct declared seed")
    truth = arrays(references)
    case_ids, subjects = identifiers(truth)
    real = np.asarray(truth["future_cycles"], np.float32)
    real_durations = np.asarray(truth["future_durations"], np.float32)
    if (
        real.ndim != 3
        or real.shape[0] != len(case_ids)
        or real.shape[2] != 256
        or real_durations.shape != real.shape[:2]
    ):
        raise ValueError(
            "References require future_cycles[N,Hmax,256] and future_durations[N,Hmax]"
        )
    available = np.asarray(
        truth.get("available_future_cycles", np.full(len(case_ids), real.shape[1]))
    )
    if (
        available.shape != (len(case_ids),)
        or np.any(available != available.astype(int))
        or np.any(available < 0)
        or np.any(available > real.shape[1])
    ):
        raise ValueError("Invalid available_future_cycles")
    available = available.astype(int)
    expected_valid = np.arange(real.shape[1])[None, :] < available[:, None]
    if "future_valid" in truth and not np.array_equal(
        np.asarray(truth["future_valid"], bool), expected_valid
    ):
        raise ValueError(
            "Reference validity must be a contiguous prefix; padding never enters metrics"
        )
    slots = len(case_ids) if requested_slots is None else int(requested_slots)
    if slots < len(case_ids):
        raise ValueError("requested_slots cannot be smaller than the context cohort")
    actuals = []
    for seed, path in zip(seeds, generated_paths):
        payload = arrays(path)
        ids, sub = identifiers(payload)
        if set(ids.tolist()) != set(case_ids.tolist()):
            raise ValueError(
                "Generated/reference case sets differ; generation failures must not become a selected subset"
            )
        lookup = {case: i for i, case in enumerate(ids)}
        indices = [lookup[case] for case in case_ids]
        if not np.array_equal(sub[indices], subjects):
            raise ValueError("Generated/reference canonical subject identities differ")
        fake = np.asarray(payload["future_cycles"], np.float32)[indices]
        duration = np.asarray(payload["durations"], np.float32)[indices]
        if (
            fake.ndim != 3
            or fake.shape[0] != len(case_ids)
            or fake.shape[2] != 256
            or duration.shape != fake.shape[:2]
        ):
            raise ValueError(
                "Generated NPZ requires future_cycles[N,Hmax,256] and durations[N,Hmax]"
            )
        if not np.isfinite(fake).all() or not np.isfinite(duration).all() or np.any(duration <= 0):
            raise ValueError(
                "Invalid generated values; generation failure is not reference unavailability"
            )
        actuals.append(dict(seed=seed, path=Path(path), fake=fake, durations=duration))
    if not actuals:
        raise ValueError("At least one generated file is required")
    if len({row["fake"].shape[1] for row in actuals}) != 1:
        raise ValueError(
            "All seed files must contain the same maximum supplied horizon; IBI peak scope must match"
        )
    h = actuals[0]["fake"].shape[1] if horizon is None else int(horizon)
    if h < 2 or h > real.shape[1] or any(h > row["fake"].shape[1] for row in actuals):
        raise ValueError(
            "Metric horizon must be >=2 and fit every supplied generated/reference array"
        )
    cohorts = [("available_reference", np.flatnonzero(available >= h))]
    if common_reference_horizon is not None:
        common = int(common_reference_horizon)
        if common < h:
            raise ValueError("common_reference_horizon must cover the requested horizon")
        cohorts.append((f"common_H{common}_reference", np.flatnonzero(available >= common)))
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    pooled, ibi_rows, hr_rows, missing = [], [], [], []
    for cohort, eligible in cohorts:
        for i in range(len(case_ids)):
            if i not in eligible:
                missing.append(
                    dict(
                        cohort=cohort,
                        case_id=case_ids[i],
                        subject_id=subjects[i],
                        horizon=h,
                        available_future_cycles=int(available[i]),
                        reason="REFERENCE_HORIZON_UNAVAILABLE",
                        generation_failed=False,
                    )
                )
        if not len(eligible):
            continue
        r, rd = real[eligible, :h], real_durations[eligible, :h]
        if not np.isfinite(r).all() or not np.isfinite(rd).all() or np.any(rd <= 0):
            raise ValueError("Invalid reference values within the declared available cohort")
        for actual in actuals:
            seed = actual["seed"]
            f, fd = actual["fake"][eligible, :h], actual["durations"][eligible, :h]
            registry = {}
            values = metrics.primary(
                r, f, rd, fd, case_ids[eligible].tolist(), method, output / "scratch", registry
            )
            pooled.append(
                dict(
                    method=method,
                    seed=seed,
                    horizon=h,
                    cohort=cohort,
                    requested_slots=slots,
                    context_available=len(case_ids),
                    reference_eligible=len(eligible),
                    reference_missing=slots - len(eligible),
                    generation_failed=0,
                    n_subjects=len(set(subjects[eligible].tolist())),
                    **values,
                )
            )
            for row in registry.values():
                hr_rows.append(dict(seed=seed, horizon=h, cohort=cohort, **row))
            for i in eligible:
                # IBI uses a single detection on the maximum paired-available
                # trajectory supplied, then an original-complete-interval view.
                maximum = min(actual["fake"].shape[1], int(available[i]))
                rmax, rdmax = real[i, :maximum], real_durations[i, :maximum]
                if (
                    not np.isfinite(rmax).all()
                    or not np.isfinite(rdmax).all()
                    or np.any(rdmax <= 0)
                ):
                    raise ValueError(
                        "Invalid reference inside the maximum paired-available trajectory"
                    )
                real_record = metrics.derive(rmax, rdmax)
                fake_record = metrics.derive(
                    actual["fake"][i, :maximum], actual["durations"][i, :maximum]
                )
                rv, _, _ = metrics.view(real_record, h)
                fv, _, _ = metrics.view(fake_record, h)
                row = dict(
                    method=method,
                    seed=seed,
                    horizon=h,
                    cohort=cohort,
                    case_id=case_ids[i],
                    subject_id=subjects[i],
                    analyzed_max_horizon=maximum,
                    peak_scope="one maximum-paired-trajectory detection; complete original interval prefix",
                )
                for kind in ("raw", "filtered"):
                    ra, fa = rv["IBI", kind], fv["IBI", kind]
                    _, w1 = metrics.distance(ra, fa)
                    row["IBI_" + kind + "_W1_ms"] = w1
                    row["n_real_" + kind + "_intervals"] = int(np.isfinite(ra).sum())
                    row["n_generated_" + kind + "_intervals"] = int(np.isfinite(fa).sum())
                ibi_rows.append(row)
    per_seed_ibi = []
    for cohort, _ in cohorts:
        records = [row for row in ibi_rows if row["cohort"] == cohort]
        for kind in ("raw", "filtered"):
            field = "IBI_" + kind + "_W1_ms"
            by_case = defaultdict(dict)
            for row in records:
                by_case[row["case_id"]][row["seed"]] = row
            common_cases = {
                case
                for case, values in by_case.items()
                if set(values) == set(seeds)
                and all(np.isfinite(values[seed][field]) for seed in seeds)
            }
            for seed in seeds:
                by_subject = defaultdict(list)
                for row in records:
                    if row["seed"] == seed and row["case_id"] in common_cases:
                        by_subject[row["subject_id"]].append(row[field])
                estimate = (
                    float(metrics.mean([metrics.mean(values) for values in by_subject.values()]))
                    if by_subject
                    else None
                )
                per_seed_ibi.append(
                    dict(
                        method=method,
                        seed=seed,
                        horizon=h,
                        cohort=cohort,
                        interval_set=kind,
                        IBI_W1_ms=estimate,
                        n_common_complete_seed_anchors=len(common_cases),
                        n_subjects=len(by_subject),
                        aggregation="Within-subject anchor mean, equal mean across subjects; same finite anchors in all declared seeds",
                    )
                )
    summaries = [
        dict(
            method=method,
            horizon=h,
            cohort=cohort,
            declared_seed_count=len(seeds),
            **summary([row for row in pooled if row["cohort"] == cohort], PRIMARY_FIELDS),
        )
        for cohort, _ in cohorts
    ]
    ibi_summaries = [
        dict(
            method=method,
            horizon=h,
            cohort=cohort,
            interval_set=kind,
            declared_seed_count=len(seeds),
            **summary(
                [
                    row
                    for row in per_seed_ibi
                    if row["cohort"] == cohort and row["interval_set"] == kind
                ],
                ("IBI_W1_ms",),
            ),
        )
        for cohort, _ in cohorts
        for kind in ("raw", "filtered")
    ]
    for name, rows in (
        ("per_seed_metrics.csv", pooled),
        ("metrics_summary.csv", summaries),
        ("per_anchor_IBI.csv", ibi_rows),
        ("per_seed_IBI.csv", per_seed_ibi),
        ("IBI_summary.csv", ibi_summaries),
        ("native_HR_validity.csv", hr_rows),
        ("reference_missing.csv", missing),
    ):
        csv_write(output / name, rows)
    result = dict(
        status="COMPLETE_SAVED_TRAJECTORY_EVALUATION",
        method=method,
        horizon=h,
        seeds=list(seeds),
        sources=[
            dict(path=str(p), sha256=sha(p)) for p in (Path(__file__), Path(metrics.__file__))
        ],
        inputs=[
            dict(path=str(Path(p).resolve()), sha256=sha(p)) for p in [references, *generated_paths]
        ],
        definitions=dict(
            shape_mmd="Pooled all-cycle biased RBF MMD squared, diagonal terms, positive-pdist median bandwidth; disk-backed exact distances and bounded kernel sums",
            shape_reduction="Per declared seed and reference cohort, never an average of per-anchor MMD",
            hr="Validated peak-interval HR on native-duration reconstruction; per-horizon detection, paired MAE and separate-valid-distribution W1",
            seam="Original native500Hz Seam-v2 value+d1+d2 index jumps at future-internal 3-cycle block joins",
            cycle_acf="Cycle-index phase-zscored ACF at lag20 and lag63, reported separately",
            psd="Original separate-cohort median-length rectangular resampling, Welch and pooled log-PSD MSE over0.5..15Hz",
            IBI="Exact derive/view/distance: one maximum paired-available native-waveform detection; raw and300..2000ms plus15percent-legal-median filtered intervals",
            IBI_aggregation="Same finite anchor support across all declared seeds; within-subject anchor mean then equal subject mean per seed",
            summary="Equal means across declared generation seeds, sample SD ddof1; finite seed counts explicit; no IIDcycle/bootstrap/significance claim",
            unavailable="Missing reference slots/short prefixes retained; no padding, replacement, or generation-failure subset selection",
        ),
        pooled=pooled,
        primary_summary=summaries,
        IBI_summary=ibi_summaries,
        model_loads=0,
        generation=False,
        no_fitting=True,
    )
    with (output / "evaluation.json").open("x") as stream:
        json.dump(clean(result), stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generated",
        nargs="+",
        required=True,
        type=Path,
        help="One generated H*.npz per declared seed",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        required=True,
        type=int,
        help="Seed labels in the same order as --generated",
    )
    parser.add_argument(
        "--references",
        required=True,
        type=Path,
        help="Paired real future_cycles/future_durations/case_id/subject_id NPZ",
    )
    parser.add_argument(
        "--horizon", type=int, help="Default: first generated file horizon; minimum2"
    )
    parser.add_argument(
        "--requested-slots",
        type=int,
        help="Include unavailable-context requested slots in denominators",
    )
    parser.add_argument(
        "--common-reference-horizon",
        type=int,
        help="Also evaluate the cohort with this much reference availability, e.g.256",
    )
    parser.add_argument("--method-name", default="longmorph")
    parser.add_argument("--output-dir", required=True, type=Path, help="Fresh output directory")
    args = parser.parse_args()
    result = evaluate(
        args.generated,
        args.references,
        args.output_dir,
        args.seeds,
        args.horizon,
        args.requested_slots,
        args.common_reference_horizon,
        args.method_name,
    )
    print(
        json.dumps(
            dict(
                status=result["status"],
                horizon=result["horizon"],
                per_seed_cohort_rows=len(result["pooled"]),
            )
        )
    )


if __name__ == "__main__":
    main()
