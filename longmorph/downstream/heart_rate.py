#!/usr/bin/env python3
"""PPG-DaLiA 8-second wrist-PPG -> ECG-HR downstream protocol.

The builder uses official per-window PPG-DaLiA labels for real data and never
uses test subjects for normalization or training.  Synthetic labels are named
``generator_side_rhythm_target`` and are computed only from generated cycle
durations; no peak detection is applied to rendered synthetic waveforms.
"""
from __future__ import annotations
import argparse, json, pickle, random
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import resample

from .models import HeartRateRegressor

EXPECTED = {
    "train": ["S14", "S15", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"],
    "validation": ["S12", "S13"],
    "test": ["S1", "S10", "S11"],
}
FS_REAL = 64.0
WINDOW_SEC = 8.0
SHIFT_SEC = 2.0
WINDOW_SAMPLES = 512
SHIFT_SAMPLES = 128
GEN_FS_DEFAULT = 500.0


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def load_split(path: Path) -> dict[str, list[str]]:
    raw = json.loads(path.read_text())
    row = next((x for x in raw if int(x.get("fold", 1)) == 1), raw[0] if raw else None)
    if row is None:
        raise ValueError(f"empty split file: {path}")
    # The existing split stores integer indices in dictionary order: S1, S10,
    # S11, ..., S15, S2, ..., S9.
    names = sorted([f"S{i}" for i in range(1, 16)])
    out = {k: [names[int(i)] for i in row[k]] for k in ("train", "validation", "test")}
    for k, expected in EXPECTED.items():
        if out[k] != expected:
            raise ValueError(f"fold01 {k} mismatch: got {out[k]}, expected {expected}")
    if (
        set(out["train"]) & set(out["validation"])
        or set(out["train"]) & set(out["test"])
        or set(out["validation"]) & set(out["test"])
    ):
        raise ValueError("subject split overlap")
    return out


def robust_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32).reshape(-1)
    med = float(np.median(x))
    q25, q75 = np.percentile(x, [25, 75])
    scale = (float(q75) - float(q25)) / 1.349
    return ((x - med) / (scale + 1e-6)).astype(np.float32)


def load_raw_subject(raw_root: Path, sid: str) -> tuple[np.ndarray, np.ndarray]:
    path = raw_root / "PPG_FieldStudy" / sid / f"{sid}.pkl"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("rb") as f:
        obj = pickle.load(f, encoding="latin1")
    bvp = np.asarray(obj["signal"]["wrist"]["BVP"], np.float32).reshape(-1)
    label = np.asarray(obj["label"], np.float32).reshape(-1)
    return bvp, label


def official_windows(
    bvp: np.ndarray, labels: np.ndarray, sid: str, max_windows: int = 0
) -> dict[str, np.ndarray]:
    # PPG-DaLiA stores the official ECG-derived HR sequence at the 2-second
    # label cadence.  Window i starts at i*2 seconds and uses label[i], which
    # is the paper/dataset 8-second target alignment (not a BVP-derived label).
    starts = np.arange(0, len(bvp) - WINDOW_SAMPLES + 1, SHIFT_SAMPLES, dtype=np.int64)
    starts = starts[starts // SHIFT_SAMPLES < len(labels)]
    if max_windows > 0:
        starts = starts[:max_windows]
    x = (
        np.stack([robust_normalize(bvp[int(s) : int(s) + WINDOW_SAMPLES]) for s in starts]).astype(
            np.float32
        )
        if len(starts)
        else np.zeros((0, WINDOW_SAMPLES), np.float32)
    )
    y = np.asarray([labels[int(s // SHIFT_SAMPLES)] for s in starts], np.float32)
    return {
        "x": x[:, None, :],
        "y": y,
        "subject": np.asarray([sid] * len(starts)),
        "start_sample": starts,
    }


def concat_rows(rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not rows:
        return {
            "x": np.zeros((0, 1, WINDOW_SAMPLES), np.float32),
            "y": np.zeros((0,), np.float32),
            "subject": np.asarray([], dtype="U4"),
            "start_sample": np.zeros((0,), np.int64),
        }
    return {k: np.concatenate([r[k] for r in rows], axis=0) for k in rows[0]}


def build_real(
    raw_root: Path,
    split: dict[str, list[str]],
    max_windows: int = 0,
    subject_limit: int = 0,
    smoke_subjects: list[str] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    selected = set(smoke_subjects or [])
    out = {}
    for part, subjects in split.items():
        if smoke_subjects is not None:
            subjects = [s for s in subjects if s in selected]
        elif subject_limit > 0:
            subjects = subjects[:subject_limit]
        rows = []
        for sid in subjects:
            bvp, labels = load_raw_subject(raw_root, sid)
            rows.append(official_windows(bvp, labels, sid, max_windows))
        out[part] = concat_rows(rows)
    return out


def synthetic_windows(
    cycle_npz: Path,
    duration_npz: Path,
    anchor_npz: Path,
    context_cycles: int = 3,
    generation_sample_rate: float = GEN_FS_DEFAULT,
    max_windows: int = 0,
) -> tuple[dict[str, np.ndarray], dict]:
    cz = np.load(cycle_npz, allow_pickle=True)
    dz = np.load(duration_npz, allow_pickle=True)
    az = np.load(anchor_npz, allow_pickle=True)
    cycles = np.asarray(cz["cycles"], np.float32)
    durations = np.asarray(dz["durations"], np.float32)
    if (
        cycles.ndim != 3
        or durations.ndim != 2
        or cycles.shape[:2] != durations.shape
        or cycles.shape[2] != 256
    ):
        raise ValueError("synthetic cycles/durations shapes are incompatible")
    if "anchor_index" not in cz or "subject_id" not in az:
        raise ValueError(
            "train-only synthetic artifact must retain anchor_index and anchor subject_id metadata"
        )
    anchor_ids = np.asarray(az["subject_id"]).astype(str)
    anchor_index = np.asarray(cz["anchor_index"], int)

    def canonical_sid(value: str) -> str:
        value = str(value)
        if value.startswith("subject_"):
            try:
                idx = int(value.split("_", 1)[1])
                names = sorted([f"S{i}" for i in range(1, 16)])
                if 0 <= idx < len(names):
                    return names[idx]
            except ValueError:
                pass
        return value

    source_subjects = [canonical_sid(anchor_ids[i]) for i in anchor_index]
    rows_x, rows_y, rows_s, rows_start = [], [], [], []
    for seq, dur, sid in zip(cycles, durations, source_subjects):
        seq = seq[context_cycles:]
        dur = dur[context_cycles:]
        rr_ms = dur.astype(np.float64) / float(generation_sample_rate) * 1000.0
        parts = [
            resample(
                np.asarray(c, np.float32),
                max(8, int(round(float(d) / generation_sample_rate * FS_REAL))),
            ).astype(np.float32)
            for c, d in zip(seq, dur)
        ]
        wave = np.concatenate(parts) if parts else np.zeros((0,), np.float32)
        cycle_starts = np.cumsum(np.r_[0.0, dur[:-1] / generation_sample_rate * FS_REAL])
        cycle_ends = cycle_starts + dur / generation_sample_rate * FS_REAL
        starts = np.arange(0, max(0, len(wave) - WINDOW_SAMPLES + 1), SHIFT_SAMPLES, dtype=np.int64)
        if max_windows > 0:
            starts = starts[:max_windows]
        for st in starts:
            en = st + WINDOW_SAMPLES
            centers = (cycle_starts + cycle_ends) / 2.0
            keep = (centers >= st) & (centers < en)
            if not np.any(keep):
                continue
            target = 60000.0 / max(float(np.mean(rr_ms[keep])), 1e-6)
            rows_x.append(robust_normalize(wave[int(st) : int(en)]))
            rows_y.append(target)
            rows_s.append(sid)
            rows_start.append(int(st))
    data = {
        "x": (
            np.asarray(rows_x, np.float32)[:, None, :]
            if rows_x
            else np.zeros((0, 1, WINDOW_SAMPLES), np.float32)
        ),
        "y": np.asarray(rows_y, np.float32),
        "subject": np.asarray(rows_s),
        "start_sample": np.asarray(rows_start, np.int64),
    }
    meta = {
        "label_definition": "generator_side_rhythm_target = 60000 / mean(RR_ms) over generated cycles whose centers fall in the 8-second window; RR comes only from generated cycle-duration metadata; no waveform peak detection",
        "generation_sample_rate_hz": float(generation_sample_rate),
        "context_cycles_excluded": int(context_cycles),
        "source_subjects": sorted(set(rows_s)),
        "num_sequences": int(len(cycles)),
        "num_windows": int(len(rows_x)),
    }
    return data, meta


def save_data(
    out: Path, real: dict, synthetic: dict, split: dict, audit: dict, synth_meta: dict
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for part, d in real.items():
        np.savez_compressed(out / f"real_{part}.npz", **d)
    np.savez_compressed(out / "synthetic_train.npz", **synthetic)
    (out / "synthetic_dataset_summary.json").write_text(json.dumps(synth_meta, indent=2))
    (out / "data_audit.json").write_text(json.dumps(audit, indent=2))
    (out / "subject_split_resolved.json").write_text(json.dumps(split, indent=2))


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    return (
        float(np.corrcoef(a, b)[0, 1])
        if len(a) > 1 and np.std(a) > 0 and np.std(b) > 0
        else float("nan")
    )


def train(
    dataset_dir: Path,
    output_dir: Path,
    variant: str,
    seed: int = 7,
    device: str = "cuda",
    epochs: int = 100,
    patience: int = 12,
    batch_size: int = 128,
    lr: float = 1e-3,
) -> None:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")

    def load(name):
        z = np.load(dataset_dir / name, allow_pickle=True)
        return {k: z[k] for k in z.files}

    tr = load("real_train.npz")
    va = load("real_validation.npz")
    te = load("real_test.npz")
    sy = load("synthetic_train.npz")
    real_x, real_y = tr["x"], tr["y"]
    if variant in {"real_only_25pct", "real25_plus_longmorph"}:
        rng = np.random.default_rng(seed)
        ids = []
        for sid in np.unique(tr["subject"]):
            ii = np.flatnonzero(tr["subject"] == sid)
            ids.extend(
                rng.choice(ii, max(1, int(np.floor(0.25 * len(ii)))), replace=False).tolist()
            )
        ids = np.asarray(sorted(ids), int)
        real_x, real_y = real_x[ids], real_y[ids]
        np.savez_compressed(output_dir / "real25_selection_indices.npz", indices=ids)
    if variant in {"real_plus_longmorph_full", "real25_plus_longmorph"}:
        x, y = np.concatenate([real_x, sy["x"]]), np.concatenate([real_y, sy["y"]])
    elif variant == "synthetic_only":
        x, y = sy["x"], sy["y"]
    else:
        x, y = real_x, real_y
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "train_manifest.npz",
        n_real=np.asarray([len(real_x)]),
        n_synthetic=np.asarray(
            [
                (
                    len(sy["x"])
                    if variant
                    in {"real_plus_longmorph_full", "real25_plus_longmorph", "synthetic_only"}
                    else 0
                )
            ]
        ),
    )
    model = HeartRateRegressor().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.HuberLoss()
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(y[:, None]).float()),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(va["x"]).float(), torch.from_numpy(va["y"][:, None]).float()
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    best_state = None
    best = float("inf")
    stale = 0
    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        vals = []
        with torch.no_grad():
            for xb, yb in val_loader:
                vals.append(float(loss_fn(model(xb.to(dev)), yb.to(dev)).cpu()))
        v = float(np.mean(vals))
        hist.append(
            {"variant": variant, "epoch": ep, "train_huber": float(np.mean(losses)), "val_huber": v}
        )
        if v < best - 1e-6:
            best = v
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(te["x"]), batch_size):
            preds.append(
                model(torch.from_numpy(te["x"][i : i + batch_size]).float().to(dev))
                .cpu()
                .numpy()
                .reshape(-1)
            )
    pred = np.concatenate(preds) if preds else np.zeros((0,), np.float32)
    true = te["y"].astype(np.float32)
    subjects = te["subject"].astype(str)
    rows = []
    for sid in np.unique(subjects):
        m = subjects == sid
        err = pred[m] - true[m]
        rows.append(
            {
                "variant": variant,
                "subject": sid,
                "n_test": int(m.sum()),
                "mae": float(np.mean(np.abs(err))),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "pearson": pearson(pred[m], true[m]),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "per_subject.csv", index=False)
    pd.DataFrame(hist).to_csv(output_dir / "training_history.csv", index=False)
    np.savez_compressed(output_dir / "test_predictions.npz", pred=pred, true=true, subject=subjects)
    torch.save(model.state_dict(), output_dir / "hr_cnn.pt")
    pd.DataFrame(
        [
            {
                "variant": variant,
                "n_real_train": int(len(real_x)),
                "n_synthetic_train": int(
                    len(sy["x"])
                    if variant
                    in {"real_plus_longmorph_full", "real25_plus_longmorph", "synthetic_only"}
                    else 0
                ),
                "best_val_huber": best,
                "epochs": len(hist),
            }
        ]
    ).to_csv(output_dir / "run_summary.csv", index=False)


def aggregate(output_dir: Path, variants: list[str]) -> None:
    rows = []
    per = []
    hist = []
    for v in variants:
        root = output_dir / "variants" / v
        rs = pd.read_csv(root / "run_summary.csv").iloc[0].to_dict()
        ps = pd.read_csv(root / "per_subject.csv")
        hh = pd.read_csv(root / "training_history.csv")
        per.append(ps)
        hist.append(hh)
        pred = np.load(root / "test_predictions.npz", allow_pickle=True)
        e = pred["pred"] - pred["true"]
        rows.append(
            {
                "variant": v,
                "n_real_train": int(rs["n_real_train"]),
                "n_synthetic_train": int(rs["n_synthetic_train"]),
                "synthetic_real_ratio": float(rs["n_synthetic_train"])
                / max(1, float(rs["n_real_train"])),
                "test_mae": float(np.mean(np.abs(e))),
                "test_rmse": float(np.sqrt(np.mean(e**2))),
                "test_pearson": pearson(pred["pred"], pred["true"]),
                "macro_mae": float(ps.mae.mean()),
                "macro_rmse": float(ps.rmse.mean()),
                "macro_pearson": float(ps.pearson.mean()),
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "hr_downstream_summary.csv", index=False)
    pd.concat(per, ignore_index=True).to_csv(
        output_dir / "hr_downstream_per_subject.csv", index=False
    )
    pd.concat(hist, ignore_index=True).to_csv(output_dir / "training_history.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--raw-root", type=Path, required=True)
    b.add_argument("--split-json", type=Path, required=True)
    b.add_argument("--output-dir", type=Path, required=True)
    b.add_argument("--synthetic-cycles", type=Path, default=None)
    b.add_argument("--synthetic-durations", type=Path, default=None)
    b.add_argument("--synthetic-anchor-npz", type=Path, default=None)
    b.add_argument("--generation-sample-rate", type=float, default=500.0)
    b.add_argument("--max-windows-per-subject", type=int, default=0)
    b.add_argument("--smoke", action="store_true")
    t = sub.add_parser("train")
    t.add_argument("--dataset-dir", type=Path, required=True)
    t.add_argument("--output-dir", type=Path, required=True)
    t.add_argument(
        "--variant",
        required=True,
        choices=[
            "real_only_full",
            "real_plus_longmorph_full",
            "real_only_25pct",
            "real25_plus_longmorph",
            "synthetic_only",
        ],
    )
    t.add_argument("--seed", type=int, default=7)
    t.add_argument("--device", default="cuda")
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--patience", type=int, default=12)
    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-3)
    g = sub.add_parser("aggregate")
    g.add_argument("--output-dir", type=Path, required=True)
    g.add_argument(
        "--variants",
        default="real_only_full,real_plus_longmorph_full,real_only_25pct,real25_plus_longmorph",
    )
    a = ap.parse_args()
    if a.cmd == "build":
        split = load_split(a.split_json)
        smoke_subjects = ["S14", "S12", "S1"] if a.smoke else None
        real = build_real(
            a.raw_root, split, a.max_windows_per_subject, smoke_subjects=smoke_subjects
        )
        synth_meta = {
            "status": "missing_train_only_cycles_durations",
            "label_definition": "not available until train-only cycles+durations are supplied",
        }
        synthetic = {
            "x": np.zeros((0, 1, WINDOW_SAMPLES), np.float32),
            "y": np.zeros((0,), np.float32),
            "subject": np.asarray([]),
            "start_sample": np.zeros((0,), np.int64),
        }
        if a.synthetic_cycles or a.synthetic_durations or a.synthetic_anchor_npz:
            if not (a.synthetic_cycles and a.synthetic_durations and a.synthetic_anchor_npz):
                raise ValueError(
                    "synthetic cycles, durations and anchor NPZ must be supplied together"
                )
            synthetic, synth_meta = synthetic_windows(
                a.synthetic_cycles,
                a.synthetic_durations,
                a.synthetic_anchor_npz,
                generation_sample_rate=a.generation_sample_rate,
                max_windows=a.max_windows_per_subject,
            )
        train = set(split["train"])
        val = set(split["validation"])
        test = set(split["test"])
        src = set(map(str, synth_meta.get("source_subjects", [])))
        audit = {
            "split_json": str(a.split_json),
            "train_subjects": sorted(train),
            "validation_subjects": sorted(val),
            "test_subjects": sorted(test),
            "train_validation_subject_overlap": len(train & val),
            "train_test_subject_overlap": len(train & test),
            "validation_test_subject_overlap": len(val & test),
            "synthetic_source_subjects": sorted(src),
            "synthetic_source_subject_intersection_train": sorted(src & train),
            "synthetic_source_subjects_not_in_train": sorted(src - train),
            "synthetic_source_subject_intersection_validation": sorted(src & val),
            "synthetic_source_subject_intersection_test": sorted(src & test),
            "test_labels_used_during_training": False,
            "test_statistics_used_during_normalization": False,
            "normalization": "per_window_robust_median_IQR_over_1.349_plus_eps",
            "real_target": "official PPG-DaLiA label[i] aligned to window start i*2 seconds",
            "synthetic_target_source": "generator_side_duration_metadata",
            "smoke": bool(a.smoke),
        }
        save_data(a.output_dir, real, synthetic, split, audit, synth_meta)
        print(
            json.dumps(
                {
                    "output": str(a.output_dir),
                    "real_windows": {k: int(len(v["x"])) for k, v in real.items()},
                    "synthetic_windows": int(len(synthetic["x"])),
                    "audit": audit,
                },
                indent=2,
            )
        )
    elif a.cmd == "train":
        train(
            a.dataset_dir,
            a.output_dir,
            a.variant,
            a.seed,
            a.device,
            a.epochs,
            a.patience,
            a.batch_size,
            a.lr,
        )
    else:
        aggregate(a.output_dir, [x for x in a.variants.split(",") if x])


if __name__ == "__main__":
    main()
