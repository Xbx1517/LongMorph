#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, pickle
from pathlib import Path
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, resample


def anon(subject: str, salt: str) -> str:
    return hashlib.sha256((salt + subject).encode()).hexdigest()[:16]


def load_subject(path: Path):
    obj = pickle.load(open(path, "rb"), encoding="latin1")
    wrist = obj["signal"]["wrist"]
    bvp = np.asarray(wrist["BVP"], np.float32).reshape(-1)
    activity = np.asarray(obj.get("activity", np.zeros(len(bvp) // 64)), np.float32).reshape(-1)
    ecg_hr = np.asarray(obj.get("label", []), np.float32).reshape(-1)
    return bvp, activity, ecg_hr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--salt", default="longmorph-ppg-dalia-split-v1")
    ap.add_argument("--sample-rate", type=float, default=64)
    ap.add_argument("--cycle-len", type=int, default=256)
    ap.add_argument("--min-prominence", type=float, default=0.08)
    ap.add_argument(
        "--subject-dir",
        default="",
        help="Optional anonymized continuous-wave files for fold-specific LongMorph training",
    )
    args = ap.parse_args()
    files = sorted(Path(args.raw_dir).glob("S*/S*.pkl")) or sorted(
        Path(args.raw_dir).glob("S*.pkl")
    )
    if len(files) < 15:
        raise FileNotFoundError(
            f"Expected 15 PPG-DaLiA subject pickle files, found {len(files)} in {args.raw_dir}"
        )
    all_cycles = []
    all_rr = []
    all_subject = []
    all_start = []
    all_activity = []
    all_ecg_hr = []
    all_quality = []
    b, a = butter(3, [0.5 / (args.sample_rate / 2), 8 / (args.sample_rate / 2)], btype="band")
    subject_dir = (
        Path(args.subject_dir)
        if args.subject_dir
        else Path(args.output).with_suffix("").parent / "ppg_dalia_subjects"
    )
    subject_dir.mkdir(parents=True, exist_ok=True)
    for sid, path in enumerate(files):
        x, activity, ecg_hr = load_subject(path)
        x = filtfilt(b, a, x).astype(np.float32)
        lo, hi = np.percentile(x, [1, 99])
        x = np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)
        np.savez_compressed(
            subject_dir / f"subject_{sid:02d}_ppg.npz",
            ppg=x.astype(np.float32),
            sample_rate=np.float32(args.sample_rate),
            subject_index=np.int16(sid),
        )
        troughs, _ = find_peaks(
            -x, distance=max(1, round(0.32 * args.sample_rate)), prominence=args.min_prominence
        )
        for s, e in zip(troughs[:-1], troughs[1:]):
            rr = (e - s) / args.sample_rate * 1000
            if not 300 <= rr <= 1800 or e - s < 12:
                continue
            c = x[s:e]
            c = (c - c.min()) / (c.max() - c.min() + 1e-8)
            all_cycles.append(resample(c, args.cycle_len).astype(np.float32))
            all_rr.append(rr)
            all_subject.append(sid)
            all_start.append(s)
            ai = min(len(activity) - 1, round((s / args.sample_rate) * 4)) if len(activity) else 0
            all_activity.append(float(activity[ai]) if len(activity) else 0.0)
            hi_idx = (
                min(len(ecg_hr) - 1, round((s / max(1, len(x) - 1)) * max(0, len(ecg_hr) - 1)))
                if len(ecg_hr)
                else 0
            )
            href = float(ecg_hr[hi_idx]) if len(ecg_hr) else np.nan
            all_ecg_hr.append(href)
            all_quality.append(bool(np.isfinite(href) and abs(60000 / rr - href) <= 40))
    cycles = np.asarray(all_cycles)
    subjects = np.asarray(all_subject)
    starts = np.asarray(all_start)
    rr = np.asarray(all_rr, np.float32)
    acts = np.asarray(all_activity, np.float32)
    refs = np.asarray(all_ecg_hr, np.float32)
    quality = np.asarray(all_quality, bool)
    windows = []
    durations = []
    win_subject = []
    win_start = []
    win_activity = []
    win_ref = []
    win_quality = []
    for sid in np.unique(subjects):
        idx = np.flatnonzero(subjects == sid)
        for j in range(0, len(idx) - 5, 3):
            ii = idx[j : j + 6]
            if np.any(np.diff(starts[ii]) > round(2 * args.sample_rate)):
                continue
            windows.append(cycles[ii])
            durations.append(rr[ii])
            win_subject.append(sid)
            win_start.append(starts[ii[0]])
            win_activity.append(float(np.median(acts[ii])))
            win_ref.append(float(np.nanmedian(refs[ii])))
            win_quality.append(float(np.mean(quality[ii])))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    subject_hashes = np.asarray(
        [anon(p.parent.name if p.parent.name.startswith("S") else p.stem, args.salt) for p in files]
    )
    np.savez_compressed(
        out,
        cycles=np.asarray(windows, np.float32),
        durations_ms=np.asarray(durations, np.float32),
        subject_index=np.asarray(win_subject, np.int16),
        subject_hashes=subject_hashes,
        source_start_sample=np.asarray(win_start, np.int64),
        activity=np.asarray(win_activity, np.float32),
        ecg_reference_hr_bpm=np.asarray(win_ref, np.float32),
        quality_fraction=np.asarray(win_quality, np.float32),
        sample_rate=np.float32(args.sample_rate),
    )
    print(
        {
            "subjects": len(files),
            "cycles": len(cycles),
            "six_cycle_windows": len(windows),
            "output": str(out),
            "subject_dir": str(subject_dir),
        }
    )


if __name__ == "__main__":
    main()
