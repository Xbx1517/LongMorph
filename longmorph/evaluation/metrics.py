"""Pulse morphology, native-time physiology, and distribution metrics."""

from __future__ import annotations
from pathlib import Path
import tempfile
from longmorph.data.features import (
    SHAPE_FEATURE_NAMES,
    compute_ppg_shape_features,
    compute_psd_tail_ratio,
    minmax_1d,
)
import numpy as np
from scipy.signal import find_peaks, resample, welch
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import rbf_kernel


HRV_PEAK_MIN_DISTANCE_SEC = 0.35


HRV_PEAK_PROMINENCE = 0.25


HRV_PEAK_PROMINENCE_STD_SCALE = 0.12


HRV_RR_MIN_MS = 300.0


HRV_RR_MAX_MS = 2000.0


HRV_RR_REL_FILTER = 0.15


def _validated_hrv_rr_intervals(x: np.ndarray, sample_rate: float) -> np.ndarray:
    prominence = max(HRV_PEAK_PROMINENCE, HRV_PEAK_PROMINENCE_STD_SCALE * float(np.std(x)))
    distance = max(1, int(HRV_PEAK_MIN_DISTANCE_SEC * sample_rate))
    peaks, _ = find_peaks(x, prominence=prominence, distance=distance)
    intervals_ms = np.diff(peaks).astype(np.float32) / float(sample_rate) * 1000.0
    intervals_ms = intervals_ms[np.isfinite(intervals_ms)]
    intervals_ms = intervals_ms[(intervals_ms >= HRV_RR_MIN_MS) & (intervals_ms <= HRV_RR_MAX_MS)]
    if intervals_ms.size == 0:
        return intervals_ms
    median_rr = float(np.median(intervals_ms))
    return intervals_ms[np.abs(intervals_ms - median_rr) <= HRV_RR_REL_FILTER * median_rr]


def hrv_features(arr: np.ndarray, sample_rate: float) -> np.ndarray:
    flat = arr.reshape(arr.shape[0], arr.shape[1], -1)[:, :, 0]
    rows = []
    for x in flat:
        intervals_ms = _validated_hrv_rr_intervals(x, sample_rate)
        valid = float(intervals_ms.size > 0)
        if intervals_ms.size == 0:
            rows.append([0.0, 0.0, 0.0, 0.0, 0.0, valid])
            continue
        mean_rr = float(np.mean(intervals_ms))
        sdnn = float(np.std(intervals_ms))
        if intervals_ms.size > 1:
            diff_rr = np.diff(intervals_ms)
            rmssd = float(np.sqrt(np.mean(diff_rr**2)))
            pnn50 = float(np.mean(np.abs(diff_rr) > 50.0))
        else:
            rmssd = 0.0
            pnn50 = 0.0
        hr_bpm = float(60000.0 / max(mean_rr, 1e-6))
        rows.append([mean_rr, sdnn, rmssd, pnn50, hr_bpm, valid])
    return np.asarray(rows, dtype=np.float32)


def psd_metrics(real: np.ndarray, fake: np.ndarray, sample_rate: float) -> dict[str, float]:
    grid = np.linspace(0.5, 15, 291)

    def rows(x):
        ans = []
        for v in x:
            fr, p = welch(v, fs=sample_rate, nperseg=min(len(v), 2048))
            ans.append(np.interp(grid, fr, p))
        return np.asarray(ans)

    r, f = rows(real), rows(fake)
    rm, fm = r.mean(0), f.mean(0)
    out = {"psd_log_mse_0p5_15hz": float(np.mean((np.log(rm + 1e-12) - np.log(fm + 1e-12)) ** 2))}
    for lo, hi, name in ((0.5, 4, "0p5_4hz"), (4, 8, "4_8hz"), (8, 15, "8_15hz")):
        m = (grid >= lo) & (grid < hi)
        rr = np.trapezoid(rm[m], grid[m])
        ff = np.trapezoid(fm[m], grid[m])
        out[f"psd_band_error_{name}"] = float(abs(np.log((ff + 1e-12) / (rr + 1e-12))))
    return out


def rbf_mmd(x: np.ndarray, y: np.ndarray) -> float:
    x = x.reshape(len(x), -1)
    y = y.reshape(len(y), -1)
    z = np.vstack([x, y])
    d = pdist(z)
    gamma = 1 / (2 * (np.median(d[d > 0]) ** 2 + 1e-12)) if np.any(d > 0) else 1.0
    return float(
        rbf_kernel(x, x, gamma=gamma).mean()
        + rbf_kernel(y, y, gamma=gamma).mean()
        - 2 * rbf_kernel(x, y, gamma=gamma).mean()
    )


def reconstruct(cycles, durations):
    rows = []
    for cs, ds in zip(np.asarray(cycles), np.asarray(durations)):
        parts = []
        for c, d in zip(cs, ds):
            n = max(8, int(round(float(d))))
            parts.append(resample(np.asarray(c, np.float32), n).astype(np.float32))
        rows.append(np.concatenate(parts).astype(np.float32))
    return rows


def rectangular(rows):
    if not rows:
        return np.zeros((0, 1), np.float32)
    target = max(32, int(round(np.median([len(x) for x in rows]))))
    return np.stack([resample(x, target).astype(np.float32) for x in rows])


def cycle_acf_mse(real_cycles, fake_cycles, max_lag=63):
    """ACF over cycle index, not over reconstructed waveform sample lags."""
    n = min(len(real_cycles), len(fake_cycles))
    if n == 0:
        return float("nan")
    r = np.asarray(real_cycles[:n], np.float32)
    f = np.asarray(fake_cycles[:n], np.float32)
    h = min(r.shape[1], f.shape[1])
    max_lag = min(int(max_lag), h - 1)
    if max_lag < 1:
        return float("nan")

    def normalize(x):
        x = x[:, :h]
        x = x - x.mean(axis=-1, keepdims=True)
        return x / (x.std(axis=-1, keepdims=True) + 1e-6)

    r, f = normalize(r), normalize(f)
    ra = []
    fa = []
    for lag in range(1, max_lag + 1):
        ra.append(np.mean(np.mean(r[:, :-lag] * r[:, lag:], axis=-1)))
        fa.append(np.mean(np.mean(f[:, :-lag] * f[:, lag:], axis=-1)))
    return float(np.mean((np.asarray(ra) - np.asarray(fa)) ** 2))


def bounded_shape_mmd(x, y, scratch, block=256):
    """The original biased RBF MMD, all cycles and original median bandwidth.

    Disk-backed pdist and block kernel sums only bound memory; neither changes
    the cohort, features, kernel, or bandwidth policy.
    """
    x, y = np.asarray(x).reshape(len(x), -1), np.asarray(y).reshape(len(y), -1)
    z = np.vstack([x, y])
    size = len(z) * (len(z) - 1) // 2
    Path(scratch).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="block_mmd_", dir=scratch) as directory:
        distances = np.memmap(
            Path(directory) / "pdist.bin", mode="w+", dtype=np.float64, shape=(size,)
        )
        pdist(z, out=distances)
        positive = 0
        for start in range(0, size, 131072):
            chunk = np.asarray(distances[start : start + 131072])
            good = chunk[chunk > 0].copy()
            distances[positive : positive + len(good)] = good
            positive += len(good)
        median = float(np.median(distances[:positive], overwrite_input=True)) if positive else None
        gamma = 1.0 / (2.0 * (median**2 + 1e-12)) if positive else 1.0
        distances.flush()
        del distances

    def kernel_mean(a, b):
        total = 0.0
        for i in range(0, len(a), block):
            for j in range(0, len(b), block):
                total += float(
                    rbf_kernel(a[i : i + block], b[j : j + block], gamma=gamma).sum(
                        dtype=np.float64
                    )
                )
        return total / (len(a) * len(b))

    return float(kernel_mean(x, x) + kernel_mean(y, y) - 2.0 * kernel_mean(x, y))


def acf_curves(cycles, max_lag):
    x = np.asarray(cycles, np.float32)
    x = x - x.mean(axis=-1, keepdims=True)
    x = x / (x.std(axis=-1, keepdims=True) + 1e-6)
    return np.stack(
        [
            np.mean(np.mean(x[:, :-lag] * x[:, lag:], axis=-1), axis=-1)
            for lag in range(1, max_lag + 1)
        ],
        axis=1,
    )


CORR_LAGS = (1, 2, 4, 8)


ACF_LAGS = tuple(range(1, 21))


DETECTOR_NAMES = (
    "HRV_PEAK_MIN_DISTANCE_SEC",
    "HRV_PEAK_PROMINENCE",
    "HRV_PEAK_PROMINENCE_STD_SCALE",
    "HRV_RR_MIN_MS",
    "HRV_RR_MAX_MS",
    "HRV_RR_REL_FILTER",
)


def mean(a, axis=None):
    a = np.asarray(a, float)
    good = np.isfinite(a)
    count = good.sum(axis=axis)
    return np.divide(
        np.where(good, a, 0).sum(axis=axis),
        count,
        out=np.full(np.shape(count), np.nan, float),
        where=count > 0,
    )


def distance(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = np.sort(a[np.isfinite(a)]), np.sort(b[np.isfinite(b)])
    if not len(a) or not len(b):
        return np.nan, np.nan
    grid = np.unique(np.r_[a, b])
    kd = np.max(
        np.abs(
            np.searchsorted(a, grid, side="right") / len(a)
            - np.searchsorted(b, grid, side="right") / len(b)
        )
    )
    return float(kd), float(wasserstein_distance(a, b))


def acf_with_gaps(rr, retained, lag):
    rr = np.asarray(rr, float)
    retained = np.asarray(retained, bool) & np.isfinite(rr)
    if len(rr) <= lag:
        return np.nan, 0, "INSUFFICIENT_ORIGINAL_INTERVALS"
    bad = np.r_[0, np.cumsum(~retained)]
    valid = bad[lag + 1 :] - bad[: len(rr) - lag] == 0
    pairs = int(valid.sum())
    if retained.sum() < 2:
        return np.nan, pairs, "INSUFFICIENT_RETAINED_INTERVALS"
    centered = rr - rr[retained].mean()
    variance = np.mean(centered[retained] ** 2)
    if variance <= 1e-12:
        return np.nan, pairs, "ZERO_VARIANCE"
    if not pairs:
        return np.nan, pairs, "NO_CONTIGUOUS_VALID_PAIRS"
    return float(np.mean(centered[:-lag][valid] * centered[lag:][valid]) / variance), pairs, ""


def derive(cycles, durations, fs=500.0):
    cycles, durations = np.asarray(cycles, np.float32), np.asarray(durations, np.float32)
    assert fs == 500.0 and cycles.ndim == 2 and cycles.shape[1] == 256
    assert durations.shape == (len(cycles),) and np.isfinite(cycles).all()
    assert np.isfinite(durations).all() and (durations > 0).all()
    lengths = np.maximum(8, np.rint(durations).astype(np.int64))
    bounds = np.r_[0, lengths.cumsum()]
    wave = np.concatenate([resample(c, int(n)).astype(np.float32) for c, n in zip(cycles, lengths)])
    prominence = max(
        DETECTOR["HRV_PEAK_PROMINENCE"],
        DETECTOR["HRV_PEAK_PROMINENCE_STD_SCALE"] * float(np.std(wave)),
    )
    peaks, _ = find_peaks(
        wave,
        prominence=prominence,
        distance=max(1, int(DETECTOR["HRV_PEAK_MIN_DISTANCE_SEC"] * fs)),
    )
    rr = np.diff(peaks).astype(np.float32) / fs * 1000.0
    legal = np.isfinite(rr) & (rr >= DETECTOR["HRV_RR_MIN_MS"]) & (rr <= DETECTOR["HRV_RR_MAX_MS"])
    median = float(np.median(rr[legal])) if legal.any() else np.nan
    keep = legal & (np.abs(rr - median) <= DETECTOR["HRV_RR_REL_FILTER"] * median)
    assert np.array_equal(rr[keep], ORIGINAL_INTERVALS(wave, fs))
    features = SHAPE(cycles, fs)
    centered = cycles.astype(float) - cycles.astype(float).mean(1, keepdims=True)
    norms = np.sqrt((centered**2).sum(1))
    corr = {}
    for lag in CORR_LAGS:
        if len(cycles) <= lag:
            corr[lag] = np.empty(0)
            continue
        denominator = norms[:-lag] * norms[lag:]
        corr[lag] = np.divide(
            (centered[:-lag] * centered[lag:]).sum(1),
            denominator,
            out=np.full(len(cycles) - lag, np.nan),
            where=denominator > 1e-12,
        )
    return dict(
        features=features,
        peaks=peaks,
        rr=rr,
        legal=legal,
        keep=keep,
        bounds=bounds,
        corr=corr,
        prominence=prominence,
        legal_median_ms=median,
        max_horizon=len(cycles),
    )


def windows(h):
    w = h // 4
    return [("early", 0, w), ("middle", (h - w) // 2, (h - w) // 2 + w), ("late", h - w, h)]


def view(record, h):
    assert 1 <= h <= record["max_horizon"]
    # Peaks and original interval indices are never compacted after filtering.
    prefix = record["peaks"][1:] < record["bounds"][h]
    rr = record["rr"][prefix]
    keep = record["keep"][prefix]
    vals = {("morphology", name): record["features"][:h, j] for j, name in enumerate(FEATURES)}
    vals.update({("IBI", "raw"): rr, ("IBI", "filtered"): rr[keep]})
    vals.update(
        {
            ("intercycle_correlation", f"lag{lag}"): arr[: max(0, h - lag)]
            for lag, arr in record["corr"].items()
        }
    )
    scalars = {}
    reasons = {}
    for label, mask in [("raw", np.isfinite(rr)), ("filtered_gap_aware", keep)]:
        for lag in ACF_LAGS:
            value, pairs, reason = acf_with_gaps(rr, mask, lag)
            scalars["IBI_ACF", f"{label}_lag{lag}", "value"] = value
            scalars["IBI_ACF_pairs", f"{label}_lag{lag}", "count"] = pairs
            reasons[label, lag] = (reason, pairs)
    for name, value in dict(
        retention=keep.mean() if len(rr) else np.nan,
        range_legal_fraction=record["legal"][prefix].mean() if len(rr) else np.nan,
        n_peaks=int((record["peaks"] < record["bounds"][h]).sum()),
        n_raw_intervals=len(rr),
        n_filtered_intervals=int(keep.sum()),
    ).items():
        scalars["IBI_validity", name, "value"] = value
    intervals = {}
    for name, start, end in windows(h):
        complete = (record["peaks"][:-1] >= record["bounds"][start]) & (
            record["peaks"][1:] < record["bounds"][end]
        )
        v = {feature: record["features"][start:end, j] for j, feature in enumerate(FEATURES)}
        v.update(
            IBI_raw=record["rr"][complete], IBI_filtered=record["rr"][complete & record["keep"]]
        )
        intervals[name] = v
    for name, _, _ in windows(h):
        for variable, current in intervals[name].items():
            early = intervals["early"][variable]
            kd, w1 = distance(early, current)
            scalars["drift", f"{variable}/{name}", "KS_from_own_early"] = kd
            scalars["drift", f"{variable}/{name}", "W1_from_own_early"] = w1
            scalars["drift", f"{variable}/{name}", "signed_mean_shift"] = float(
                mean(current) - mean(early)
            )
    return vals, scalars, reasons


FEATURES = SHAPE_FEATURE_NAMES
SHAPE = compute_ppg_shape_features
DETECTOR = {name: globals()[name] for name in DETECTOR_NAMES}
ORIGINAL_INTERVALS = _validated_hrv_rr_intervals


def measure(
    real,
    fake,
    real_durations,
    fake_durations,
    case_ids,
    method,
    hr_registry=None,
    *,
    mmd_function=None,
):
    mmd_function = rbf_mmd if mmd_function is None else mmd_function
    real = np.asarray(real, np.float32)
    fake = np.asarray(fake, np.float32)
    rd = np.asarray(real_durations, np.float32)
    fd = np.asarray(fake_durations, np.float32)
    assert real.shape == fake.shape and real.ndim == 3 and real.shape[-1] == 256
    assert (
        rd.shape == fd.shape == real.shape[:2]
        and np.isfinite(real).all()
        and np.isfinite(fake).all()
    )
    assert (rd > 0).all() and (fd > 0).all() and len(case_ids) == len(real)
    rw, fw = reconstruct(real, rd), reconstruct(fake, fd)
    rfeatures = compute_ppg_shape_features(real.reshape(-1, 256), 500.0)
    ffeatures = compute_ppg_shape_features(fake.reshape(-1, 256), 500.0)
    rhr = []
    fhr = []
    for i, (rwave, fwave) in enumerate(zip(rw, fw)):
        for side, wave, out in [("real", rwave, rhr), ("generated", fwave, fhr)]:
            features = hrv_features(wave[None], 500.0)[0]
            hr = float(features[4]) if features[5] > 0 else np.nan
            out.append(hr)
            if hr_registry is not None:
                hr_registry.setdefault(
                    (method, case_ids[i], side),
                    dict(
                        method=method,
                        case_id=case_ids[i],
                        side=side,
                        hr_bpm=hr if np.isfinite(hr) else None,
                        valid=bool(np.isfinite(hr)),
                        validated_independent_peak_intervals=len(
                            _validated_hrv_rr_intervals(wave, 500.0)
                        ),
                    ),
                )
    rhr, fhr = np.asarray(rhr), np.asarray(fhr)
    valid_r, valid_f = np.isfinite(rhr), np.isfinite(fhr)
    paired = valid_r & valid_f
    horizon = real.shape[1]

    def cycle_acf(x):
        x = x - x.mean(-1, keepdims=True)
        x = x / (x.std(-1, keepdims=True) + 1e-6)
        return np.asarray(
            [
                np.mean(np.mean(x[:, :-lag] * x[:, lag:], axis=-1))
                for lag in range(1, min(20, horizon - 1) + 1)
            ]
        )

    acf = float(np.mean((cycle_acf(real) - cycle_acf(fake)) ** 2))
    assert np.isclose(acf, cycle_acf_mse(real, fake, min(20, horizon - 1)), atol=1e-9, rtol=1e-10)
    seam_values = []
    for wave, durations in zip(fw, fd):
        lengths = [max(8, int(round(float(d)))) for d in durations]
        derivatives = [wave, np.diff(wave), np.diff(wave, n=2)]
        for boundary in range(3, horizon, 3):
            t = sum(lengths[:boundary])
            seam_values.append(sum(abs(float(v[t]) - float(v[t - 1])) for v in derivatives))
    result = dict(
        shape_mmd=mmd_function(rfeatures, ffeatures),
        psd_log_mse=psd_metrics(rectangular(rw), rectangular(fw), 500.0)["psd_log_mse_0p5_15hz"],
        cycle_acf_mse=acf,
        hr_mae_bpm=float(np.abs(rhr[paired] - fhr[paired]).mean()) if paired.any() else None,
        hr_w1_bpm=(
            float(wasserstein_distance(rhr[valid_r], fhr[valid_f]))
            if valid_r.any() and valid_f.any()
            else None
        ),
        real_hr_valid=int(valid_r.sum()),
        fake_hr_valid=int(valid_f.sum()),
        paired_hr_valid=int(paired.sum()),
        mean_future_internal_seam=float(np.mean(seam_values)) if seam_values else None,
        future_internal_seam_boundaries=len(seam_values),
    )
    return result


def primary(
    real, fake, real_durations, fake_durations, case_ids, method, scratch, hr_registry=None
):
    result = measure(
        real,
        fake,
        real_durations,
        fake_durations,
        case_ids,
        method,
        hr_registry,
        mmd_function=lambda x, y: bounded_shape_mmd(x, y, scratch),
    )
    result["cycle_acf_mse_lag20"] = result.pop("cycle_acf_mse")
    lag = min(63, real.shape[1] - 1)
    result["cycle_acf_mse_lag63"] = float(
        np.mean((acf_curves(real, lag).mean(0) - acf_curves(fake, lag).mean(0)) ** 2)
    )
    return result
