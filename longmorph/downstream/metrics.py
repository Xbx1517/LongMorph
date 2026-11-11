"""Forecasting metrics evaluated on held-out real cycles."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch

METRICS = [
    "waveform_mae",
    "waveform_rmse",
    "pearson_r",
    "d1_mae",
    "d2_mae",
    "log_psd_mse_phase_domain",
    "peak_time_mae_ms_target_duration_scaled",
]
LOWER_BETTER = set(METRICS) - {"pearson_r"}


def metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    durations: np.ndarray,
    subjects: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for index, (pred, true) in enumerate(zip(prediction, target)):
        pred_flat = pred.reshape(-1)
        true_flat = true.reshape(-1)
        if np.std(pred_flat) > 0 and np.std(true_flat) > 0:
            pearson = float(np.corrcoef(pred_flat, true_flat)[0, 1])
        else:
            pearson = 0.0
        d1 = float(np.mean(np.abs(np.diff(pred, axis=-1) - np.diff(true, axis=-1))))
        d2 = float(np.mean(np.abs(np.diff(pred, n=2, axis=-1) - np.diff(true, n=2, axis=-1))))
        peak_ms = float(
            np.mean(
                [
                    abs(np.argmax(pred[j]) - np.argmax(true[j])) / 255.0 * durations[index, j]
                    for j in range(target.shape[1])
                ]
            )
        )
        spectra = []
        for signal in (pred_flat, true_flat):
            _, power = welch(signal, fs=256, nperseg=min(256, len(signal)))
            spectra.append(power)
        rows.append(
            {
                "subject_index": int(subjects[index]),
                "waveform_mae": float(np.mean(np.abs(pred - true))),
                "waveform_rmse": float(np.sqrt(np.mean((pred - true) ** 2))),
                "pearson_r": pearson,
                "d1_mae": d1,
                "d2_mae": d2,
                "log_psd_mse_phase_domain": float(
                    np.mean((np.log(spectra[0] + 1e-12) - np.log(spectra[1] + 1e-12)) ** 2),
                ),
                "peak_time_mae_ms_target_duration_scaled": peak_ms,
            }
        )
    return pd.DataFrame(rows)
