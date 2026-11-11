"""Iterative amplitude-adjusted Fourier-transform surrogates."""

import numpy as np


def iaaft(x, rng, iterations):
    sorted_x = np.sort(x)
    target_amp = np.abs(np.fft.rfft(x))
    y = rng.permutation(x)
    for _ in range(iterations):
        phase = np.angle(np.fft.rfft(y))
        y_spec = np.fft.irfft(target_amp * np.exp(1j * phase), n=len(x))
        ranks = np.argsort(np.argsort(y_spec))
        y = sorted_x[ranks]
    return y.astype(np.float32)


def generate(train, seed, num_samples, iterations=100):
    rng = np.random.default_rng(seed)
    return np.stack(
        [iaaft(train[rng.integers(len(train))], rng, iterations) for _ in range(num_samples)]
    )
