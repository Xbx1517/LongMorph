"""Distribution pooling, native physiology, and missing-reference contracts."""

import tempfile
import unittest
from pathlib import Path
import numpy as np
from longmorph.evaluation.generation import evaluate
from longmorph.evaluation import metrics as _metrics


class EvaluationTests(unittest.TestCase):
    def test_current_pooled_metrics_and_reference_availability(self):
        phase = np.linspace(0, 1, 256, dtype=np.float32)
        pulses = np.stack(
            [
                np.exp(-(((phase - center) / width) ** 2))
                for center, width in ((0.23, 0.1), (0.46, 0.17))
            ]
        )
        real = np.repeat(pulses[:, None], 21, axis=1).astype(np.float32)
        fake = real[::-1].copy()
        durations = np.full((2, 21), 500.0, np.float32)
        with tempfile.TemporaryDirectory(prefix="longmorph-metrics-") as directory:
            folder = Path(directory)
            metadata = dict(
                case_id=np.array(["CASE_A", "CASE_B"]),
                subject_id=np.array(["A", "B"]),
                sample_rate=np.array(500.0),
            )
            np.savez_compressed(
                folder / "real.npz", future_cycles=real, future_durations=durations, **metadata
            )
            for seed in (7, 17):
                np.savez_compressed(
                    folder / f"fake{seed}.npz", future_cycles=fake, durations=durations, **metadata
                )
            result = evaluate(
                [folder / "fake7.npz", folder / "fake17.npz"],
                folder / "real.npz",
                folder / "evaluated",
                [7, 17],
            )
            self.assertEqual(len(result["pooled"]), 2)
            self.assertAlmostEqual(result["pooled"][0]["shape_mmd"], 0.0, places=7)
            self.assertAlmostEqual(result["pooled"][0]["hr_mae_bpm"], 0.0, places=7)
            self.assertAlmostEqual(result["pooled"][0]["hr_w1_bpm"], 0.0, places=7)
            self.assertIn("cycle_acf_mse_lag20", result["pooled"][0])
            self.assertIn("cycle_acf_mse_lag63", result["pooled"][0])
            self.assertEqual(result["primary_summary"][0]["shape_mmd_finite_seed_count"], 2)
            per_anchor_mmd = []
            for i in range(2):
                r = _metrics.compute_ppg_shape_features(real[i], 500.0)
                f = _metrics.compute_ppg_shape_features(fake[i], 500.0)
                per_anchor_mmd.append(_metrics.rbf_mmd(r, f))
            self.assertGreater(np.mean(per_anchor_mmd), 1e-5)
            r = _metrics.compute_ppg_shape_features(real.reshape(-1, 256), 500.0)
            f = _metrics.compute_ppg_shape_features(fake.reshape(-1, 256), 500.0)
            self.assertAlmostEqual(
                _metrics.bounded_shape_mmd(r, f, folder / "scratch"),
                _metrics.rbf_mmd(r, f),
                places=7,
            )
            padded = real.copy()
            padded[1, 10:] = np.nan
            padded_duration = durations.copy()
            padded_duration[1, 10:] = np.nan
            np.savez_compressed(
                folder / "partial.npz",
                future_cycles=padded,
                future_durations=padded_duration,
                available_future_cycles=np.array([21, 10]),
                **metadata,
            )
            partial = evaluate(
                [folder / "fake7.npz"],
                folder / "partial.npz",
                folder / "partial_evaluation",
                [7],
                requested_slots=3,
            )
            self.assertEqual(partial["pooled"][0]["reference_eligible"], 1)
            self.assertEqual(partial["pooled"][0]["reference_missing"], 2)
            self.assertIsNone(partial["primary_summary"][0]["shape_mmd_sample_sd"])


if __name__ == "__main__":
    unittest.main()
