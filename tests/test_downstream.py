"""Synthetic checks for downstream windowing, losses and result aggregation."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from longmorph.downstream import augmentation, metrics, models, pairs, reporting, training
from longmorph.downstream.forecasting import fixed_training_groups


class DownstreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.rng = np.random.default_rng(22)
        self.cycles = self.rng.normal(size=(12, 9, 256)).astype(np.float32)

    def test_forecaster_shapes(self):
        for model, n_input, n_target in [
            (models.TCNForecaster(), 3, 3),
            (models.ResidualTCNForecaster(), 3, 3),
            (models.PatchTSTForecaster(d_model=8, n_heads=2, e_layers=1, d_ff=16), 5, 1),
        ]:
            model.eval()
            with torch.no_grad():
                result = model(torch.randn(2, n_input, 256))
            self.assertEqual(tuple(result.shape), (2, n_target, 256))
            self.assertTrue(torch.isfinite(result).all())
        with torch.no_grad():
            self.assertEqual(
                tuple(models.HeartRateRegressor()(torch.randn(2, 1, 512)).shape), (2, 1)
            )

    def test_residual_forecaster_identity_when_delta_is_zero(self):
        model = models.ResidualTCNForecaster().eval()
        for parameter in model.parameters():
            parameter.data.zero_()
        x = torch.from_numpy(self.cycles[:2, :3])
        with torch.no_grad():
            self.assertTrue(torch.equal(model(x), x))

    def test_real_pairs_respect_subjects_and_target_durations(self):
        subjects = np.repeat(np.arange(6), 2)
        durations = np.arange(108, dtype=np.float32).reshape(12, 9)
        data = {"cycles": self.cycles, "durations_ms": durations, "subject_index": subjects}
        x, y, target_durations, selected = pairs.real_pairs(data, {1, 3}, 5, 1)
        mask = np.isin(subjects, [1, 3])
        np.testing.assert_array_equal(x, self.cycles[mask, :5])
        np.testing.assert_array_equal(y, self.cycles[mask, 5:6])
        np.testing.assert_array_equal(target_durations, durations[mask, 5:6])
        self.assertEqual(set(selected), {1, 3})

    def test_synthetic_pairs_exclude_real_prefix(self):
        x, y = pairs.sequence_pairs(self.cycles, 3, 3, prefix=3, stride=3)
        self.assertEqual(len(x), 12)
        np.testing.assert_array_equal(x[0], self.cycles[0, 3:6])
        np.testing.assert_array_equal(y[0], self.cycles[0, 6:9])
        with self.assertRaises(ValueError):
            pairs.sequence_pairs(self.cycles, prefix=-1)

    def test_generator_subject_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.npz"
            np.savez_compressed(
                path,
                synthetic_x=self.cycles[:, :5],
                synthetic_y=self.cycles[:, 5:6],
                train_subject_index=[1, 2],
            )
            x, _ = pairs.synthetic_pairs(path, {1, 2})
            self.assertEqual(len(x), 12)
            with self.assertRaises(ValueError):
                pairs.synthetic_pairs(path, {1})

    def test_subject_and_source_splits_are_disjoint(self):
        splits = pairs.subject_splits({"subject_index": np.arange(15)}, 5)
        self.assertEqual(sorted(s for split in splits for s in split["test"]), list(range(15)))
        for split in splits:
            self.assertFalse(set(split["train"]) & set(split["validation"]))
            self.assertFalse(set(split["train"]) & set(split["test"]))
        sources = [f"source_{index}" for index in range(10)]
        train, validation = pairs.split_sources(sources, 0.2, 2, 7)
        self.assertFalse(train & validation)
        self.assertEqual(train | validation, set(sources))
        self.assertEqual((train, validation), pairs.split_sources(sources, 0.2, 2, 7))

    def test_equal_budget_training_groups(self):
        x, y = self.cycles[:, :5], self.cycles[:, 5:6]
        groups = fixed_training_groups(
            x,
            y,
            {"generator": (x, y)},
            5,
            7,
            1000,
            ["real_only", "synthetic_only", "half_real_half_generated"],
        )
        self.assertEqual([len(row[2]) for row in groups], [5, 5, 5])
        self.assertEqual(groups[-1][4:6], (2, 3))
        self.assertEqual([row[-1] for row in groups], [7, 10_000_007, 10_000_007])

    def test_exact_forecast_metrics(self):
        target = self.cycles[:3, :1]
        result = metrics.metrics(target, target, np.full((3, 1), 800), np.arange(3))
        for name in metrics.LOWER_BETTER:
            np.testing.assert_allclose(result[name], 0, atol=1e-12)
        np.testing.assert_allclose(result["pearson_r"], 1, atol=1e-12)

    def test_training_objectives_accept_matching_targets(self):
        args = SimpleNamespace(
            loss_raw_kind="l1",
            spectral_loss_mode="phase_welch_log_power",
            peak_softmax_temperature=20.0,
            loss_d1_weight=0.15,
            loss_d2_weight=0.05,
            loss_spectral_weight=0.02,
            loss_correlation_weight=0.05,
            loss_peak_weight=0.05,
        )
        target = torch.from_numpy(self.cycles[:3, :3])
        self.assertLess(abs(float(training.fixed_budget_loss(target, target, args))), 1e-6)
        self.assertTrue(torch.all(training.per_sample_loss(target, target, args).abs() < 1e-6))

    def test_quality_filter_fits_only_real_reference_statistics(self):
        x, y = self.cycles[:6, :3], self.cycles[:6, 3:6]
        synthetic_x, synthetic_y = x + 0.1, y + 0.1
        selected_x, _, info = augmentation.quality_filter(
            x, y, synthetic_x, synthetic_y, 0.5, 1, np.ones(12, np.float32)
        )
        self.assertEqual(len(selected_x), 3)
        np.testing.assert_allclose(
            info["real_center"], np.median(augmentation.pair_features(x, y), axis=0)
        )
        self.assertEqual(info["kept_pairs"], 3)

    def test_paired_report_uses_matching_subjects_and_seeds(self):
        target = self.cycles[:3, :1]
        frame = metrics.metrics(target, target, np.full((3, 1), 800), np.arange(3))
        rows = []
        for group, generator, offset in [
            ("real_only", "__shared__", 0),
            ("synthetic_only", "generator", 0.1),
        ]:
            for seed in [7, 17]:
                sample = frame.copy()
                sample["waveform_mae"] += offset
                sample["generator"] = generator
                sample["group"] = group
                sample["seed"] = seed
                sample["fold"] = 1
                rows.append(sample)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "metrics").mkdir()
            paths = reporting.summarize_fixed_budget(pd.concat(rows, ignore_index=True), output)
            delta = pd.read_csv(paths["paired_delta"])
            self.assertEqual(delta.paired_subject_seed_count.iloc[0], 6)
            self.assertAlmostEqual(delta.waveform_mae_delta_vs_real_only.iloc[0], 0.1)


if __name__ == "__main__":
    unittest.main()
