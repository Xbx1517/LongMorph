"""Synthetic checks for baseline representation, adapters and explicit run I/O."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from longmorph.baselines import common
from longmorph.baselines.cli import parser, train
from longmorph.baselines.iaaft import generate, iaaft


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.train = self.rng.uniform(0, 1, (8, 192)).astype(np.float32)

    def test_iaaft_preserves_empirical_distribution_and_seed(self):
        source = self.train[0]
        result = iaaft(source, np.random.default_rng(17), 10)
        np.testing.assert_array_equal(np.sort(result), np.sort(source))
        np.testing.assert_array_equal(
            generate(self.train, 17, 3, 10), generate(self.train, 17, 3, 10)
        )
        self.assertFalse(np.array_equal(generate(self.train, 17, 3), generate(self.train, 27, 3)))

    def test_codec_uses_training_statistics(self):
        codec, latent, report = common.fit_codec(self.train, n_blocks=24, rank=4)
        self.assertEqual(latent.shape, (8, 24, 4))
        np.testing.assert_allclose(latent.mean(axis=(0, 1)), 0, atol=2e-7)
        np.testing.assert_allclose(latent.std(axis=(0, 1)), 1, atol=2e-7)
        np.testing.assert_allclose(
            codec["mean"],
            self.train.reshape(-1, 8).astype(np.float64).mean(axis=0),
            atol=1e-7,
        )
        before = {key: value.copy() for key, value in codec.items()}
        test = self.rng.uniform(10, 20, (3, 192)).astype(np.float32)
        common.encode_latents(codec, test)
        for key in codec:
            np.testing.assert_array_equal(codec[key], before[key])
        self.assertEqual(report["fit_split"], "train_only")
        reconstructed = common.decode_latents(codec, latent)
        self.assertEqual(reconstructed.shape, self.train.shape)
        self.assertTrue(np.all((reconstructed >= 0) & (reconstructed <= 1)))

    def test_context_encoding_uses_observed_patches_only(self):
        codec, _z, _report = common.fit_codec(self.train, n_blocks=24, rank=4)
        full = common.encode_latents(codec, self.train)
        context = common.encode_latents(codec, self.train[:, :24], n_blocks=3)
        np.testing.assert_array_equal(full[:, :3], context)
        _codec, _train, context, _report = common.prepare_latents(
            self.train, self.train[:, :24], n_blocks=24, rank=4, test_blocks=3
        )
        self.assertEqual(context.shape, (8, 3, 4))

    def test_codec_validates_rank_and_shapes(self):
        with self.assertRaises(ValueError):
            common.fit_codec(self.train, n_blocks=24, rank=9)
        with self.assertRaises(ValueError):
            common.fit_codec(self.train, n_blocks=25, rank=4)
        codec, _z, _report = common.fit_codec(self.train, n_blocks=24, rank=4)
        with self.assertRaises(ValueError):
            common.encode_latents(codec, np.zeros((1, 128)))
        with self.assertRaises(ValueError):
            common.decode_latents(codec, np.zeros((1, 24, 5)))

    def test_iaaft_cli_uses_explicit_data_and_exports_finite_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, output = root / "reference", root / "generated"
            (reference / "data").mkdir(parents=True)
            np.savez_compressed(reference / "data/train_ppg_segments.npz", x=self.train)
            np.savez_compressed(reference / "data/test_ppg_segments.npz", x=self.train[:2])
            args = parser().parse_args(
                [
                    "train",
                    "--model",
                    "iaaft",
                    "--reference-run",
                    str(reference),
                    "--output-dir",
                    str(output),
                    "--seed",
                    "17",
                    "--num-samples",
                    "3",
                    "--iterations",
                    "2",
                ]
            )
            result = train(args)
            with np.load(result["generated"]) as generated:
                values = generated["x"]
            np.testing.assert_array_equal(values[..., 0], generate(self.train, 17, 3, 2))
            self.assertEqual(values.shape, (3, 192, 1))
            self.assertTrue((output / "data/train_ppg_segments.npz").is_symlink())
            self.assertTrue((output / "runtime/training_and_inference.json").is_file())
            with self.assertRaises(FileExistsError):
                train(args)
            np.testing.assert_array_equal(
                common.load_waveforms(reference / "data/train_ppg_segments.npz"),
                self.train,
            )

    def test_loading_validates_data_without_flattening_channels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "segments.npz"
            np.savez_compressed(path, x=np.zeros((3, 24, 2)))
            with self.assertRaises(ValueError):
                common.load_waveforms(path)
            np.savez_compressed(path, x=np.full((3, 24), np.nan))
            with self.assertRaises(ValueError):
                common.load_waveforms(path)

    def test_optional_frameworks_are_not_loaded_for_listing(self):
        program = (
            "import sys; from longmorph.baselines.cli import main; main(['list']); "
            "assert not any(name in sys.modules for name in ('torch', 'tensorflow', 'transformers'))"
        )
        completed = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for name in ("iaaft", "aec_gan", "timegan", "timevae", "timevqvae"):
            self.assertIn(name, completed.stdout)

    def test_unavailable_manual_source_cannot_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            args = parser().parse_args(
                [
                    "train",
                    "--model",
                    "aec_gan",
                    "--reference-run",
                    str(Path(directory) / "reference"),
                    "--output-dir",
                    str(output),
                    "--seed",
                    "7",
                    "--context-path",
                    str(Path(directory) / "context.npz"),
                ]
            )
            with self.assertRaisesRegex(ValueError, "explicit --vendor-dir"):
                train(args)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
