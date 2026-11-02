"""Policy tests with controlled data, independent of the large FMA CSVs."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from preprocessing import correlation_filter, duplicate_decisions, restore_representations, run_pipeline
from exploration import readable_table


class CleaningPolicies(unittest.TestCase):
    def test_duplicate_vectors_require_matching_artist_and_title(self):
        features = pd.DataFrame({"x": [1., 1., 1., 1., 2., 2.]},
                                index=pd.Index([1, 2, 3, 4, 5, 6], name="track_id"))
        metadata = pd.DataFrame({
            ("track", "title"): ["Song", " BAND - song ", "Different", "Song", None, None],
            ("artist", "name"): ["Band", "band", "Band", "Other", "Band", "Band"],
            ("set", "split"): ["test", "training", "training", "training", "test", "test"],
        }, index=features.index)
        mapping, audit = duplicate_decisions(features, metadata)
        # Prefer training even though it has a larger ID; consolidate title prefix.
        self.assertEqual(mapping.loc[1, "canonical_track_id"], 2)
        self.assertEqual(mapping.loc[2, "canonical_track_id"], 2)
        # Distinct titles/artists and absent titles never justify consolidation.
        for tid in [3, 4, 5, 6]:
            self.assertEqual(mapping.loc[tid, "canonical_track_id"], tid)
        self.assertEqual(audit.decision.eq("remove_supported_duplicate").sum(), 1)

    def test_correlation_chain_keeps_a_surviving_witness(self):
        rng = np.random.default_rng(12)
        values = rng.normal(size=(100, 2))
        values -= values.mean(axis=0)
        orthogonal, _ = np.linalg.qr(values)
        x, y = orthogonal.T
        frame = pd.DataFrame({
            "mfcc__mean__01": x,
            "mfcc__median__01": np.cos(.25) * x + np.sin(.25) * y,
            "mfcc__std__01": np.cos(.5) * x + np.sin(.5) * y,
        })
        # A~B and B~C exceed .95, while A~C does not. Dropping both B and C
        # would leave C without a retained correlated representative.
        kept, drops = correlation_filter(frame)
        self.assertEqual(kept, ["mfcc__mean__01", "mfcc__std__01"])
        self.assertEqual(drops.removed_feature.tolist(), ["mfcc__median__01"])
        self.assertEqual(drops.retained_feature.tolist(), ["mfcc__mean__01"])
        self.assertEqual(correlation_filter(frame[frame.columns[::-1]])[0], kept[::-1])

    def test_flattening_rejects_ambiguous_labels(self):
        frame = pd.DataFrame([[1, 2]], columns=pd.MultiIndex.from_tuples(
            [("a__b", "c"), ("a", "b__c")]))
        with self.assertRaises(ValueError):
            readable_table(frame)

    def test_holdout_values_do_not_change_fitted_transforms_or_selection(self):
        rng = np.random.default_rng(8)
        features = pd.DataFrame(
            np.exp(rng.normal(size=(60, 3))), index=pd.Index(range(1, 61), name="track_id"),
            columns=pd.MultiIndex.from_tuples([
                ("mfcc", "mean", "01"), ("mfcc", "mean", "02"), ("rmse", "mean", "01")]))
        metadata = pd.DataFrame({
            ("track", "title"): [f"Track {i}" for i in features.index],
            ("artist", "name"): ["Artist"] * 60,
            ("set", "split"): ["training"] * 40 + ["validation"] * 10 + ["test"] * 10,
        }, index=features.index)
        bundles = []
        with tempfile.TemporaryDirectory() as directory:
            for iteration in range(2):
                root = Path(directory) / str(iteration)
                changed = features.copy()
                if iteration:
                    changed.iloc[40:] *= 7  # Only held-out values differ.
                with patch("preprocessing.load_small", return_value=(metadata, changed, pd.Series(dtype=float))):
                    run_pipeline(root)
                bundle = joblib.load(root / "data/processed/transformations.joblib")
                bundles.append(bundle)
                clean = pd.read_parquet(root / "data/processed/features_clean.parquet")
                restored = restore_representations(clean[clean.columns[::-1]], bundle)
                for name, actual in restored.items():
                    expected = pd.read_parquet(root / f"data/processed/{name}.parquet")
                    pd.testing.assert_frame_equal(actual, expected)
                with self.assertRaises(ValueError):
                    restore_representations(clean.iloc[:, :-1], bundle)
        first, second = bundles
        np.testing.assert_array_equal(first["fit_track_ids"], range(1, 41))
        for name in ["baseline_scaler", "transformed_scaler"]:
            np.testing.assert_array_equal(first[name].mean_, second[name].mean_)
            np.testing.assert_array_equal(first[name].scale_, second[name].scale_)
        self.assertEqual(first["correlation_features"], second["correlation_features"])
        self.assertEqual(set(first["power_transformers"]), set(second["power_transformers"]))
        for name, transform in first["power_transformers"].items():
            np.testing.assert_array_equal(transform.lambdas_, second["power_transformers"][name].lambdas_)
        np.testing.assert_array_equal(first["pca"].components_, second["pca"].components_)


if __name__ == "__main__":
    unittest.main()
