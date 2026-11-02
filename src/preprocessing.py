"""Conservative cleaning and training-fitted representations of FMA Small.

Run from the repository: python src/preprocessing.py
Raw metadata/audio are never changed. Every output row retains a track ID.
"""

import argparse
import json
from importlib.metadata import version
from pathlib import Path
import unicodedata
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer, StandardScaler

from exploration import flatten_labels, load_small, readable_table


def normalized_text(value):
    if pd.isna(value):
        return ""
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def duplicate_decisions(features, metadata):
    """Match exact vectors AND artist/title; preserve conflicting titles.

    Normalize case/whitespace/Unicode and an explicit 'Artist - ' title prefix.
    Equality of vectors is checked after grouping by hash. Prefer the earliest
    official split (training, validation, test), then the lowest track ID.
    This is a metadata-supported consolidation, not an audio identity test.
    """
    mapping = pd.Series(features.index, index=features.index, name="canonical_track_id")
    rows = []
    candidates = features.loc[features.duplicated(keep=False)]
    hashes = pd.util.hash_pandas_object(candidates, index=False)
    split_rank = {"training": 0, "validation": 1, "test": 2}
    for group, (_, ids) in enumerate(hashes.groupby(hashes, sort=False).groups.items(), 1):
        values = features.loc[ids].to_numpy()
        if not np.array_equal(values, np.broadcast_to(values[0], values.shape), equal_nan=True):
            raise ValueError("Hash collision in duplicate grouping")
        keys = {}
        for tid in ids:
            artist = normalized_text(metadata.loc[tid, ("artist", "name")])
            title = normalized_text(metadata.loc[tid, ("track", "title")])
            prefix = artist + " - "
            if artist and title.startswith(prefix):
                title = title[len(prefix):]
            # Missing labels never justify merging distinct records.
            key = (artist, title) if artist and title else ("unmatched", int(tid))
            keys.setdefault(key, []).append(tid)
        for members in keys.values():
            canonical = min(members, key=lambda tid: (
                split_rank[metadata.loc[tid, ("set", "split")]], int(tid)))
            for tid in members:
                mapping.loc[tid] = canonical
                rows.append({
                    "track_id": int(tid), "vector_group": group,
                    "canonical_track_id": int(canonical),
                    "title": metadata.loc[tid, ("track", "title")],
                    "artist": metadata.loc[tid, ("artist", "name")],
                    "split": metadata.loc[tid, ("set", "split")],
                    "decision": "remove_supported_duplicate" if tid != canonical else
                                "retain_unresolved" if len(keys) > 1 else "retain_canonical",
                })
    columns = ["track_id", "vector_group", "canonical_track_id", "title", "artist", "split", "decision"]
    audit = pd.DataFrame(rows, columns=columns).sort_values("track_id")
    return mapping.to_frame(), audit


def correlation_filter(frame, threshold=.95):
    """Greedy Pearson filtering with retained-column witnesses for every drop.

    Prefer interpretable mean, median, std, min, max, skew, then kurtosis;
    break ties by family/component/name. Only compare to columns still kept,
    so each discarded column has a surviving correlated representative.
    """
    preference = {stat: i for i, stat in enumerate(
        ["mean", "median", "std", "min", "max", "skew", "kurtosis"])}
    def order(name):
        family, stat, component = name.split("__")
        return preference.get(stat, 99), family, component, name
    corr = frame.corr(method="pearson")
    kept, removed = [], []
    for name in sorted(frame.columns, key=order):
        witnesses = corr.loc[name, kept].abs() if kept else pd.Series(dtype=float)
        if witnesses.ge(threshold).any():
            representative = witnesses.idxmax()
            removed.append({"removed_feature": name, "retained_feature": representative,
                            "pearson_r": corr.loc[name, representative],
                            "absolute_r": witnesses.loc[representative]})
        else:
            kept.append(name)
    # Preserve the source feature order in output matrices.
    retained = [name for name in frame.columns if name in set(kept)]
    drops = pd.DataFrame(removed, columns=["removed_feature", "retained_feature", "pearson_r", "absolute_r"])
    return retained, drops

#power transform reshapes skewed feature distributions and normalizes them
def apply_power_transforms(frame, transformers):
    result = frame.copy()
    for name, transformer in transformers.items():
        result[name] = transformer.transform(frame[[name]]).ravel()
    return result


def restore_representations(features, bundle):
    """Apply saved transforms to clean flat features in their recorded order.

    This does not deduplicate or merge tracks. Accepts the exact clean schema;
    a different column order is safely restored, but missing/extra columns fail.
    """
    names = bundle["feature_names"]
    if not features.columns.is_unique or set(features.columns) != set(names):
        raise ValueError("Input columns must match the saved clean feature schema")
    frame = features.loc[:, names]
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Non-finite values in transformation input")
    baseline = pd.DataFrame(bundle["baseline_scaler"].transform(frame),
                            index=frame.index, columns=names)
    transformed = apply_power_transforms(frame, bundle["power_transformers"])
    scaled = pd.DataFrame(bundle["transformed_scaler"].transform(transformed),
                          index=frame.index, columns=names)
    filtered = scaled[bundle["correlation_features"]]
    def project(pca):
        return pd.DataFrame(pca.transform(scaled), index=frame.index,
                            columns=[f"PC{i + 1:03d}" for i in range(pca.n_components_)])
    return {"features_scaled": baseline, "features_transformed_scaled": scaled,
            "features_correlation_filtered": filtered,
            "features_pca": project(bundle["pca"])}


def run_pipeline(root, correlation_threshold=.95, pca_variance=.95):
    root = Path(root)
    output, reports = root / "data/processed", root / "reports/cleaning"
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    metadata, original, inventory = load_small(root / "data/fma_metadata")
    mapping, duplicates = duplicate_decisions(original, metadata)
    keep = mapping.index == mapping.canonical_track_id
    clean = readable_table(original.loc[keep].sort_index())
    if not np.isfinite(clean.to_numpy()).all():
        raise ValueError("Unexpected non-finite values: rerun exploration before choosing imputation")
    constant = clean.columns[clean.nunique().le(1)].tolist()
    clean = clean.drop(columns=constant)
    flat_metadata = readable_table(metadata)
    clean_metadata = flat_metadata.loc[clean.index]
    train_ids = clean_metadata.index[clean_metadata["set__split"].eq("training")]
    train = clean.loc[train_ids]
    if not len(train):
        raise ValueError("No training rows available")
    if train.nunique().le(1).any():
        raise ValueError("Training-constant features found; review before fitting transformations")

    baseline_scaler = StandardScaler().fit(train)
    before_skew = train.skew()
    modal_fraction = train.apply(lambda s: s.value_counts(normalize=True).iloc[0])
    # Power transforms are inappropriate for saturated/nearly invariant columns.
    eligible = before_skew.abs().gt(2) & modal_fraction.lt(.99) & train.nunique().gt(10)
    transformers, transform_rows = {}, []
    for name in train.columns:
        decision = "keep_original"
        reason = "below_skew_threshold"
        if modal_fraction[name] >= .99 or train[name].nunique() <= 10:
            reason = "near_invariant_or_low_cardinality"
        elif eligible[name]:
            # Compare the transformed TRAINING skew before adopting a transform.
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    transformer = PowerTransformer(method="yeo-johnson", standardize=False).fit(train[[name]])
                    transformed_all = transformer.transform(clean[[name]]).ravel()
                if not np.isfinite(transformed_all).all():
                    raise ValueError("non-finite transformed values")
                candidate_skew = pd.Series(transformer.transform(train[[name]]).ravel()).skew()
                if abs(candidate_skew) < abs(before_skew[name]):
                    transformers[name] = transformer
                    decision, reason = "yeo_johnson", "training_absolute_skew_improved"
                else:
                    reason = "training_absolute_skew_not_improved"
            except (ValueError, FloatingPointError, RuntimeWarning) as exc:
                reason = f"numerical_failure: {type(exc).__name__}"
        transform_rows.append({"feature_name": name, "decision": decision, "reason": reason,
                               "training_skew_before": before_skew[name],
                               "training_modal_fraction": modal_fraction[name],
                               "lambda": transformers[name].lambdas_[0] if name in transformers else np.nan})
    transformed = apply_power_transforms(clean, transformers)
    transformed_scaler = StandardScaler().fit(transformed.loc[train_ids])
    scaled = pd.DataFrame(transformed_scaler.transform(transformed), index=clean.index, columns=clean.columns)
    retained, correlations_dropped = correlation_filter(scaled.loc[train_ids], correlation_threshold)
    # PCA is an independent alternative on all transformed/scaled features,
    # not a second reduction of the correlation-filtered matrix.
    pca = PCA(n_components=pca_variance, svd_solver="full").fit(scaled.loc[train_ids])
    # Baseline PCA gives a fair dimensionality comparison with no power transforms.
    baseline_train = pd.DataFrame(baseline_scaler.transform(train), index=train.index, columns=train.columns)
    baseline_pca = PCA(n_components=pca_variance, svd_solver="full").fit(baseline_train)
    bundle = {"feature_names": clean.columns.tolist(), "fit_track_ids": train_ids.tolist(),
              "power_transformers": transformers, "baseline_scaler": baseline_scaler,
              "transformed_scaler": transformed_scaler, "correlation_features": retained,
              "pca": pca, "baseline_pca": baseline_pca,
              "config": {"correlation_threshold": correlation_threshold, "pca_variance": pca_variance,
                         "skew_threshold": 2, "near_invariant_fraction": .99, "fit_split": "training"}}
    representations = restore_representations(clean, bundle)
    clean.to_parquet(output / "features_clean.parquet")
    flat_metadata.to_parquet(output / "metadata_all.parquet")
    clean_metadata.to_parquet(output / "metadata_clean.parquet")
    mapping.to_parquet(output / "track_mapping.parquet")
    for name, frame in representations.items():
        frame.to_parquet(output / f"{name}.parquet")
    joblib.dump(bundle, output / "transformations.joblib")

    transform_audit = pd.DataFrame(transform_rows).set_index("feature_name")
    transform_audit["training_skew_after"] = transformed.loc[train_ids].skew()
    transform_audit["all_tracks_skew_before"] = clean.skew()
    transform_audit["all_tracks_skew_after"] = transformed.skew()
    transform_audit.to_csv(reports / "transform_decisions.csv")
    duplicates.to_csv(reports / "duplicate_decisions.csv", index=False)
    correlations_dropped.to_csv(reports / "correlation_drops.csv", index=False)
    baseline_retained, baseline_drops = correlation_filter(baseline_train, correlation_threshold)
    baseline_drops.to_csv(reports / "baseline_correlation_drops.csv", index=False)
    variance = pd.DataFrame({"component": np.arange(1, pca.n_components_ + 1),
                            "explained_variance_ratio": pca.explained_variance_ratio_,
                            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_)})
    variance.to_csv(reports / "pca_variance.csv", index=False)
    pd.DataFrame(pca.components_, index=[f"PC{i+1:03d}" for i in range(pca.n_components_)],
                 columns=clean.columns).rename_axis("component").to_csv(reports / "pca_loadings.csv")

    matrices = {"clean": clean, **representations}
    comparison = []
    for name, frame in matrices.items():
        values = frame.loc[train_ids].to_numpy()
        is_standardized = name in {"features_scaled", "features_transformed_scaled", "features_correlation_filtered"}
        if not np.isfinite(frame.to_numpy()).all() or not frame.index.equals(clean.index):
            raise AssertionError(f"Invalid matrix: {name}")
        if is_standardized:
            np.testing.assert_allclose(values.mean(axis=0), 0, atol=1e-10)
            np.testing.assert_allclose(values.std(axis=0), 1, atol=1e-10)
        comparison.append({"representation": name, "tracks": len(frame), "columns": frame.shape[1],
                           "training_abs_skew_gt2": int(frame.loc[train_ids].skew().abs().gt(2).sum())})
    pd.DataFrame(comparison).to_csv(reports / "representation_comparison.csv", index=False)
    heldout_rows = []
    reconstruction = pca.inverse_transform(representations["features_pca"])
    for split in ["training", "validation", "test"]:
        mask = clean_metadata["set__split"].eq(split)
        actual = scaled.loc[mask].to_numpy()
        residual = actual - reconstruction[mask]
        centered = actual - pca.mean_
        heldout_rows.append({"split": split, "tracks": int(mask.sum()),
                             "mean_squared_reconstruction_error": float(np.mean(residual ** 2)),
                             "retained_energy_fraction_about_training_mean":
                                 float(1 - (residual ** 2).sum() / (centered ** 2).sum())})
    pd.DataFrame(heldout_rows).to_csv(reports / "pca_reconstruction.csv", index=False)

    # Structural checks: every dropped feature maps to a retained one, and
    # no retained pair meets the filtering threshold on the training data.
    assert set(correlations_dropped.retained_feature).issubset(retained)
    remaining_corr = scaled.loc[train_ids, retained].corr().abs().to_numpy()
    assert not np.triu(remaining_corr >= correlation_threshold, k=1).any()
    assert pca.explained_variance_ratio_.sum() >= pca_variance
    assert mapping.canonical_track_id.isin(clean.index).all()
    # Diagnose any equal vectors crossing official splits after conservative cleaning.
    remaining_hashes = pd.util.hash_pandas_object(clean, index=False)
    cross_split = clean_metadata["set__split"].groupby(remaining_hashes).nunique().gt(1).sum()
    manifest = {"input_tracks": len(original), "clean_tracks": len(clean),
                "supported_duplicates_removed": len(original) - len(clean),
                "unresolved_tracks_retained": int(duplicates.decision.eq("retain_unresolved").sum()),
                "remaining_duplicate_vectors_beyond_first": int(clean.duplicated().sum()),
                "duplicate_vector_groups_crossing_splits": int(cross_split),
                "constant_features_removed": constant, "clean_features": clean.shape[1],
                "fit_tracks": len(train_ids), "power_transformed_features": len(transformers),
                "training_skew_gt2_before": int(before_skew.abs().gt(2).sum()),
                "training_skew_gt2_after": int(transformed.loc[train_ids].skew().abs().gt(2).sum()),
                "correlation_features": len(retained), "baseline_correlation_features": len(baseline_retained),
                "pca_components": int(pca.n_components_), "baseline_pca_components": int(baseline_pca.n_components_),
                "pca_training_variance": float(pca.explained_variance_ratio_.sum()),
                "config": bundle["config"],
                "versions": {name: version(name) for name in ["numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "joblib"]}}
    (reports / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(run_pipeline(args.root), indent=2))
