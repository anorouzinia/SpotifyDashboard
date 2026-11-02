"""Read-only exploration of FMA precomputed features; never loads audio."""

from pathlib import Path

import numpy as np
import pandas as pd


def flatten_labels(index):
    """Join MultiIndex levels without shortening their meaning or losing order."""
    if not isinstance(index, pd.MultiIndex):
        return index.copy()
    labels = pd.Index(["__".join(map(str, key)) for key in index], name="feature_name")
    if not labels.is_unique:
        raise ValueError("Flattened labels are ambiguous; choose another separator")
    return labels


def readable_table(frame):
    """Return a display/export copy with flat row and column labels."""
    result = frame.copy()
    result.columns = flatten_labels(frame.columns)
    result.index = flatten_labels(frame.index)
    return result


def load_small(metadata_dir):
    """Use the CSV header conventions in utils.load without its audio dependencies.

    Keep metadata as supplied: no date/category/list parsing is needed for this
    numeric audit. Fail on ambiguous IDs instead of silently duplicating joins.
    """
    metadata_dir = Path(metadata_dir)
    tracks = pd.read_csv(metadata_dir / "tracks.csv", index_col=0, header=[0, 1])
    features = pd.read_csv(metadata_dir / "features.csv", index_col=0, header=[0, 1, 2])
    for name, frame in [("tracks", tracks), ("features", features)]:
        if frame.index.hasnans or not frame.index.is_unique:
            raise ValueError(f"{name}: missing or duplicate track IDs")
        if not pd.api.types.is_integer_dtype(frame.index) or (frame.index <= 0).any():
            raise ValueError(f"{name}: expected positive integer track IDs")
        if not frame.columns.is_unique:
            raise ValueError(f"{name}: duplicate column labels")
    small = tracks.loc[tracks[("set", "subset")].eq("small")].copy()
    missing = small.index.difference(features.index)
    if len(missing):
        raise ValueError(f"Small tracks missing feature rows: {missing.tolist()}")
    aligned = features.loc[small.index].copy()
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in aligned.dtypes):
        raise ValueError("Non-numeric audio feature columns")
    assert small.index.equals(aligned.index)
    inventory = pd.Series({
        "metadata_tracks": len(tracks), "all_feature_tracks": len(features),
        "small_tracks": len(small), "numeric_features": aligned.shape[1],
        "small_missing_feature_rows": len(missing),
        "feature_ids_without_metadata": len(features.index.difference(tracks.index)),
        "metadata_ids_without_features": len(tracks.index.difference(features.index)),
        "metadata_duplicate_ids": int(tracks.index.duplicated().sum()),
        "feature_duplicate_ids": int(features.index.duplicated().sum()),
    }, name="count")
    return small, aligned, inventory


def quality_tables(features):
    """Count NaN and each infinity separately, preserving raw values and IDs."""
    values = features.to_numpy(dtype=float)
    nan = np.isnan(values)
    pos = np.isposinf(values)
    neg = np.isneginf(values)
    invalid = ~np.isfinite(values)
    finite = features.where(np.isfinite(values))  # analysis view only
    by_feature = pd.DataFrame({
        "nan": nan.sum(axis=0), "positive_inf": pos.sum(axis=0),
        "negative_inf": neg.sum(axis=0), "nonfinite": invalid.sum(axis=0),
        "missing_pct": 100 * nan.mean(axis=0),
        "nonfinite_pct": 100 * invalid.mean(axis=0),
        "finite_unique": finite.nunique(), "finite_std": finite.std(),
    }, index=features.columns)
    by_track = pd.DataFrame({
        "nan": nan.sum(axis=1), "positive_inf": pos.sum(axis=1),
        "negative_inf": neg.sum(axis=1), "nonfinite": invalid.sum(axis=1),
        "missing_pct": 100 * nan.mean(axis=1),
        "nonfinite_pct": 100 * invalid.mean(axis=1),
    }, index=features.index)
    return finite, by_feature, by_track


def distribution_table(finite):
    """Describe finite observations; IQR flags are diagnostics, not removals."""
    result = finite.describe(percentiles=[.01, .25, .5, .75, .99]).T
    result["skew"] = finite.skew()
    result["excess_kurtosis"] = finite.kurt()
    q1, q3 = finite.quantile(.25), finite.quantile(.75)
    iqr = q3 - q1
    flags = finite.lt(q1 - 3 * iqr) | finite.gt(q3 + 3 * iqr)
    result["outside_3iqr_pct"] = 100 * flags.sum() / finite.notna().sum()
    result["iqr"] = iqr
    result["modal_fraction"] = finite.apply(
        lambda column: column.value_counts(normalize=True).iloc[0]
        if column.notna().any() else np.nan
    )
    return result


def correlation_pairs(finite, threshold=.95):
    """Return all upper-triangle Pearson pairs above an absolute threshold.

    Each column is one family/statistic/component combination, and tracks are
    observations. Pandas uses pairwise complete finite observations. Exclude
    self-pairs and mirrored pairs; constant columns have undefined correlation.
    """
    corr = finite.corr(min_periods=3)
    row, col = np.where(np.triu(corr.abs().to_numpy() >= threshold, k=1))
    pairs = pd.DataFrame({
        "feature_a": flatten_labels(finite.columns)[row],
        "feature_b": flatten_labels(finite.columns)[col],
        "relationship": [
            "different families" if a[0] != b[0] else
            "same component, different statistic" if a[2] == b[2] else
            "different components, same family"
            for a, b in zip(finite.columns[row], finite.columns[col])
        ],
        "pearson_r": corr.to_numpy()[row, col],
        "paired_tracks": [int((finite.iloc[:, i].notna() & finite.iloc[:, j].notna()).sum())
                          for i, j in zip(row, col)],
    })
    pairs["absolute_r"] = pairs["pearson_r"].abs()
    return corr, pairs.sort_values("absolute_r", ascending=False).reset_index(drop=True)
