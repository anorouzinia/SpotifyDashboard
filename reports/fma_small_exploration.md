# FMA Small exploration findings

## Scope and alignment

- Full metadata: 106,574 tracks; full feature table: 106,574 tracks.
- Small subset: 8,000 unique positive track IDs, aligned with 518 numerical audio columns in 11 families and seven summary statistics.
- Missing Small feature rows: 0; duplicate metadata/feature IDs: 0/0. No unmatched IDs across the full metadata and feature tables.
- Genre labels contain eight groups of 1,000 tracks each; they remain metadata and are excluded from the numeric matrix.
- Families (column counts): chroma_cens (84), chroma_cqt (84), chroma_stft (84), mfcc (140), rmse (7), spectral_bandwidth (7), spectral_centroid (7), spectral_contrast (49), spectral_rolloff (7), tonnetz (42), zcr (7).

## Data quality and distributions

- NaNs: 0; positive infinity: 0; negative infinity: 0.
- Constant columns: 0; entirely invalid columns: 0.
- Exact repeated numeric vectors beyond the first: 17; all involved tracks: 30. See duplicate_vectors.csv for IDs, groups, and metadata. Distinct IDs alone do not prove that the audio is distinct.
- Basic summary consistency violations: 0 cells (negative standard deviations or means/medians outside min/max, tolerance 1e-8).
- Absolute skew above 2: 153 of 518 columns. This includes extreme kurtosis tails and highly concentrated chroma maxima; skew is a diagnostic, not evidence of corrupt recordings.
- At least 99% of finite observations share one value in 24 columns. Inspect these for saturation and limited information before deciding whether to remove them.
- Feature standard deviations range from 0.00296148 to 1241.25; raw scales differ substantially.
- Unique feature pairs with absolute Pearson correlation >= 0.95: 80. Relationship counts: same component, different statistic: 74, different families: 4, different components, same family: 2. Correlation is computed across tracks between distinct family/statistic/component columns; mean-versus-median comparisons of one descriptor test summary redundancy. Self-pairs are excluded. See high_correlations.csv; the count is not the number of removable columns.

## Recommended decisions for later phases

1. No missing-value imputation or constant-column removal is indicated by this Small-subset audit. Retain the raw numeric matrix and its track ID index unchanged.
2. Investigate equal-vector track groups using titles, artists, and later targeted audio checks before deduplicating; do not automatically discard songs.
3. Review nearly invariant columns and long-tailed summaries individually. Preserve unusual songs; investigate saturation and robust or skew-reducing transformations rather than blanket outlier deletion. Signed MFCC/tonnetz/skew values rule out a universal logarithm.
4. Compare StandardScaler as a baseline with robust scaling or suitable per-column transforms after reviewing distributions. Fit transformations on training data only if evaluating held-out results, then apply the same fitted transformation to validation/test data.
5. Explore correlation filtering at 0.95 with an explicit deterministic policy that preserves interpretable summaries. Check sensitivity to outliers or rank correlation before removal.
6. Subsequently compare PCA at 95% explained variance against cleaned/scaled originals and correlation-filtered features. PCA is not performed here and is not assumed preferable.
7. Metadata, including genre/artist/title, stays separate from acoustic features. Record feature order and ID alignment with any future saved matrices and fitted transformations.

## Audio structure and limits

Expected path: data/fma_small/<first-three-digits>/<six-digit-track-id>.mp3. The initial shallow inspection found 156 numbered directories and sampled three entries from directory 000. No recursive audio enumeration, audio decoding, full existence checks, or readability checks were performed. Sampling-rate changes are deferred because this phase uses existing features.

## Reproduction and artifacts

Run all cells in notebooks/explore_fma.ipynb with the project environment. Feature names use family__statistic__component; feature_dictionary.csv maps labels to the original levels. CSV files in reports/ contain complete per-feature/per-track quality counts, descriptive statistics, duplicate-vector groups, and correlation candidates. Percentages use all tracks/columns for missingness, and finite observations for distribution diagnostics. No raw files or precomputed feature values were changed.
