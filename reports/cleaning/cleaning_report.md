# FMA Small cleaning and representation report

## Cleaning decisions

- Input: 8,000 tracks and 518 numeric feature columns.
- Removed 11 records with exactly equal feature vectors AND matching normalized artist/title. Artist-prefixed title variants are normalized. This is metadata-supported consolidation, not audio identity verification.
- Retained 7,989 tracks; 8 conflicting-title tracks remain flagged across the Moon Veil and Janneh groups. There are 6 repeated vectors beyond their first occurrence among these retained rows.
- All original metadata and the source-to-canonical mapping are saved separately, preserving all 8,000 source IDs. Raw files remain unchanged.
- No imputation, constant-column removal, clipping, or outlier-row removal was necessary. Numerical features remain separate from metadata. The cleaned matrix retains raw units.
- Remaining exact-vector groups crossing official splits: 0.

## Scaling and skew

- All scalers, power transforms, correlation decisions, and PCA fit only the 6,392 retained training rows; validation/test receive those fitted transformations.
- StandardScaler provides a full-feature baseline; it changes scale but not skew.
- Selective Yeo–Johnson transforms were adopted for 129 columns. Eligible columns have training |skew| > 2, >10 unique values, and modal fraction <99%; accept only improved training absolute skew and numerically finite outputs.
- Training columns with |skew| > 2 decreased from 153 to 38. Residual skew and saturated columns remain visible in transform_decisions.csv. Reduced skew does not prove better musical similarity.

## Reduction comparisons

- Correlation filtering retained 440 of 518 transformed/scaled columns at |Pearson r| >=0.95. Standard-scaled baseline filtering retained 440 columns, potentially with different membership.
- Prefer mean, median, std, min, max, skew, then kurtosis; each dropped column maps to a surviving correlated representative. No retained training pair exceeds the threshold. The input feature order is preserved.
- Independent PCA on all transformed/scaled columns retained 197 components for 95.01% of training variance; baseline standard-scaled PCA required 205 components for at least 95% variance.
- PCA is not chained after correlation filtering. Loadings and split-specific reconstruction metrics are saved. Variance is not a measure of preserved musical meaning.

```text
     split  tracks  mean_squared_reconstruction_error  retained_energy_fraction_about_training_mean
  training    6392                           0.049851                                      0.950149
validation     800                           0.110865                                      0.916764
      test     797                           0.124823                                      0.890752
```

## Artifacts and reproduction

Run all cells in notebooks/02_preprocessing.ipynb or run `.venv/bin/python src/preprocessing.py` from the repository root. Generated data in data/processed/ is ignored by Git; reports/cleaning/ holds the small audit artifacts. Re-running replaces generated outputs only.

- features_clean.parquet: cleaned rows, raw units, all numeric columns.
- features_scaled.parquet: StandardScaler baseline.
- features_transformed_scaled.parquet: selective Yeo–Johnson followed by StandardScaler.
- features_correlation_filtered.parquet: interpretable subset of the transformed/scaled matrix.
- features_pca.parquet: PCA alternative from the full transformed/scaled matrix.
- metadata_all.parquet, metadata_clean.parquet, track_mapping.parquet: metadata and original-to-canonical IDs.
- transformations.joblib: fitted transformations, selected columns, training IDs, PCA alternatives, and configuration. manifest.json records package versions and counts.

## Recommended use and remaining limits

Start with the 440-column correlation-filtered representation for interpretable analysis and compare the PCA representation when compactness matters. Keep the full baseline as a control. Investigate the eight conflicting-title tracks using targeted audio checks before removing them; this workflow never opens audio. Review near-invariant columns and residual skew before treating standardized distances as musically meaningful. These are preprocessing alternatives, not evaluated recommendation models.
