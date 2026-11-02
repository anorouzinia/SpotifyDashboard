# Music similarity: FMA exploration and preprocessing

DVA project investigating similarity based on audio characteristics. Current work
includes data exploration, conservative cleaning, and comparisons of scaling
and feature reduction. Recommendation models and a dashboard are deferred.

## Local setup

From this repository directory (the inner `spotifydashboard/` folder):

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The requirements record the versions used for the exploration on Python 3.14.
In VS Code, open `notebooks/explore_fma.ipynb`, select this `.venv` as the
Jupyter kernel, and run all cells. The notebook works from the repository root
or `notebooks/` and writes diagnostic CSVs and a Markdown report to `reports/`.
It can also be executed from the repository root with:

```sh
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/explore_fma.ipynb
```

## Data and layout

Download and extract the `fma_metadata` and `fma_small` archives from the
[FMA dataset repository](https://github.com/mdeff/fma) into `data/`:

```text
data/
  fma_metadata/       # tracks.csv, features.csv, genres.csv, echonest.csv, etc.
  fma_small/
    000/000010.mp3    # six-digit ID, directory = first three digits
    ...
notebooks/
  explore_fma.ipynb   # executed Phase 1 audit and plots
  02_preprocessing.ipynb # cleaning and representation comparisons
src/
  exploration.py     # reusable loading and diagnostic functions
  preprocessing.py   # reproducible cleaning, scaling, correlation filtering, PCA
  features.py        # original FMA extraction reference
  utils.py           # original FMA helpers
  usage.ipynb        # original FMA usage reference
reports/             # full diagnostic tables and concise findings
```

Source data and the virtual environment are excluded from Git. Obtain the
archives separately; the notebook never downloads data or opens audio files.
Only `tracks.csv` and `features.csv` are required for this audit. The metadata
and feature CSVs cover more than FMA Small; the loader selects the `small`
subset and aligns by track ID using the original CSV multi-level headers.

## Exploration results

See [the findings report](reports/fma_small_exploration.md) for observed data
quality and recommendations. Complete tables are saved for feature/track
missingness, finite-value distributions, duplicate vectors, and correlation
candidates. The original FMA scripts are preserved; their optional extraction
dependencies are not part of the Phase 1 environment.

## Cleaning and feature representations

Run all cells in `notebooks/02_preprocessing.ipynb` with the same `.venv` kernel,
or run the reusable workflow directly:

```sh
python src/preprocessing.py
python -m unittest discover -s tests -v
```

The workflow consolidates 11 records whose exact feature vectors and normalized
artist/title match. It retains eight conflicting-title tracks for investigation,
leaving 7,989 tracks and 518 original numeric columns. All source IDs are
preserved through metadata and a canonical-ID mapping. No missing-value
imputation or constant-column removal is indicated by the observed data.

Training-fitted StandardScaler provides the baseline. Selective Yeo–Johnson
transforms address skew before standard scaling for an alternative. Pearson
filtering retains 440 named features at an absolute correlation threshold of
0.95; independent PCA retains 197 components for 95% of training variance.
Validation/test reconstruction is reported separately; this target does not
guarantee 95% retained variance on those splits or preserved musical meaning.

Generated files in `data/processed/` are ignored by Git:

| File | Contents |
| --- | --- |
| `features_clean.parquet` | Cleaned rows in original units |
| `features_scaled.parquet` | StandardScaler baseline, all columns |
| `features_transformed_scaled.parquet` | Selective Yeo–Johnson then StandardScaler |
| `features_correlation_filtered.parquet` | Interpretable subset of transformed/scaled features |
| `features_pca.parquet` | PCA alternative on all transformed/scaled features |
| `metadata_all.parquet`, `metadata_clean.parquet` | Metadata separate from numeric matrices |
| `track_mapping.parquet` | All source IDs mapped to retained representatives |
| `transformations.joblib` | Fitted scalers, power transforms, selectors, PCA, fit IDs/config |

Every matrix preserves track IDs and flat feature labels. All fitted choices use
only the official training split. Re-running replaces these generated outputs;
it never changes source CSVs or opens audio. The earlier exploration notebook
remains an audit of the original 8,000-track dataset.

See [the cleaning report](reports/cleaning/cleaning_report.md), diagnostic tables,
and package/configuration manifest in `reports/cleaning/` for decisions and
limitations. The saved representations are alternatives for later analysis;
their musical-similarity quality has not yet been evaluated.
