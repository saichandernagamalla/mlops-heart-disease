# Dataset provenance and interpretation

This project uses the processed Cleveland subset from the official [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease). UCI describes 303 Cleveland records and a target `num` ranging from 0 to 4; this project uses the binary mapping 0 = no presence and 1-4 = presence. The processed file has 13 selected predictors plus `num`.

The downloader retrieves the official UCI ZIP archive and saves the original processed file locally under `data/raw/`; it is not committed. The cleaned CSV is written to `data/processed/heart.csv`. `?` values are parsed as missing. Median imputation is applied to numeric fields and most-frequent imputation to categorical codes inside each training fold. This avoids fitting imputers on validation folds.

Feature set: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal. The coded fields are one-hot encoded; numeric fields are median-imputed and standardized. Cite UCI dataset DOI 10.24432/C52P4X in submitted work and check the current UCI reuse terms before redistribution. This historical, small subset is not contemporary clinical evidence.
