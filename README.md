# SMMT 2025 UK Car Sales – Data Science Project

This repository contains an end‑to‑end data science project analysing UK new car registrations using SMMT 2025 data, complemented by DfT/DVLA VEH0160 registrations and a simple macroeconomic indicator. The work covers data ingestion, cleaning, exploratory data analysis (EDA), feature engineering, supervised learning (classification and regression), and a time‑series baseline.

The project is part of the MSc Data Science at the University of Hertfordshire.

## Project overview

The core starting dataset is the SMMT `Cars_12_2025.xlsx` file (sheet `CARS_Month_End`), which reports registrations of new cars in the UK by marque with both December and year‑to‑date (YTD) figures for 2024 and 2025.

**Key objectives:**

- Clean and restructure the original SMMT Excel sheet into a modelling‑ready DataFrame.
- Explore market structure, fuel types, manufacturers, and top models.
- Engineer business‑relevant features and build a RandomForest classifier to predict high‑sales manufacturers.
- Extend the analysis with DfT/DVLA quarterly registrations, lagged features, and macroeconomic data and build a RandomForestRegressor on volume.
- Implement a simple pure time‑series model (Prophet) on aggregated registrations as a complementary forecasting baseline.

Supervisor feedback is addressed via: using DVLA registration time series (with lags), avoiding manufacturer as a feature, incorporating macroeconomic context, switching to regression for volume, and adding both hybrid (lags + tree) and pure (Prophet) time‑series approaches.

## Data sources

### SMMT 2025 new car registrations

- File: `Cars_12_2025.xlsx` (sheet `CARS_Month_End`).
- Level: manufacturer (marque), December and YTD registrations for 2024 and 2025.

### DfT/DVLA VEH0160 registrations

- File: `df_VEH0160_UK.csv` (quarterly registrations by body type, make, model, fuel).
- Converted to a long panel at `(BodyType, Make, GenModel, Model, Fuel, YearQuarter)`.
- Lag features (`Registrations_lag1`, `Registrations_lag2`, `Registrations_lag4`) created per model to capture past 1/2/4‑quarter history.

### Macroeconomic indicator

- Simple quarterly series `bank_rate` (e.g. Bank of England interest rate) constructed/loaded and merged by `YearQuarter` into the DVLA panel.

The `src/dsutils.py` module provides reusable functionality for loading data, creating lag features, splitting by time, building RF pipelines, and evaluating regression models.

## Notebook 01 – SMMT 2025 EDA and RF classification

**File:** `notebooks/01_SMMT_2025_EDA_and_RF.ipynb`

### Data and preprocessing

- Unzip `Full-year-car-registrations-2025.zip` and load `Cars_12_2025.xlsx` from sheet `CARS_Month_End`.
- Rename opaque columns to meaningful names:  
  `marque`, `dec_2025`, `dec_share_2025`, `dec_2024`, `dec_share_2024`, `dec_change_pct`,  
  `ytd_2025`, `ytd_share_2025`, `ytd_2024`, `ytd_share_2024`, `ytd_change_pct`.
- Cast numeric columns, drop header/summary rows, and remove the “Total Market” aggregate row.
- Create a binary target:  
  - `High_Sales = 1` if `ytd_2025 ≥ median(ytd_2025)`, else `0`.  
  - `Purchase` mirrors `High_Sales` and is used as the final target (balanced 29/29).

Result: a clean dataset of about 58 manufacturers (after filtering) and around 13 base features.

In all SMMT models the `marque` (manufacturer) column is excluded from the input features so that the model does not simply learn that large brands sell many cars, in line with supervisor guidance.

### Exploratory data analysis

Key EDA components:

- Top‑N manufacturers by YTD 2025 sales (bar charts and labelled values).
- Correlation heatmap of numeric features and a focused feature–target correlation view.
- Narrative describing dataset shape, variables, and obvious patterns (e.g. dominance of large marques and BEV/HEV trends).

A supplementary table summarises 2025 fuel‑type performance (petrol, diesel, HEV, PHEV, BEV) with units, market share, and growth, with supporting pie and bar charts.

### Feature engineering

Based on correlation analysis and supervisor feedback, the following engineered features are added:

- `ytd_growth_rate ≈ (ytd_2025 - ytd_2024) / ytd_2024 × 100`
- `share_change = ytd_share_2025 - ytd_share_2024`
- `sales_momentum = dec_2025 / (ytd_2025 / 12)` (December vs average month)
- `top_quartile` flag for top 25% by `ytd_2025`

These expand the feature space to roughly 17 columns.

### Random Forest classifier

Target: `Purchase` (high vs low manufacturer).

- Train–test split: 46 train, 12 test, balanced classes.
- Baseline `RandomForestClassifier` achieves near‑perfect metrics on the small test set.
- `GridSearchCV` over `n_estimators`, `max_depth`, `max_features`, `min_samples_split`, `min_samples_leaf` using ROC‑AUC as the scoring metric.
- Best configuration yields cross‑validated ROC‑AUC close to 0.99 and a perfect confusion matrix on the 12‑sample test split.

A dedicated markdown section interprets this: the task is almost deterministic on this small sample because YTD and market‑share features nearly fully determine the label, so perfect scores are not strong evidence of generalisation.

## Notebook 02 – SMMT supporting EDA

**File:** `notebooks/01_EDA_SMMT.ipynb`

This notebook contains additional EDA and narrative around the SMMT 2025 dataset, used to structure the business story and plots that complement Notebook 01.

Typical content:

- Extended descriptive statistics and distribution plots for YTD and December volumes.
- Additional visualisations for manufacturer ranking, fuel mix, and growth patterns.
- Business commentary that links numerical findings to the UK car market context.

This notebook is mainly descriptive and supports the report text; the main modelling code remains in Notebooks 01, 03, and 04.

## Notebook 03 – DVLA lagged regression

**File:** `notebooks/02_DVLA_lagged_regression.ipynb`

### DVLA data preparation

- Load `df_VEH0160_UK.csv` with quarterly registrations by body type, make, model, and fuel.
- Separate identifier columns (`BodyType`, `Make`, `GenModel`, `Model`, `Fuel`) from time columns (e.g. `2014 Q3`, …, `2025 Q3`).
- Melt the wide table into a long panel with columns:  
  `BodyType`, `Make`, `GenModel`, `Model`, `Fuel`, `YearQuarter`, `Registrations`.
- Parse `YearQuarter` to an actual quarter‑end `Date` and sort by `(Make, GenModel, Model, Date)`.

### Lag feature engineering

Using a helper such as `dsutils.create_lag_features`:

- Within each `(Make, GenModel, Model)` group, create lagged features for 1, 2, and 4 quarters:  
  `Registrations_lag1`, `Registrations_lag2`, `Registrations_lag4`.
- Save the resulting long panel with lags as `dvla_with_lags_long.csv` in `data/processed/`.

### DVLA RandomForestRegressor

Objective: predict quarterly `Registrations` using only past lags (and optionally a macro feature).

- Load `dvla_with_lags_long.csv` and drop rows where lag features are missing.
- Feature matrix:  
  `X = [Registrations_lag1, Registrations_lag2, Registrations_lag4]`
- Target:  
  `y = Registrations`
- Time‑based train–test split using `train_test_split(..., shuffle=False)` or a custom time split.
- Train `RandomForestRegressor` and evaluate with \(R^2\) and RMSE from `mean_squared_error`.

A markdown section interprets these metrics: past registrations explain a meaningful portion of variance, but not all, and some models have long runs of zeros that make prediction harder.

### Derived high/low volume label

To connect back to a classification framing:

- Compute `threshold = y_train.median()`.
- Define `High_Volume = 1` if `Registrations > threshold` else `0`.

This shows how a binary decision can be derived from a volume regression model rather than being directly trained from a small cross‑section.

### Adding a macro feature

To satisfy the supervisor’s recommendation on macroeconomic context:

- Build a simple quarterly macro DataFrame, e.g. `macro_df = [YearQuarter, bank_rate]`.
- Merge into the DVLA panel on `YearQuarter` to obtain `bank_rate` for each model‑quarter row.
- Extend features to  
  `feature_cols = [Registrations_lag1, Registrations_lag2, Registrations_lag4, bank_rate]`  
  and re‑train the RF regressor.
- Compare \(R^2\)/RMSE and feature importances with and without the macro feature.

A short narrative comments on whether the macro variable adds signal or remains dominated by lag features.

## Notebook 04 – Macro and time‑series models

**File:** `notebooks/03_Macro_and_time_series_models.ipynb`

### Aggregated quarterly time series

- Aggregate the DVLA panel to total quarterly registrations:  
  `ts_df = dvla_with_lags_long.groupby("YearQuarter")["Registrations"].sum()`
- Convert `YearQuarter` to a proper quarterly date (e.g. last day of the quarter) and rename to Prophet‑style `ds` and `y`.

### Prophet time‑series model

- Fit a basic Prophet model without weekly/daily seasonality on the aggregated quarterly series.
- Generate a 4‑quarter‑ahead forecast and visualise forecast vs history.

This provides a pure time‑series forecasting baseline that complements the hybrid lag‑plus‑RandomForest approach.

## How to run

1. Clone the repository and create a Python environment using `requirements.txt`.
2. Place the SMMT Excel file and DVLA CSV into `data/raw/` if not already present.
3. Run the notebooks in order:

   - `01_SMMT_2025_EDA_and_RF.ipynb` – SMMT EDA and RandomForest classifier.
   - `01_EDA_SMMT.ipynb` – Extended SMMT EDA and narrative plots.
   - `02_DVLA_lagged_regression.ipynb` – DVLA lag features and RandomForestRegressor on volume with optional macro.
   - `03_Macro_and_time_series_models.ipynb` – Aggregated time series and Prophet forecast.

The notebooks import shared utilities from `src/` to ensure results are reproducible and code is not duplicated across multiple exploratory runs.

## Repository structure

```text
.
├── data/
│   ├── raw/
│   │   ├── Full-year-car-registrations-2025.zip
│   │   └── df_VEH0160_UK.csv
│   └── processed/
│       ├── smmt_clean.csv
│       └── dvla_with_lags_long.csv
├── notebooks/
│   ├── 01_SMMT_2025_EDA_and_RF.ipynb
│   ├── 01_EDA_SMMT.ipynb
│   ├── 02_DVLA_lagged_regression.ipynb
│   └── 03_Macro_and_time_series_models.ipynb
├── src/
│   ├── dsutils.py
│   ├── data_preparation.py
│   ├── features.py
│   └── model_random_forest.py
├── reports/
│   ├── figures/
│   └── tables/
├── README.md
└── requirements.txt
