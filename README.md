SMMT 2025 UK Car Sales – Data Science Project
This repository contains an end‑to‑end data science project analysing UK new car registrations using SMMT 2025 data, complemented by DfT/DVLA VEH0160 registrations and a simple macroeconomic indicator. The work covers data ingestion, cleaning, exploratory data analysis (EDA), feature engineering, supervised learning (classification and regression), and a time‑series baseline.

The project is part of the MSc Data Science at the University of Hertfordshire.

Project Overview
The core starting dataset is the SMMT “Cars_12_2025.xlsx” file (sheet CARS_Month_End), which reports registrations of new cars in the UK by marque with both December and year‑to‑date (YTD) figures for 2024 and 2025.

Key objectives:

Clean and restructure the original SMMT Excel sheet into a modelling‑ready DataFrame.

Explore market structure, fuel types, manufacturers, and top models.

Engineer business‑relevant features and build a RandomForest classifier to predict high‑sales manufacturers.

Extend the analysis with DfT/DVLA quarterly registrations, lagged features, and macroeconomic data and build a RandomForestRegressor on volume.

Implement a simple pure time‑series model (Prophet) on aggregated registrations as a complementary forecasting baseline.
​

Data Sources
SMMT 2025 new car registrations

File: Cars_12_2025.xlsx (sheet CARS_Month_End).

Level: manufacturer (marque), December and YTD registrations for 2024 and 2025.

DfT/DVLA VEH0160 registrations

File: df_VEH0160_UK.csv (quarterly registrations by body type, make, model, fuel).
​

Converted to a long panel at (BodyType, Make, GenModel, Model, Fuel, YearQuarter).
​

Lag features (Registrations_lag1, Registrations_lag2, Registrations_lag4) created per model to capture past 1/2/4‑quarter history.

Macroeconomic indicator (example)

Simple quarterly series bank_rate (e.g. Bank of England interest rate) constructed/loaded and merged by YearQuarter into the DVLA panel.
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
│   ├── 02_DVLA_lags_and_regression.ipynb
│   └── 03_Macro_and_time_series_models.ipynb
├── src/
│   ├── dsutils.py                 # shared helpers: loading, lag creation, time splits, RF pipelines
│   ├── data_preparation.py        # (optional) SMMT-specific cleaning logic
│   ├── features.py                # (optional) feature engineering helpers
│   └── model_random_forest.py     # (optional) model wrapper
├── reports/
│   ├── figures/                   # exported plots for the dissertation
│   └── tables/                    # CSVs with final metrics / summaries
├── README.md
└── requirements.txt
The src/dsutils.py module provides reusable functionality for loading data, creating lag features, splitting by time, building RF pipelines, and evaluating regression models.
​

Notebook 01 – SMMT 2025 EDA and RF Classification
File: Anandasai Sunke DS PROJECT.ipynb

Data and preprocessing
Steps:

Unzip Full-year-car-registrations-2025.zip and load Cars_12_2025.xlsx from CARS_Month_End.
​

Rename opaque columns to meaningful names:
marque, dec_2025, dec_share_2025, dec_2024, dec_share_2024, dec_change_pct,
ytd_2025, ytd_share_2025, ytd_2024, ytd_share_2024, ytd_change_pct.
​

Cast numeric columns, drop header/summary rows, and remove the “Total Market” aggregate row.
​

Create a binary target:

High_Sales = 1 if ytd_2025 ≥ median YTD 2025 units, else 0.
​

Purchase mirrors High_Sales and is used as the final target (balanced 29/29).
​

Result: a clean dataset of 58 manufacturers (51 rows after filtering) and ~13 features.
​

Exploratory data analysis
Key EDA components:

Top‑N manufacturers by YTD 2025 sales (bar charts and labelled values).

Correlation heatmap of numeric features and a focused feature–target correlation view.
​

Short narrative describing dataset shape, variables, and obvious patterns (e.g. dominance of large marques).
​

A supplementary table summarises 2025 fuel‑type performance (petrol, diesel, HEV, PHEV, BEV) with units, market share, and growth, with supporting pie and bar charts.
​

Feature engineering
Based on correlation analysis and supervisor feedback:

ytd_growth_rate ≈ 
(
y
t
d
2
025
−
y
t
d
2
024
)
/
y
t
d
2
024
×
100
(ytd 
2
 025−ytd 
2
 024)/ytd 
2
 024×100.
​

share_change = ytd_share_2025 - ytd_share_2024.
​

sales_momentum = dec_2025 / (ytd_2025 / 12) as a simple December vs average month measure.
​

top_quartile flag for top 25% by ytd_2025.
​

These expand the feature space to ~17 columns.
​

Random Forest classifier
Target: Purchase (high vs low manufacturer).
​

Train–test split: 46 train, 12 test, balanced classes.
​

Baseline RandomForestClassifier (with reasonable defaults) achieves near‑perfect metrics on the small test set.
​

GridSearchCV over n_estimators, max_depth, max_features, min_samples_split, min_samples_leaf using ROC‑AUC as the scoring metric.
​

Best configuration (e.g. n_estimators=100, max_depth=10) yields cross‑validated ROC‑AUC around 0.99 and perfect confusion matrix on the 12‑sample test split.
​

A dedicated markdown section interprets this: the task is almost deterministic on this small sample because YTD/market‑share features nearly fully determine the label, so perfect scores are not strong evidence of generalisation.
​

Notebook 02 – DVLA Lags and Regression
File: notebooks/02_DVLA_lags_and_regression.ipynb

DVLA data preparation
Load df_VEH0160_UK.csv with quarterly registrations by body type, make, model, and fuel.
​

Separate identifier columns (BodyType, Make, GenModel, Model, Fuel) from time columns (e.g. 2014 Q3, …, 2025 Q3).
​

Melt the wide table into a long panel with columns:
BodyType, Make, GenModel, Model, Fuel, YearQuarter, Registrations.
​

Parse YearQuarter to an actual quarter‑end Date and sort by (Make, GenModel, Model, Date).
​

Lag feature engineering
Using dsutils.createlagfeatures:

Within each (Make, GenModel, Model) group, create lagged features for 1, 2, and 4 quarters:
Registrations_lag1, Registrations_lag2, Registrations_lag4.
​

Save the resulting long panel with lags as dvla_with_lags_long.csv in data/processed/.
​

DVLA RandomForestRegressor
Objective: predict quarterly Registrations using only past lags (and optionally a macro feature).

Load dvla_with_lags_long.csv and drop rows where lag features are missing.
​

Feature matrix: X = [Registrations_lag1, Registrations_lag2, Registrations_lag4].
​

Target: y = Registrations.
​

Time‑based train–test split using train_test_split(..., shuffle=False) or a custom time split.
​

Train RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1) and evaluate:

R
2
R 
2
  around 0.50.

RMSE computed from mean_squared_error.
​

A markdown section interprets these metrics: past registrations explain a meaningful portion of variance, but not all, and some models have long runs of zeros that make prediction harder.
​

Derived high/low volume label
To connect back to a classification framing:

Compute median on the training target:
threshold = y_train.median().

Define: High_Volume = 1 if Registrations > threshold else 0.
​

This shows how a binary decision can be derived from a volume regression model rather than being directly trained from a small cross‑section.
​

Adding a macro feature
To satisfy the supervisor’s recommendation on macroeconomic context:

Build a simple quarterly macro DataFrame, e.g.:
macro_df = [YearQuarter, bank_rate].
​

Merge into DVLA panel on YearQuarter to obtain bank_rate for each model‑quarter row.
​

Extend features to feature_cols = [Registrations_lag1, Registrations_lag2, Registrations_lag4, bank_rate] and re‑train the RF regressor.
​

Compare 
R
2
R 
2
 /RMSE and feature importances with and without the macro feature.
​

A short narrative comments on whether the macro variable adds signal or remains dominated by lag features.
​

Notebook 03 – Macro and Time-Series Models
File: notebooks/03_Macro_and_time_series_models.ipynb
​

Aggregated quarterly time series
Aggregate DVLA panel to total quarterly registrations:
ts_df = dvla_with_lags_long.groupby("YearQuarter")["Registrations"].sum().
​

Convert YearQuarter to a proper quarterly date (e.g. last day of the quarter) and rename to Prophet‑style ds and y.
​

Prophet time-series model
Fit a basic Prophet model without weekly/daily seasonality (appropriate for quarterly data):
​

m = Prophet()

m.fit(ts_df[["ds", "y"]])

Create a small forecast horizon (e.g. 4 additional quarters) with make_future_dataframe(periods=4, freq="Q") and generate forecast = m.predict(future).
​

Plot historical vs forecast and briefly summarise trend and any detected seasonality.
​

A markdown section explains how this pure time‑series approach differs from the RandomForestRegressor:

Prophet uses only time information and learns trend/seasonality.

RandomForestRegressor uses lagged registrations (and optionally macro), so it can leverage more structured features.
​

Technologies Used
Python: pandas, numpy.

Visualisation: matplotlib, seaborn.

Machine Learning: scikit‑learn (RandomForestClassifier, RandomForestRegressor, GridSearchCV, metrics).

Time series: Prophet for quarterly forecasts.
​

Environment: Google Colab / Jupyter, with dsutils.py providing reusable utilities.

How to Run
Clone the repository:

bash
git clone https://github.com/SunkeAnandasai3784/SMMT-2025-UK-Car-Sales-Analysis.git
cd SMMT-2025-UK-Car-Sales-Analysis
Create a virtual environment (optional but recommended) and install dependencies:

bash
pip install -r requirements.txt
Place raw data files:

Full-year-car-registrations-2025.zip into data/raw/.

df_VEH0160_UK.csv into data/raw/ (or adjust paths in the notebooks).

Run notebooks in order:

01_SMMT_2025_EDA_and_RF.ipynb – SMMT cleaning, EDA, and RandomForest classifier.

02_DVLA_lags_and_regression.ipynb – DVLA long + lags, RF regression, macro feature, derived binary label.

03_Macro_and_time_series_models.ipynb – aggregated quarterly series and Prophet time‑series forecast.
​

You can use Jupyter, VS Code, or Google Colab to open and run these notebooks.

Interpretation and Limitations
The SMMT manufacturer‑level dataset is small (around 58 rows), and the YTD/market‑share features almost fully determine the high vs low label, leading to near‑perfect classifier metrics but limited evidence of generalisation.
​

The DVLA panel and lag features provide a richer setting for regression and time‑series analysis, but many model‑quarters have zero registrations, which can make predictions and evaluation noisier.

The macroeconomic integration here is deliberately simple (one indicator) to illustrate the concept without overcomplicating the pipeline.
​

Future Work
Planned or possible extensions:

Add more macroeconomic series (e.g. fuel prices, consumer confidence) and compare models with different macro feature sets.
​

Explore more advanced time‑series models (SARIMA, LSTM) if the time depth and data quality support them.

Investigate manufacturer‑ or segment‑specific models (e.g. separate models for BEVs vs ICE, or for top N marques).

Package the key steps (loading, features, models) into a small Python package or CLI for easier reuse.

Author
Anandasai Sunke
MSc Data Science, University of Hertfordshire


