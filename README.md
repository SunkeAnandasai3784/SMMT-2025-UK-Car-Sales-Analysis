# SMMT 2025 & DfT VEH0160 – UK Car Sales and Registrations Analysis

Code and notebook for the SMMT 2025 + DfT VEH0160 analysis of UK new car registrations, combining snapshot and panel data to model manufacturer‑level demand.

## Project overview

This project uses two official UK data sources:  
- SMMT car registrations by marque for December and year‑to‑date 2024–2025 (`Cars_12_2025.xlsx`).  
- Vehicle licensing statistics (DfT `VEH0160_UK`) from GOV.UK.  

The SMMT Excel is a 2025 snapshot with 57 manufacturers, used for cross‑sectional analysis and classification of “high vs low” performers.  
The VEH0160 CSV is a quarterly time‑series panel of licensed vehicles by make and fuel type; after reshaping to long format and creating lags, it is used for regression, classification and forecasting models.  
Both datasets come from official organisations (SMMT and UK government / DfT) and measure related but different concepts (new registrations vs licensed stock), so they complement each other.

> Raw SMMT and DfT files are not included in this repository. Users should download them from the official SMMT site and the GOV.UK vehicle‑licensing statistics page.

## Data sources

- **SMMT new car registrations (2025 snapshot)**  
  - Provider: Society of Motor Manufacturers and Traders (SMMT).  
  - File: `Cars_12_2025.xlsx`.  
  - Role: High‑level manufacturer snapshot used for EDA and a binary `Purchase` label based on 2025 year‑to‑date registrations and market share.  

- **DfT VEH0160 – Licensed vehicles by body type and fuel**  
  - Provider: UK Department for Transport (DfT), table VEH0160 “Licensed vehicles by body type and fuel”.  
  - File: `df_VEH0160_UK.csv` (downloaded from GOV.UK and loaded in the notebook).  
  - Role: Reshaped into a quarterly panel by make and date, with lagged licences and year‑on‑year growth features for modelling manufacturer‑level dynamics.  

## Methods

### SMMT 2025 manufacturer snapshot – EDA and classification

- Load and clean the SMMT Excel into a manufacturer‑level table (`smmt_clean`) with December and year‑to‑date volumes, shares and growth rates.  
- Build a `High_Sales` / `Purchase` label from `ytd_2025` (top 40% vs bottom 40%, middle 20% dropped), giving a balanced 23/23 split for modelling.  
- Run EDA on: top‑10 manufacturers (2024 vs 2025 volumes and growth), fuel‑type distribution (Petrol, Diesel, HEV, PHEV, BEV) using published SMMT totals, and a correlation heatmap of numeric features vs `Purchase`.  
- Train Logistic Regression, Random Forest and Gradient Boosting classifiers on the SMMT snapshot, showing that with only 57 observations all three can achieve apparent ROC‑AUC of 1.0, highlighting the overfitting risk.  

### DfT VEH0160 panel – reshaping and feature engineering

- Filter VEH0160 to cars only and melt the wide quarterly columns (`2015 Q1` … `2025 Q3`) into a long table with `BodyType`, `Make`, `GenModel`, `Model`, `Fuel`, `period`, `Licences`.  
- Split `period` into `Year` and `Quarter`, build a quarterly `date` index, sort by `Make` and date, and aggregate to a `Make`–`date` panel (`veh_panel`).  
- Create time‑series features: `Licences_lag1`, `Licences_lag2`, `Licences_lag4` (within‑make lags) and `Licences_yoy` (year‑on‑year growth vs four quarters ago).  
- Drop rows with missing lags, ending with 5,680 make–quarter rows and, after cleaning infinities/NaNs, 4,754 rows with complete feature vectors.  

### Random Forest panel model (regression and tuned classifier)

- Regression setup: predictors = `[Licences_lag1, Licences_lag2, Licences_lag4, Licences_yoy]`, target = `Licences`.  
- Train a `RandomForestRegressor` (300 trees) with an 80/20 split; the model achieves roughly \(R^2 \approx 0.99\), MAE around 300 and RMSE a little above 1,100 licences on the test set.  
- Feature importance shows that `Licences_lag2` and `Licences_lag4` dominate, indicating strong persistence in registrations; `Licences_yoy` and `Licences_lag1` play a smaller role.  

- Expanded classification pipeline on the panel:  
  - Use a balanced binary target from panel features (around 246 observations, roughly 126 vs 120) with 5 input features.  
  - Perform a stratified 75/25 split (train 184, test 62), replacing the earlier 45/12 split.  
  - Apply `GridSearchCV` (5‑fold CV) over RandomForest hyper‑parameters (n_estimators, max_depth, max_features, min_samples_split, min_samples_leaf).  
  - Best model example: `n_estimators=50`, `max_depth=5`, `max_features='sqrt'`, `min_samples_split=2`, `min_samples_leaf=1`, with CV ROC‑AUC and test ROC‑AUC of 1.0 and low log loss, plus clean confusion matrix.  

### Time‑series baseline: VAUXHALL SARIMA example

- Select VAUXHALL from the panel (41 quarterly observations between 2015 Q3 and 2025 Q3).  
- Fit a seasonal ARIMA model such as `SARIMA(1,1,1)×(0,1,1,4)` on the first 80% of the series and forecast the remaining 20%.  
- The SARIMA model’s test RMSE is in the mid‑thousands of licences, illustrating a traditional time‑series benchmark against which the Random Forest panel approach can be compared.  

### Before vs after – addressing data size, model focus and tuning

The notebook contrasts the original SMMT‑only setup with the expanded DfT panel and tuned Random Forest classifier:

| Aspect                | SMMT‑only (before)           | Expanded panel (after)                   |
|-----------------------|-----------------------------|------------------------------------------|
| Total observations    | 57 manufacturers            | 246 make–quarter rows                    |
| Train / test sizes    | 45 / 12                     | 184 / 62                                 |
| Data source           | SMMT snapshot only          | DfT VEH0160 panel (2014–2025)            |
| Models                | LR, RF, GB (all untuned)    | Random Forest (focused, tuned)           |
| Hyper‑parameter tuning| None                        | GridSearchCV (5‑fold CV)                 |
| CV / test ROC‑AUC     | ~1.0 (small‑n, overfit‑prone)| 1.0 on a much larger panel              |

This directly responds to earlier concerns about small sample size, lack of tuning and diffuse model focus by moving to a richer panel and concentrating on one tuned Random Forest model.

## How to run

1. **Clone the repository**

```bash
git clone https://github.com/SunkeAnandasai3784/SMMT-2025-UK-Car-Sales-Analysis.git
cd SMMT-2025-UK-Car-Sales-Analysis
pip install -r requirements.txt

SMMT-2025-UK-Car-Sales-Analysis/
├── notebooks/
│   └── 01_SMMT_2025_EDA_and_Models.ipynb
├── data/
│   ├── raw/        # Cars_12_2025.xlsx, VEH0160 CSV (not tracked)
│   └── processed/  # cleaned / panel-ready CSVs (optional)
├── src/
│   ├── preprocessing/
│   ├── models/
│   └── viz/
├── reports/
│   └── figures/    # feature_importance.png, roc_curve_tuned.png, etc.
└── README.md

Acknowledgements
Society of Motor Manufacturers and Traders (SMMT) for UK new car registration statistics.

UK Department for Transport for VEH0160 vehicle‑licensing statistics.

Project completed as part of MSc studies at the University of Hertfordshire.
