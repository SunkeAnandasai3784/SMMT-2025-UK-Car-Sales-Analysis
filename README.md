# Predicting Strong‑Growth UK Car Manufacturers Using DfT VEH0160

This project analyses new car registrations in the UK using the **DfT VEH0160** dataset, which reports vehicles registered for the first time by body type, make, generic model and model.[web:24]  
I focus on **Cars** and build **Random Forest** and **XGBoost** classifiers that predict whether a manufacturer experiences **strong year‑on‑year growth** in registrations in a given quarter.

## Project structure

- `01_SMMT_2025_EDA_and_Models.ipynb` – main notebook with data loading, manufacturer–quarter panel creation, feature engineering, exploratory data analysis and model training/evaluation.  
- `data/df_VEH0160_UK.csv` – DfT “Vehicles registered for the first time” dataset (not committed if too large).  
- `presentation/Anandasai-Sunke-24025759-Presentation.pptx` – final project presentation aligned with the MSc report and this notebook.

## Data source

The project uses one official UK dataset:

- **DfT VEH0160 – Vehicles registered for the first time**  
  - Source: UK Department for Transport, Vehicle Licensing Statistics data tables.[web:24]  
  - Scope used here: `BodyType == "Cars"` only.  
  - Level (after processing): Make–quarter (aggregated from make, generic model and model).  
  - Time span used here: 2014 Q1 to 2025 Q3.  

No personal or individual‑level data is used; all records are aggregated counts by manufacturer and time period.

## Aim

The aim of the project is to:

- Detect **strong‑growth quarters** for each car manufacturer in the UK, defined as more than 10% year‑on‑year growth in registrations.  
- Build **interpretable yet realistic** tree‑based models (Random Forest and XGBoost) that avoid feature leakage and report credible metrics (test ROC‑AUC around 0.95 for XGBoost) rather than unrealistic near‑perfect scores.

## Methodology

### 1. Pre‑processing & panel creation

1. Read `df_VEH0160_UK.csv` with the correct encoding.  
2. Filter to `BodyType == "Cars"`.  
3. Melt the wide quarterly columns into long format so each row is a `(Make, Quarter, Registered)` combination.  
4. Extract `Year` and `QuarterNo` and create a `Date` at the end of each quarter.  
5. Aggregate to a **Make–Date panel**: one row per manufacturer and quarter with total new registrations.  

This results in a manufacturer–quarter panel with 13,756 observations after filtering and cleaning.

### 2. Feature engineering

On the Make–Date panel, three compact, growth‑related features are constructed:

- `QoQ_Growth_Rate` – quarter‑on‑quarter percentage change in registrations (short‑term momentum).  
- `Volatility` – standard deviation of registrations over the previous four quarters (stability vs instability).  
- `Market_Share` – each manufacturer’s share of total car registrations in that quarter (competitive position).

These features replace highly correlated raw lags and summarise recent performance without severe multicollinearity.

### 3. Binary target (strong growth)

The binary target is defined as:

- Compute year‑on‑year growth rate using registrations four quarters apart.  
- Set `Target = 1` if `YoY_Growth_Rate > 0.10` (more than 10% year‑on‑year growth), otherwise `Target = 0`.  

This means the model predicts **strong growth**, not just any positive change, and produces an imbalanced dataset with about 14.7% strong‑growth quarters.

### 4. Feature matrix

The final feature matrix uses only the engineered inputs:

- `X = [QoQ_Growth_Rate, Volatility, Market_Share]`  
- `y = Target`  

All infinities and missing values are cleaned and rows with incomplete features are dropped before modelling.

### 5. Train–validation–test split

A **temporal split** is used to respect time order:

- Training/validation: 2014 Q1 – 2022 Q4  
- Test: 2023 Q1 – 2025 Q3  

Within the training period, a 70/30 split is used for validation during hyperparameter tuning. Splits are stratified on `Target` to preserve the class imbalance structure.

## Models

Two main model families are used: `RandomForestClassifier` and XGBoost’s `XGBClassifier`.

### Random Forest (baseline)

An initial baseline Random Forest is trained with:

- Reasonable default hyperparameters and `class_weight="balanced"` to account for class imbalance.  
- Temporal, stratified split for training/validation.

### Tuned Random Forest and XGBoost

Both models are tuned using grid search with cross‑validation (on the training period), optimising ROC‑AUC. Example grids:

- **XGBoost**  
  - `max_depth: [3, 6, 8]`  
  - `learning_rate: [0.05, 0.1]`  
  - `n_estimators: [200, 300]`  
  - `subsample: [0.8, 1.0]`  
  - `colsample_bytree: [0.8, 0.9]`  

- **Random Forest**  
  - `n_estimators: [200, 400]`  
  - `max_depth: [20, 35]`  
  - `min_samples_split: [2, 4]`  
  - `min_samples_leaf: [1, 2]`  

The tuned XGBoost model is used as the final model in the MSc report.

## Results

### Tuned XGBoost (final model)

- Test ROC‑AUC ≈ **0.950**  
- Test accuracy ≈ **0.89**  
- High recall for the strong‑growth class (around the high‑80% range), meaning most strong‑growth quarters are correctly identified.

### Tuned Random Forest (baseline)

- Test ROC‑AUC ≈ **0.912**  
- Test accuracy ≈ **0.86**

### Interpretation

- AUC ≈ 0.95 indicates that in about 95% of random pairs, the model ranks the strong‑growth quarter above the no‑growth quarter, so it is very good at ranking manufacturers by growth potential.  
- Feature importance for XGBoost shows `Market_Share` as the most important feature, followed by `QoQ_Growth_Rate` and `Volatility`, matching the intuition that momentum, stability and competitive position drive strong growth.  
- Validation and test performance are close, suggesting controlled overfitting.

## Visualisations

The notebook and presentation include:

- Class balance plot for the strong‑growth label.  
- Correlation heat map for the engineered features.  
- ROC curves for Random Forest and XGBoost on the test set.  
- Feature importance bar charts for both models.  
- Learning curves showing train vs validation performance for XGBoost.

## Limitations & future work

- Only internal registration‑based features are used; no macroeconomic variables, policy indicators or fuel‑type mix are included.  
- The 10% growth threshold is a reasonable business rule but could be varied and analysed more systematically.  
- Future work could extend the feature set, explore calibration and threshold optimisation, incorporate regional or fuel‑type information, and investigate deployment as an API or dashboard.

## How to run

1. Clone the repository.  
2. Download `df_VEH0160_UK.csv` from the DfT Vehicle Licensing Statistics data tables and place it in `data/`.[web:24]  
3. Open `01_SMMT_2025_EDA_and_Models.ipynb` in Jupyter or Google Colab.  
4. Run all cells in order; paths and encodings are set up for `df_VEH0160_UK.csv`.

This repository structure and methodology correspond to the MSc report *“Predicting Strong‑Growth UK Car Manufacturers Using Random Forest and XGBoost”*, particularly Chapters 3 (Methodology) and 4 (Results and Discussion).[page:28]
