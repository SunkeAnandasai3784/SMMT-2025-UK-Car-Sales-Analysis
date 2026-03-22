# Predicting Strong Growth in UK Car Registrations Using DfT VEH0160

This project analyses new car registrations in the UK using the **DfT df_VEH0160_UK** dataset, which reports vehicles registered for the first time by body type, make, generic model and model.  
I focus on **Cars** and build a **Random Forest classifier** that predicts whether a manufacturer experiences **strong year‑on‑year growth** in registrations in a given quarter.

## Project structure

- `01_SMMT_2025_EDA_and_Models.ipynb` – main notebook with data loading, EDA, feature engineering and modelling for df_VEH0160_UK.  
- `data/df_VEH0160_UK.csv` – DfT Vehicles registered for the first time dataset (not committed if too large).  
- `presentation/Anandasai-Sunke-24057259-Presentation.pptx` – final project presentation aligned with this notebook and README.  

## Data source

The project uses one official UK dataset:

- **DfT df_VEH0160_UK – Vehicles registered for the first time**  
  - Source: UK Department for Transport, Vehicle licensing statistics data files.  
  - Scope used here: BodyType = **Cars** only.  
  - Level: Make–quarter (aggregated from make, generic model and model).  
  - Time span: roughly 2014 Q3 to 2025 Q3.  

No personal or individual‑level data is used; all records are aggregated counts by manufacturer and time period.

## Aim

The aim of the project is to:

- Detect **strong‑growth quarters** for each car manufacturer in the UK using only historical registration information.  
- Build an **interpretable yet realistic** model (Random Forest) that avoids feature leakage and reports credible metrics (ROC‑AUC around 0.78 instead of unrealistic 1.0).  

## Methodology

### 1. Pre‑processing & panel creation

1. Read `df_VEH0160_UK.csv` with the correct encoding.  
2. Filter to `BodyType == "Cars"`.  
3. Melt the wide quarterly columns into long format (Make–Quarter–Registered).  
4. Extract `Year` and `QuarterNo` and create a `Date` at the end of each quarter.  
5. Aggregate to a Make–Date panel: one row per manufacturer and quarter with total new registrations.  

### 2. Feature engineering

On the Make–Date panel, I create:

- `Registered_lag1`, `Registered_lag2`, `Registered_lag4` – registrations 1, 2 and 4 quarters ago for each Make.  
- `YoY_Growth` – difference between current registrations and those 4 quarters ago.  
- `MA3` – 3‑quarter moving average of registrations (short‑term trend).  

### 3. Binary target (strong growth)

To avoid trivial predictions, I define a **harder** target:

- Compute `growth_rate = YoY_Growth / |Registered_lag4|`.  
- Set `Target = 1` if `growth_rate > 0.10` (more than 10% year‑on‑year growth), otherwise `Target = 0`.  

This means the model predicts **strong growth**, not just any positive change.

### 4. Feature matrix

The final feature matrix **does not include** `YoY_Growth`, to avoid label leakage:

- `X = [Registered_lag1, Registered_lag2, Registered_lag4, MA3]`  
- `y = Target`  

All infinities and missing values are cleaned and replaced before modelling.

### 5. Train–validation–test split

- Stratified split on `Target` into:
  - 56.25% **train**
  - 18.75% **validation**
  - 25% **test**  

The validation set is used for baseline vs tuned model comparison; the test set is only used once at the end.

## Model

I use **one main model family** for this dataset: `RandomForestClassifier` from scikit‑learn.

### Baseline Random Forest

- `n_estimators = 50`  
- `max_depth = 4`  
- `min_samples_leaf = 5`  
- `class_weight = "balanced"`  
- `random_state = 42`  

### Hyper‑parameter tuning

I tune a **smaller** Random Forest using `GridSearchCV` with 5‑fold cross‑validation, optimising ROC‑AUC.

Grid:

- `n_estimators: [30, 60]`  
- `max_depth: [3, 4, 5]`  
- `min_samples_split: [4, 6]`  
- `min_samples_leaf: [3, 5]`  
- `max_features: ["sqrt"]`  

**Best parameters**

- `max_depth = 5`  
- `max_features = "sqrt"`  
- `min_samples_leaf = 3`  
- `min_samples_split = 4`  
- `n_estimators = 60`  
- Best cross‑validated ROC‑AUC ≈ **0.807**  

## Results

### Baseline RF

- Train ROC‑AUC ≈ 0.815  
- Validation ROC‑AUC ≈ 0.764  
- Validation accuracy ≈ 0.578  
- Very high recall for the strong‑growth class (≈0.918) but lower precision (≈0.431).  

### Tuned RF (final model)

- Validation ROC‑AUC ≈ **0.795**  
- Test ROC‑AUC ≈ **0.780**  
- Validation & test accuracy ≈ **0.61**  
- Best CV ROC‑AUC ≈ **0.807**  

**Interpretation**

- ROC‑AUC around 0.78 shows the model ranks strong‑growth quarters significantly better than random (0.5) but is far from perfect, which is realistic for noisy demand data.  
- Feature importance highlights `MA3` and `Registered_lag4` as dominant predictors, consistent with business intuition that both recent trend and last year’s level matter.  
- Learning curves show a small gap between train and validation performance, indicating moderate but controlled overfitting.

## Visualisations

The notebook and presentation include:

- Class balance chart for the strong‑growth target.  
- Correlation heat maps (inputs only, and inputs + Target).  
- Baseline confusion matrix and feature importance plot.  
- Learning curve (train vs validation ROC‑AUC).  
- ROC curves for tuned RF on train+validation and test sets.  

## Limitations & future work

- Only internal lag features are used; no external macroeconomic or policy variables are included.  
- The 10% growth threshold for defining “strong growth” is somewhat arbitrary and could be tuned for different business cases.  
- Future work could explore gradient boosting models, probabilistic calibration, threshold optimisation, and the inclusion of additional explanatory variables (e.g., fuel type mix, economic indicators).  

## How to run

1. Clone the repository.  
2. Download `df_VEH0160_UK.csv` from the DfT Vehicle licensing statistics data files page and place it in `data/`.  
3. Open `01_SMMT_2025_EDA_and_Models.ipynb` in Jupyter/Colab.  
4. Run all cells in order; paths and `encoding="latin1"` are already set up for df_VEH0160_UK.
