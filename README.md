# SMMT 2025 UK Car Sales – Data Science Project

This repository contains an end-to-end data science project analysing UK new car registrations for 2025 using SMMT data. The work covers data ingestion, cleaning, exploratory data analysis (EDA), feature engineering, and a supervised learning model (Random Forest) to classify manufacturers into high vs low sales performers.

## Project Overview

The core dataset is the SMMT "Cars_12_2025.xlsx" file (sheet `CARS_Month_End`), which reports registrations of new cars in the UK by marque with both December and year-to-date (YTD) figures for 2024 and 2025.

Key objectives:
- Clean and restructure the original Excel sheet into a modelling-ready DataFrame.
- Explore market structure, fuel types, manufacturers, and top models.
- Engineer business-relevant features.
- Build and tune a Random Forest classifier to predict high-sales manufacturers.

## Data and Preprocessing

Steps performed:
- Unzip `Full-year-car-registrations-2025.zip` and load `Cars_12_2025.xlsx` from the `CARS_Month_End` sheet.
- Rename opaque columns to meaningful names: **marque**, `dec_2025`, `dec_share_2025`, `dec_2024`, `dec_share_2024`, `dec_change_pct`, `ytd_2025`, `ytd_share_2025`, `ytd_2024`, `ytd_share_2024`, `ytd_change_pct`.
- Cast numeric columns, drop header/summary rows, and remove the "Total Market" aggregate row.
- Create a binary target:
  - `High_Sales` = 1 if `ytd_2025` ≥ median YTD 2025 units, else 0.
  - `Purchase` used as the final target, mirroring `High_Sales` (balanced 29 / 29).

Result: a clean dataset of 58 manufacturers (rows) and 13+ features.

## Exploratory Data Analysis

The notebook includes multiple EDA sections:

- **Top 15 manufacturers** by YTD 2025 sales, visualised with bar charts and labelled values to highlight the largest marques.
- **Correlation analysis**:
  - Full numeric correlation heatmap.
  - Feature-target correlation plot showing `ytd_2025`, `ytd_share_2025`, `dec_2024`, `dec_share_2024`, etc., as the strongest predictors of `Purchase`.
- **Comprehensive EDA summary** reporting dataset shape, columns, and structure.

### Fuel type analysis

A supplementary table summarises fuel type performance in 2025:

| Fuel type              | Units   | Market % | Growth % |
|------------------------|---------|----------|----------|
| Petrol                 | 937,938 | 46.4     | -8.0     |
| Diesel                 | 103,906 | 5.1      | -15.6    |
| Hybrid (HEV)           | 280,185 | 13.9     | 7.2      |
| Plug-in Hybrid (PHEV)  | 225,143 | 11.1     | 34.7     |
| Battery Electric (BEV) | 473,348 | 23.4     | 23.9     |

These are visualised via:
- Pie chart for market share.
- Bar chart for total units.
- Bar chart for YoY growth.
- Pie chart comparing traditional (petrol/diesel) vs electrified powertrains.

### Additional insights

- **Manufacturers by first letter**: aggregate sales and counts by initial character of `marque`, plus the top seller for each letter (e.g. Volkswagen for V, BMW for B).
- **Top 10 models**: a synthetic table of best-selling models (e.g. Ford Puma, Kia Sportage, Nissan Qashqai) with units and body type (SUV vs hatchback), visualised with bar and pie charts.
- **Number plate patterns**: descriptive statistics and a pie chart of UK "fancy plate" patterns and approximate market shares.

## Feature Engineering

Guided by the correlation analysis and supervisor feedback, several engineered features are created:

- `ytd_growth_rate`: year-over-year growth in units ≈ (ytd_2025 - ytd_2024) / ytd_2024 × 100.
- `share_change`: change in market share between 2024 and 2025 (`ytd_share_2025 - ytd_share_2024`).
- `sales_momentum`: December 2025 sales vs average monthly 2025 sales (`dec_2025 / (ytd_2025 / 12)`).
- `top_quartile`: flag for manufacturers in the top 25% of `ytd_2025` (1 for top quartile, else 0).

These are appended to the dataset, increasing the feature space to 17 columns.

## Modelling: Random Forest Classifier

The main supervised model is a Random Forest classifier predicting `Purchase` (high vs low sales manufacturer).

Pipeline highlights:
- Select the most informative numeric features (including engineered ones) and drop identifiers.
- Train-test split:
  - 46 training samples, 12 test samples.
  - Balanced classes in the train set (23/23).

### Baseline Random Forest

A baseline Random Forest yields:
- Accuracy, precision, recall, F1, ROC-AUC all equal to 1.0000 on the test set (note: small sample, risk of overfitting).

### Hyperparameter tuning

GridSearchCV is run with ROC-AUC as the scoring metric:

Best configuration:
- `n_estimators`: 100
- `max_depth`: 10
- `max_features`: `sqrt`
- `min_samples_split`: 2
- `min_samples_leaf`: 1

Cross-validated mean ROC-AUC: 0.99, with test performance remaining at 1.0 for all key metrics and a perfect confusion matrix (no false positives or negatives).

## Repository Structure

```text
.
├── data/
│   ├── raw/
│   │   └── Full-year-car-registrations-2025.zip
│   └── processed/
│       └── smmt_clean.csv
├── notebooks/
│   └── Anandasai_Sunke_DS_PROJECT.ipynb
├── src/
│   ├── data_preparation.py
│   ├── eda.py
│   ├── features.py
│   └── model_random_forest.py
├── README.md
└── requirements.txt
```

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/SunkeAnandasai3784/SMMT-2025-UK-Car-Sales-Analysis.git
   cd SMMT-2025-UK-Car-Sales-Analysis
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place `Full-year-car-registrations-2025.zip` in `data/raw/` (or update paths in the notebook).

4. Open the notebook:
   - In Jupyter or VS Code, or
   - Upload to Google Colab and run all cells in order.

## Technologies Used

- **Python** (pandas, numpy)
- **Visualisation**: matplotlib, seaborn
- **Machine Learning**: scikit-learn (RandomForestClassifier, GridSearchCV, metrics)
- **Environment**: Google Colab with Google Drive integration

## Author

Anandasai Sunke

## Interpreting the Random Forest performance

The cleaned dataset contains 58 manufacturers (51 rows after removing inf/NaN rows), with very strong correlations between the YTD volume/share features and the `Purchase` label. This means that, for this dataset, high vs low sales status is almost fully determined by a small set of highly informative numeric variables. The learning curves show that the tuned Random Forest reaches training ROC-AUC approximately 1.0 for all training sizes, while the validation ROC-AUC remains very high (approximately 0.97-0.99) with a wide variance band due to the small sample. Combined with a perfect confusion matrix on just 12 test samples, this indicates that the model can almost perfectly separate the classes on both training and validation folds, and that the task is nearly deterministic on the available data rather than evidence of strong generalisation to a much larger, more diverse population.

## Feature importances and leakage check

The Random Forest feature importances confirm that the strongest predictors are exactly the variables highlighted in the correlation analysis: `ytd_2025`, `ytd_share_2025`, `ytd_2024`, `ytd_share_2024`, and the engineered growth / momentum features derived from them. I have explicitly checked that none of the engineered features directly encode the `Purchase` or `High_Sales` labels, and that all features used in modelling are computed purely from sales and market-share information available before the target is defined. This alignment between the correlation heatmap and the importance ranking, together with the absence of any target-derived fields in `feature_cols`, provides reassurance that there is no data leakage from the label into the predictors.

## Future work: complementary dataset and deep learning extension

A key limitation of the current work is the very small sample size (58 manufacturers) and the almost deterministic relationship between YTD / share features and the `Purchase` label. To address this, I plan to extend the analysis with a complementary dataset rather than starting a new project. In particular, I will:

- Obtain additional manufacturer-level registration tables from SMMT and/or ONS (e.g. monthly or multi-year time-series of registrations by marque) so that I can build a panel dataset with many more observations per manufacturer.
- Reuse the same pipeline defined in the `src/` modules (data preparation, feature engineering, RandomForestSalesClassifier) on this larger dataset, keeping the same business question of predicting high vs low performers.
- If the time dimension is rich enough, experiment with a simple deep learning model (for example, an MLP on lagged features or a small LSTM over monthly registrations) to showcase more advanced methods on the extended dataset, while still staying within the original project scope.

This complementary dataset plan directly follows my supervisor's recommendation to work with larger, time-series style data to better test model generalisation and to justify the use of more complex models.

MSc Data Science, University of Hertfordshire
