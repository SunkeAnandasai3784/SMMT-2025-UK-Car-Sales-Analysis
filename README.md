# SMMT 2025 – UK Car Sales Analysis

This repository contains an end‑to‑end analysis of 2025 UK car sales by
manufacturer using two official data sources:

- **SMMT** (Society of Motor Manufacturers and Traders): new car registrations
  by marque for December and year‑to‑date 2024–2025.
- **DfT VEH0160** (UK Department for Transport): quarterly licensed vehicles
  by make, 2014–2025.

The goal is to understand which manufacturers are performing strongly in
2025 and how that relates to longer‑run trends in the licensed vehicle
stock.

---

## 1. Project overview

The analysis is organised in two parts inside a single Jupyter notebook:

1. **SMMT 2025 snapshot – classification**
   - Clean the SMMT Excel into a manufacturer‑level table.
   - Explore top marques, growth rates and market shares.
   - Build a **binary classifier** to label each marque as
     “high‑volume” vs “low‑volume” in 2025.

2. **DfT VEH0160 panel – forecasting**
   - Reshape the wide VEH0160 CSV into a long **make–quarter** panel.
   - Create lags and growth rates in licensed vehicles.
   - Fit a **Random Forest regression** to model short‑term changes in
     the licensed stock.

The conclusion compares what we learn from a **single‑year snapshot**
(SMMT) with the **long‑run panel** (DfT).

---

## 2. Repository structure

```text
SMMT-2025-UK-Car-Sales-Analysis/
├─ 01_SMMT_2025_EDA_and_Models.ipynb   # main analysis notebook
├─ data/
│  ├─ Cars12_2025.xlsx                 # SMMT registrations by marque
│  └─ VEH0160UK.csv                    # DfT vehicle licensing statistics
├─ figures/                            # optional: exported charts
├─ README.md
└─ requirements.txt (or environment.yml)

3. Methods
3.1 SMMT classification (high vs low marques)
Target definition

Use 2025 year‑to‑date registrations (ytd2025) by marque.

Compute the 40th and 60th percentiles of ytd2025.

Label:

1 = marques in the top 40% by volume (high).

0 = marques in the bottom 40% (low).

Middle 20% are dropped to keep a clean decision boundary.

Features

2025 and 2024 year‑to‑date volumes and market shares.

December 2025 and 2024 volumes and shares.

Year‑on‑year percentage changes.

Features with the strongest correlation to the target are kept
(e.g. 2025 YTD volume, share and December volume).

Models

Logistic Regression.

Random Forest Classifier.

Gradient Boosting Classifier.

Evaluation

Stratified train/test split (balanced 23 vs 23 labels).

Accuracy, Precision, Recall, F1‑score.

ROC‑AUC on the test set.

5‑fold cross‑validated ROC‑AUC.

Validation log‑loss and ROC curves.

Interpretation note:
The labelled snapshot contains only 46 manufacturers and the chosen
features separate the classes very clearly. As a result, all three
models achieve ROC‑AUC close to 1.0 on the test split and in
cross‑validation. These scores should be viewed as exploratory and
not as production‑ready forecasting performance.

3.2 DfT VEH0160 panel regression
Filter the VEH0160 CSV to cars only.

Reshape from wide (columns for each quarter) to long
(Make, date, Licences).

Aggregate to Make–date pairs and build:

Lagged licences (1, 2 and 4 quarters).

Year‑on‑year growth rates.

Train a Random Forest Regressor on these features.

Report 
R
2
R 
2
 , MAE and RMSE on a held‑out test set.

This shows how past licensed stock and recent growth help explain the
next quarter’s licences for each make.

4. How to run
Clone the repository:

bash
git clone https://github.com/SunkeAnandasai3784/SMMT-2025-UK-Car-Sales-Analysis.git
cd SMMT-2025-UK-Car-Sales-Analysis
Create a Python environment and install dependencies:

bash
pip install -r requirements.txt
(or use environment.yml with conda env create -f environment.yml.)

Start Jupyter:

bash
jupyter notebook
Open 01_SMMT_2025_EDA_and_Models.ipynb and run all cells.

The notebook is designed to run end‑to‑end as long as the data files are
present under data/.

5. Data sources
SMMT car registrations:
“REGISTRATIONS OF NEW CARS IN THE UNITED KINGDOM – December and
year‑to‑date 2024–2025” (Cars12_2025.xlsx).

DfT VEH0160:
UK Department for Transport “Vehicle licensing statistics –
VEH0160: Licensed cars by make and quarter, Great Britain”.

Both are official sources; this repository uses them purely for
non‑commercial, educational analysis.

6. Contact
Created by Anandasai Sunke as part of a data science project.
If you have questions or suggestions, feel free to open an issue or
reach out via GitHub.
