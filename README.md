# Order Delay Analysis

> A production‑oriented pipeline to predict order delay status (`hyper_ack`, binary) from operational and geospatial signals. The project includes EDA, feature engineering (including K‑Means geoclusters), model selection (Logistic Regression & XGBoost with cross‑validation), and an inference workflow with persisted artifacts.

---

## 1) Project Overview

This repository builds a supervised classifier that predicts the target label `hyper_ack` (0/1) for each order. The solution covers:

* **EDA** on demand patterns by hour and day, order volumes, and distances.
* **Feature engineering**, including:

  * extracting `first_created_at_hour` from timestamps,
  * quantile binning for `sum_product`,
  * geospatial **K‑Means clustering** of `source_*` and `destination_*` coordinates,
  * preparation of numeric / categorical / ordinal feature blocks.
* **Modeling** with reproducible pipelines:

  * Logistic Regression (with `SelectKBest(chi2)` feature selection and grid search),
  * XGBoost (with grid search over depth, learning rate, estimators).
* **Model persistence** via `joblib` for both the classifier and the two K‑Means clusterers.
* **Inference** notebook section that reloads the saved artifacts and scores new data.

---

## 2) Data

* **Input file**: `data/part1-dataset.csv` (referenced in the notebook as `../data/part1-dataset.csv`).
* **Key raw columns used** (exact spellings from the notebook):

  * Timestamps: `created_date`, `first_created_at`
  * Geo: `source_latitude`, `source_longitude`, `destination_latitude`, `destination_longitude`
  * Demand / pricing: `total_distance`, `final_customer_fare`, `final_biker_fare`, `first_customer_fare`, `sum_product`
  * Categorical: `deliverey_category_id`, `weekday`, `time_bucket`
  * **Target**: `hyper_ack` (binary)

> ⚠️ Note: `deliverey_category_id` is spelled this way in the notebook and is kept as‑is in code.

---

## 3) Feature Engineering

The notebook constructs the following features:

* **Time-based**

  * `first_created_at_hour` extracted from `first_created_at` (0–23)
  * `weekday` (as provided), `time_bucket` (as provided)
* **Geo clusters** (each with `n_clusters = 3`, chosen via elbow / inertia curve):

  * `source_cluster` from `source_latitude`, `source_longitude`
  * `destination_cluster` from `destination_latitude`, `destination_longitude`
  * Pipeline: `StandardScaler → KMeans`
  * Artifacts persisted to `model/` as `clustering_source_location.joblib` and `clustering_destination_location.joblib`
* **Price / demand**

  * `sum_product` discretized into 4 quantile bins (`KBinsDiscretizer(..., n_bins=4, encode="ordinal", strategy="quantile")`)

The final **feature blocks** are:

* **Numerical**: `time_bucket`, `total_distance`, `first_created_at_hour`, `final_customer_fare`
* **Categorical (OHE)**: `deliverey_category_id`, `weekday`, `destination_cluster`, `source_cluster`
* **Ordinal**: binned `sum_product`

---

## 4) Modeling Pipeline

The project uses **scikit‑learn Pipelines** and a **ColumnTransformer** to keep preprocessing tied to the model:

```text
num_pipeline = SimpleImputer(strategy="mean") → MinMaxScaler
cat_pipeline (OHE) = SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown="ignore")
ord_pipeline = SimpleImputer(most_frequent) → KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")
ColumnTransformer = [num_pipeline, OHE cat pipeline, ordinal pipeline]
SelectKBest(chi2) for feature selection (used in the logistic pipeline)
```

### 4.1 Logistic Regression (baseline)

* Estimator: `LogisticRegression(random_state=0)`
* Pipeline: `ColumnTransformer → SelectKBest(chi2) → LogisticRegression`
* Grid search (`cv=5`, scoring=`accuracy`):

  * `penalty ∈ {l1, l2}`
  * `C ∈ logspace(-4, 4, 20)`
  * `solver = liblinear`

### 4.2 XGBoost (final model)

* Estimator: `xgboost.XGBClassifier(random_state=0)`
* Pipeline: `ColumnTransformer → XGBClassifier`
* Grid search (`cv=5`, scoring=`roc_auc`):

  * `learning_rate ∈ {0.05, 0.10, …, 0.95}`
  * `max_depth ∈ {3,4,5,6}`
  * `n_estimators ∈ {50,100,150,200}`

### 4.3 Train / Validation split

* `train_test_split(test_size=0.2, stratify=y, random_state=42)`

---

## 5) Results (validation set)

> The following come from the notebook’s printed outputs.

**Logistic Regression (after CV)**

* Accuracy: **0.856**
* Precision: **0.921**
* Recall: **0.790**
* F1‑score: **0.851**

**XGBoost (after CV)**

* Accuracy (score on validation): **0.950**
* Precision: **0.904**
* Recall: **0.866**
* F1‑score: **0.884**
* Best params (example from CV): `learning_rate=0.05`, `max_depth=3`, `n_estimators=50`

> Confusion matrices for both models are plotted in the notebook (seaborn heatmaps).

---

## 6) Saved Artifacts

Artifacts are persisted with `joblib`:

```
model/
├── classification_pipeline.joblib               # best XGBoost pipeline
├── clustering_source_location.joblib            # KMeans on source lat/lon
└── clustering_destination_location.joblib       # KMeans on destination lat/lon
```

---

## 7) How to Reproduce

### 7.1 Environment

* **Python**: 3.12+
* **Dependencies (core)**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `statsmodels`, `joblib`

Example quick setup:

```bash
# (recommended) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -U pip
pip install pandas numpy matplotlib seaborn scikit-learn xgboost statsmodels joblib
```

### 7.2 Data placement

Place your CSV as:

```
./data/part1-dataset.csv
```

(If you follow the notebook exactly, it reads from `../data/part1-dataset.csv` relative to the notebook location.)

### 7.3 Run the notebook

Open the notebook and run all cells in order. It will:

1. perform EDA,
2. build geo clusters & feature blocks,
3. run CV for Logistic & XGBoost,
4. save the best estimators under `model/`.

---

## 8) Inference (batch scoring) — example

Below is a minimal Python snippet that mirrors the notebook’s inference section. It reloads the trained pipelines and scores new data.

```python
import pandas as pd
import joblib

# 1) Load saved artifacts
clf = joblib.load("model/classification_pipeline.joblib")
kmeans_src = joblib.load("model/clustering_source_location.joblib")
kmeans_dst = joblib.load("model/clustering_destination_location.joblib")

# 2) Load fresh data
orders = pd.read_csv("data/part1-dataset.csv", parse_dates=["created_date", "first_created_at"])  # adapt path

# 3) Feature engineering consistent with training
orders = orders.assign(
    first_created_at_hour = orders["first_created_at"].dt.hour,
    source_cluster = kmeans_src.predict(orders[["source_latitude", "source_longitude"]]),
    destination_cluster = kmeans_dst.predict(orders[["destination_latitude", "destination_longitude"]]),
)

numerical_cols = ["time_bucket", "total_distance", "first_created_at_hour", "final_customer_fare"]
categorical_cols = ["deliverey_category_id", "weekday", "destination_cluster", "source_cluster"]
ordinal_catergory_cols = ["sum_product"]  # expected as in training (binned by the pipeline)

X = orders[numerical_cols + categorical_cols + ordinal_catergory_cols]

# 4) Predict
pred = clf.predict(X)
proba = getattr(clf, "predict_proba", lambda X: None)(X)
print(pred[:10])
```

> Ensure column names and dtypes match training (especially `deliverey_category_id` spelling).

---

## 9) EDA Highlights

* Hourly order distribution and **share of `hyper_ack`** by `first_created_at_hour`.
* Daily orders trend with a reference line for mean volume.
* Correlation heatmap across core engineered features.

Plots are created with `matplotlib`/`seaborn` and can be reproduced by running the notebook.

---

## 10) Next Steps (Ideas)

* Add calibration (`CalibratedClassifierCV`) to improve probability quality.
* Try gradient boosting variants (LightGBM, CatBoost) and class‑imbalance strategies.
* Use haversine distance and/or map‑matched features for better geo signals.
* Move training/inference to standalone scripts or a CLI for easier automation.

---

## 11) Author

**Ali Naddafi** — end‑to‑end data preparation, modeling, and documentation.

> For questions, please open an issue or reach out via GitHub profile.
