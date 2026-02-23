# Global Earthquake Prediction 🌍

A machine-learning project that predicts whether a significant earthquake will trigger a **tsunami**, using seismic features extracted from global earthquake records.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [What I Learned](#what-i-learned)
   - [Data Exploration](#1-data-exploration)
   - [Data Cleaning](#2-data-cleaning)
   - [Exploratory Data Analysis (EDA)](#3-exploratory-data-analysis-eda)
   - [Baseline Model – Random Forest](#4-baseline-model--random-forest)
   - [Feature Importance](#5-feature-importance)
   - [Refined Model (Dropping Low-Value Features)](#6-refined-model-dropping-low-value-features)
   - [Hyperparameter Tuning](#7-hyperparameter-tuning)
4. [Results Summary](#results-summary)
5. [Technologies Used](#technologies-used)

---

## Project Overview

Given seismic measurements (magnitude, depth, location, etc.) the goal is to classify whether the event triggered a tsunami (`1`) or not (`0`).  
The notebook walks through the full ML pipeline: data loading → cleaning → EDA → modelling → evaluation → hyperparameter tuning.

---

## Dataset

**File:** `earthquake_data_tsunami.csv`

| Column | Description |
|--------|-------------|
| `magnitude` | Richter-scale magnitude of the earthquake |
| `cdi` | Community Internet Intensity (felt intensity reported by the public) |
| `mmi` | Modified Mercalli Intensity (instrumental measure) |
| `sig` | Significance score (composite measure of impact) |
| `nst` | Number of seismic stations that reported the event |
| `dmin` | Minimum distance (degrees) to the nearest station |
| `gap` | Largest azimuthal gap between stations (degrees) |
| `depth` | Depth of the earthquake (km) |
| `latitude` | Latitude of the epicentre |
| `longitude` | Longitude of the epicentre |
| `Year` | Year of occurrence |
| `Month` | Month of occurrence |
| `tsunami` | **Target** – 1 if a tsunami was triggered, 0 otherwise |

- **Shape:** 782 rows × 13 columns  
- **Missing values:** None (all columns 100% complete)  
- **Class balance:** 478 non-tsunami events vs 304 tsunami events (~61/39 split)

---

## What I Learned

### 1. Data Exploration

```python
df = pd.read_csv('earthquake_data_tsunami.csv')
df.shape        # (782, 13)
df.info()       # all float64 / int64, no nulls
df.describe()   # magnitude range 6.5 – 9.1, depth 0 – 700+ km
```

Key statistics from `df.describe()`:

| Stat | magnitude | sig | depth |
|------|-----------|-----|-------|
| mean | 6.94 | 870 | ~84 km |
| min  | 6.50 | 650 | ~0 km |
| max  | 9.10 | — | 700+ km |

### 2. Data Cleaning

Two filtering steps removed physically impossible coordinates:

```python
df = df[(df['latitude']  >= -90)  & (df['latitude']  <= 90)]
df = df[(df['longitude'] >= -180) & (df['longitude'] <= 180)]
```

No rows were removed here because the dataset was already clean, but the check is important for production pipelines.

### 3. Exploratory Data Analysis (EDA)

Distribution plots (`sns.histplot` with KDE) were created for four key features:

- **Magnitude** – right-skewed; most events in the 6.5–7.0 range (131 events at M 6.5, 115 at M 6.6)
- **CDI** – bimodal; many events with CDI = 0 (not felt by public)
- **MMI** – concentrated between 5 and 7
- **Sig** – roughly bell-shaped, centred around 750–900

```python
def plotting(var, num):
    plt.subplot(2, 2, num)
    sns.histplot(df[var], kde=True)

plotting('magnitude', 1)
plotting('cdi', 2)
plotting('mmi', 3)
plotting('sig', 4)
plt.tight_layout()
```

### 4. Baseline Model – Random Forest

**Features:** all 12 columns except `tsunami`  
**Split:** 80% train / 20% test (`random_state=42`)

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

**Results:**

| Metric | Value |
|--------|-------|
| Accuracy | **93.6%** |
| Precision (class 0) | 0.98 |
| Recall (class 0) | 0.91 |
| Precision (class 1) | 0.89 |
| Recall (class 1) | 0.97 |
| F1-score (macro avg) | 0.94 |

**Confusion matrix:**

```
[[83  8]
 [ 2 64]]
```

The model correctly identified 83 / 91 non-tsunami events and 64 / 66 tsunami events, with only 10 misclassifications in total.

### 5. Feature Importance

```python
importances = model.feature_importances_
feat_imp = pd.Series(importances, index=X.columns)
feat_imp.sort_values().plot(kind='barh')
plt.show()
```

A horizontal bar chart shows the relative importance of each feature.  
`Year` and `Month` were identified as low-information features worth dropping.

### 6. Refined Model (Dropping Low-Value Features)

```python
X_new = df.drop(['tsunami', 'Year', 'Month'], axis=1)
model2 = RandomForestClassifier()
model2.fit(X_train, y_train)
accuracy_score(y_test, y_pred)   # ≈ 89.8%
```

Removing `Year` and `Month` slightly reduced accuracy (~89.8% vs 93.6%) suggesting those columns provided some signal in the baseline model. This motivates further feature analysis and tuning.

### 7. Hyperparameter Tuning

**Grid Search CV** – exhaustive search over all combinations:

```python
from sklearn.model_selection import GridSearchCV

classifier = GridSearchCV(model2, {
    'n_estimators': [100, 200, 300, 500],
    'max_depth':    [None, 5, 10]
})
classifier.fit(X_train, y_train)
```

Selected results from `cv_results_`:

| max_depth | n_estimators | mean_test_score |
|-----------|-------------|-----------------|
| None | 500 | **0.8880** |
| None | 300 | 0.8848 |
| None | 200 | 0.8816 |
| 5 | 100 | 0.8752 |

**Randomized Search CV** – faster alternative sampling a subset of combinations:

```python
from sklearn.model_selection import RandomizedSearchCV

classifier2 = RandomizedSearchCV(model2, {
    'n_estimators': [100, 200, 300, 500],
    'max_depth':    [None, 5, 10]
}, n_iter=5, n_jobs=3, cv=5)
classifier2.fit(X_train, y_train)
```

Key takeaway: `max_depth=None` (fully grown trees) with more estimators (`500`) consistently gave the best cross-validated score (~88.8%).

---

## Results Summary

| Model | Features | Accuracy |
|-------|----------|----------|
| Random Forest (default) | All 12 | **93.6%** |
| Random Forest (default) | Without Year & Month | 89.8% |
| Random Forest (GridSearchCV best) | Without Year & Month | ~88.8% (CV) |

The baseline Random Forest with all features achieved the highest test accuracy of **93.6%**, demonstrating that seismic measurements alone can reliably predict tsunami occurrence.

---

## Technologies Used

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading and manipulation |
| `seaborn` / `matplotlib` | Data visualisation |
| `scikit-learn` | Model training, evaluation, and hyperparameter tuning |
| Google Colab | Interactive notebook environment |
