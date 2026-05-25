# 🏠 House Price Prediction — Machine Learning Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow?logo=googlecolab)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Problem Statement

Housing prices are influenced by a complex mix of geographic, demographic, and structural factors. The goal of this project is to **build and evaluate machine learning regression models** that can accurately predict median house values based on features such as location, income levels, housing age, and proximity to the ocean.

This end-to-end pipeline covers data preprocessing, feature selection, model training, and performance evaluation — helping stakeholders estimate property values with quantifiable confidence.

---

## 📂 Dataset Details

| Property         | Details                                         |
|------------------|-------------------------------------------------|
| **Name**         | California Housing Dataset                      |
| **Source**       | `housing.csv` (from `House Price Dataset.zip`)  |
| **Total Records**| ~20,640 rows                                    |
| **Total Features**| 9 (8 input + 1 target)                         |
| **Target Variable** | `median_house_value` (continuous, USD)       |
| **Task Type**    | Supervised Regression                           |

### Feature Description

| Feature                | Type        | Description                                      |
|------------------------|-------------|--------------------------------------------------|
| `longitude`            | Numerical   | Geographic longitude of the block                |
| `latitude`             | Numerical   | Geographic latitude of the block                 |
| `housing_median_age`   | Numerical   | Median age of houses in the block                |
| `total_rooms`          | Numerical   | Total number of rooms in the block               |
| `total_bedrooms`       | Numerical   | Total number of bedrooms (has missing values)    |
| `population`           | Numerical   | Block population                                 |
| `households`           | Numerical   | Number of households in the block                |
| `median_income`        | Numerical   | Median household income (in tens of thousands $) |
| `ocean_proximity`      | Categorical | Distance category to the ocean                   |
| `median_house_value`   | Numerical   | **Target** — Median house value in USD           |

---

## ⚙️ Approach

The pipeline is structured into 6 sequential stages:

### 1. 🧹 Data Preprocessing
- Extracted dataset from zip archive
- Performed EDA: shape, dtypes, `.describe()`, and null count analysis
- Imputed missing numerical values using **mean strategy** (`SimpleImputer`)
- Applied **One-Hot Encoding** on `ocean_proximity` (categorical)
- Scaled numerical features using **StandardScaler**
- Built a unified `ColumnTransformer` + `Pipeline` for clean, reproducible transforms
- Plotted a **correlation heatmap** for numerical features

### 2. 🎯 Feature Selection
Two methods were applied and combined:

- **SelectKBest** (using `f_regression`): Selected top 10 statistically significant features
- **Random Forest Feature Importance**: Ranked all features by predictive contribution; threshold set at half the mean importance
- **Multicollinearity filter**: Dropped features with correlation > 0.9 (e.g., `latitude` dropped due to high correlation with `longitude`)

**Final 4 Selected Features:**
```
median_income | ocean_proximity_INLAND | longitude | housing_median_age
```

> `median_income` was the dominant predictor with **48.3% importance** from the Random Forest selector.

### 3. ✂️ Train / Test Split

| Split    | Samples | Features |
|----------|---------|----------|
| Training | 16,512  | 4        |
| Testing  | 4,128   | 4        |

- Split ratio: **80% train / 20% test**
- `random_state=42` for reproducibility

### 4. 🤖 Model Training

Three regression models were trained on the processed training data:

| Model                    | Configuration                        |
|--------------------------|--------------------------------------|
| Linear Regression        | Default sklearn implementation       |
| Decision Tree Regressor  | `random_state=42`                    |
| Random Forest Regressor  | `n_estimators=100`, `random_state=42`|

### 5. 📊 Evaluation

Models were evaluated on the test set using three regression metrics:

- **MAE** — Mean Absolute Error (average prediction error in USD)
- **RMSE** — Root Mean Squared Error (penalizes large errors more)
- **R² Score** — Proportion of variance explained by the model

### 6. 💾 Output
- Summary comparison table printed for all models
- Best model identified by lowest RMSE
- Best model serialized to `best_model.pkl` using `joblib`

---

## 📈 Results

### Model Performance Summary

| Model                     | MAE ($)       | RMSE ($)      | R² Score |
|---------------------------|---------------|---------------|----------|
| Linear Regression         | 53,529.41     | 74,131.80     | 0.5806   |
| Decision Tree Regressor   | 54,487.23     | 83,408.52     | 0.4691   |
| **Random Forest Regressor** | **42,737.27** | **63,904.49** | **0.6884** |

### 🏆 Best Model: Random Forest Regressor

The **Random Forest Regressor** outperformed all other models across every metric:

- Lowest MAE: **$42,737** average prediction error
- Lowest RMSE: **$63,904** (saved as `best_model.pkl`)
- Highest R²: **0.688** — explains ~69% of the variance in house prices

> The Decision Tree overfit on training data and generalized poorly (R² = 0.47). Linear Regression provided a reasonable baseline but was limited by its inability to capture non-linear relationships. The ensemble approach of Random Forest yielded the most reliable predictions.


---

## 🚀 How to Run

1. **Open in Google Colab** — Upload `House_Price_Dataset_.ipynb`
2. **Upload the dataset** — Upload `House Price Dataset.zip` to `/content/`
3. **Set paths** at the top of the notebook:
   ```python
   zip_file_path = '/content/House Price Dataset.zip'
   extract_dir = '/content/House_Price_Dataset_extracted'
   csv_file_path = '/content/House_Price_Dataset_extracted/housing.csv'
   ```
4. **Run All Cells** — `Runtime → Run All`
5. The best model will be saved as `best_model.pkl` in `/content/`

---

## 📦 Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
```

Install via:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

---

## 👤 Author

> Built as part of a Machine Learning pipeline project covering data preprocessing, feature engineering, model training, and evaluation.
