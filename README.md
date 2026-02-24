# Project Overview

This project explores **Supervised Machine Learning** through:

-  **Classification** using k-Nearest Neighbors (k-NN)
-  **Regression** using Ridge Regression

Two real-world datasets are used:

1. **Cardiovascular Disease (CVD) Dataset** - Binary Classification  
2. **Prostate Cancer Dataset** - Regression Analysis  

The notebook demonstrates the complete ML workflow:

> Data exploration → Preprocessing → Model training → Evaluation → Cross-validation → Hyperparameter tuning → Interpretation

---

#  Part 1: Classification – Cardiovascular Disease Prediction

## Objective

Predict the presence of cardiovascular disease (`cardio`) using patient health indicators.

### Target Variable
- `cardio` (0 = No CVD, 1 = CVD)

### Features Used
- age
- sex
- height
- weight
- ap_hi (systolic blood pressure)
- ap_lo (diastolic blood pressure)
- smoke
- alco
- active
- cholesterol
- gluc

---

## Methodology

### 1️ Train-Test Split
- 80% training
- 20% testing
- Stratified sampling
- `random_state = 2025`

### 2️ Feature Standardization
Standardization was applied to numerical features using `StandardScaler`.

Why?

k-NN relies on **Euclidean distance**.  
Without scaling, features with larger numeric ranges (e.g., blood pressure) dominate the distance calculation.

Standardization ensures:
- Equal feature contribution
- Improved fairness
- Better model performance

---

## k-NN Model (k = 3)

### Confusion Matrix (Test Set)

|                | Predicted 0 | Predicted 1 |
|----------------|------------|------------|
| **Actual 0**   | 115 (TN)   | 25 (FP)    |
| **Actual 1**   | 36 (FN)    | 24 (TP)    |

### Metrics

- **Accuracy:** 0.695  
- **Precision:** 0.490  
- **Recall:** 0.400  

---

## Important Observation: Class Imbalance

Dataset distribution:
- ~70% No CVD
- ~30% CVD

Because of the imbalance:

- Accuracy alone is misleading.
- A naive model predicting "No CVD" always would achieve ~70% accuracy.
- The model misses **60% of actual CVD cases** (low Recall).

In medical contexts, **False Negatives are critical errors**.

---

# Leave-One-Out Cross-Validation (LOOCV)

LOOCV was used to obtain a more stable estimate of model performance.

- **Mean LOOCV Accuracy:** 0.716

Although slightly higher than the single split result, performance remains close to the majority-class baseline.

---

# Model Selection – Finding Optimal k

Using LOOCV, multiple k values were evaluated.

- **Optimal k = 9**
- **Accuracy ≈ 0.75**

### Bias-Variance Behavior

- Very small k → High variance (overfitting)
- Very large k → High bias (predicts majority class)
- Extremely large k (201–230) → Accuracy converges to ~70%

This confirms classical bias-variance tradeoff behavior in k-NN.

---

# Part 2: Regression – Prostate Cancer Dataset

## Objective

Predict continuous target variable:

- `lpsa` (log PSA level)

---

## Features

- lcavol
- lweight
- age
- lbph
- svi
- lcp
- gleason
- pgg45

Target:
- `lpsa`

---

# Ridge Regression

## Data Preparation

- Train-test split (80/20)
- `random_state = 2025`
- Feature standardization

---

## Initial Model

Ridge Regression with:

```python
Ridge(alpha=100, random_state=2025)
```

Observation:
- Testing MSE > Training MSE (expected)
- Model was slightly over-regularized

---

# Hyperparameter Tuning (K-Fold Cross-Validation)

- 5-fold cross-validation
- Alpha search range:
  ```python
  np.logspace(-2, 10, num=13)
  ```
- Log-scale visualization of MSE vs alpha

### Optimal Alpha Found:

```
alpha = 1.0
```

This is significantly smaller than 100.

---

## Final Model (alpha = 1.0)

Results:
- Lower Training MSE
- Significantly lower Testing MSE
- Better generalization

This confirms:
> The initial model (alpha=100) was over-regularized.

---

# Model Interpretation

After retraining with optimal alpha:

- Extracted intercept
- Extracted coefficients
- Reconstructed regression formula
- Ranked features by absolute coefficient magnitude
- Visualized coefficients using a horizontal bar chart

### Key Insight

- `lcavol` (log cancer volume) is the strongest predictor of PSA level.
- Other features have smaller but meaningful contributions.

---

# Technologies Used

- Python 3
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

# Project Structure

```
.
├── Supervised-Learning-Classification-Regression.ipynb
├── ex2_cardio_data.csv
├── prostate_data.csv
└── README.md
```

---

# How to Run

1. Clone repository:

```bash
git clone https://github.com/dieeju7/Supervised-Learning-Classification-Regression.git
```

2. Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Open notebook:

```bash
jupyter notebook Supervised-Learning-Classification-Regression.ipynb
```

---

# Key Learning Outcomes

- Understanding k-NN classification
- Handling class imbalance
- Importance of feature scaling
- Bias–variance tradeoff
- Leave-One-Out Cross-Validation
- Ridge regression and regularization
- Hyperparameter tuning with K-Fold CV
- Model interpretation through coefficients

---

# Conclusion

- k-NN struggled with imbalanced medical data.
- Ridge regression benefited significantly from proper hyperparameter tuning.
- Cross-validation proved essential for robust model selection.

This exercise demonstrates a complete and reproducible supervised learning pipeline from start to finish.
