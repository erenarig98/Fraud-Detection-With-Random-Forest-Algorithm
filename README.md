# 💳 Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Neural%20Networks-ff6f00)
![Status](https://img.shields.io/badge/Status-Completed-success)

An end-to-end Machine Learning project designed to detect fraudulent credit card transactions in an **extremely imbalanced dataset**. This project addresses the core challenges of financial anomaly detection, focusing on minimizing missed frauds (False Negatives) while optimizing operational costs.

---

## 📌 Project Overview

Credit card fraud detection is a classic high-stakes classification problem. In real-world financial data, fraudulent transactions represent a tiny fraction of total volume (typically ~0.17%). 

* **The Accuracy Trap:** A naive model predicting "Non-Fraud" for every transaction achieves **99.8% accuracy** while completely failing its core objective.
* **Our Solution:** Prioritizing **Recall** and **AUPRC (Area Under the Precision-Recall Curve)** over standard accuracy, utilizing advanced class weighting and customized decision thresholding.

---

## 📊 Dataset & Preprocessing

* **Source:** Kaggle Credit Card Fraud Detection Dataset.
* **Features:** 
  * $V_1$ through $V_28$: Numerical input variables resulting from a **PCA transformation** (due to confidentiality).
  * `Time`: Seconds elapsed between each transaction and the first transaction in the dataset.
  * `Amount`: Transaction amount.
  * `Class`: Response variable ($1$ for fraud, $0$ otherwise).
* **Preprocessing Pipeline:**
  * Independent feature scaling using `StandardScaler` for `Amount` and `Time` to prevent data leakage and ensure stable gradient updates.
  * Stratified train-test splitting to preserve the rare fraud ratio across sets.

---

## 🛠️ Methodology & Models

### 1. Random Forest (Baseline & Production Model)
* **Configuration:** `n_estimators=100`, `max_depth=15`, `class_weight='balanced_subsample'`
* **Threshold Tuning:** Moving away from the default `0.50` threshold down to **`0.30`** to increase the model's sensitivity towards minority classes, successfully capturing more fraudulent transactions at a controlled False Positive rate.

### 2. Neural Networks / Deep Learning (Alternative Architecture)
* **Architecture:** Multi-Layer Perceptron (MLP) built with TensorFlow/Keras.
* **Loss Function:** `binary_crossentropy` optimized with custom class weight dictionaries to heavily penalize missed fraud cases during backpropagation.

---

## 📈 Evaluation Metrics

Given the extreme class imbalance, traditional metrics like Accuracy are misleading. We focus on:
* **Recall (Sensitivity):** Ensuring we catch the maximum possible number of actual frauds ($TP / (TP + FN)$).
* **AUPRC:** Highly recommended for imbalanced distributions as it heavily weights the minority class performance.
* **Confusion Matrix & Cost-Benefit Analysis:** Balancing the cost of a missed fraud (financial loss) against a false alarm (customer friction).

---

## 🚀 Quick Start & Code Snippet

Here is how custom thresholding is applied to capture fraud alerts effectively:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Predict probabilities
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# Apply custom decision threshold (e.g., 0.30 instead of 0.50)
CUSTOM_THRESHOLD = 0.30
y_pred_custom = (y_pred_proba >= CUSTOM_THRESHOLD).astype(int)
