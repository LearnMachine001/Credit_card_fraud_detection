# Credit_card_fraud_detection

#  Credit Card Fraud Detection

A machine learning system for detecting fraudulent credit card transactions in real time, with high precision and minimal false positives.

---

##  Overview

Credit card fraud causes billions of dollars in losses each year. This project builds a supervised machine learning pipeline that classifies transactions as **fraudulent** or **legitimate** using historical transaction data. The system is designed to handle severe class imbalance (fraud is rare) while achieving high recall to catch as many fraudulent transactions as possible.

---

##  Project Structure

```
credit-card-fraud-detection/
├── data/
│   ├── raw/                    # Original dataset(s)
│   └── processed/              # Cleaned and feature-engineered data
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # Data cleaning and feature engineering
│   └── 03_modeling.ipynb       # Model training and evaluation
├── src/
│   ├── preprocess.py           # Data preprocessing pipeline
│   ├── train.py                # Model training script
│   ├── evaluate.py             # Evaluation metrics and reporting
│   └── predict.py              # Inference / real-time prediction
├── models/
│   └── best_model.pkl          # Serialized trained model
├── requirements.txt
├── config.yaml                 # Hyperparameters and config
└── README.md
```

---

## 📊 Dataset

This project uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle (ULB Machine Learning Group).

| Property | Details |
|---|---|
| Total Transactions | 284,807 |
| Fraudulent | 492 (~0.17%) |
| Features | V1–V28 (PCA-transformed), `Time`, `Amount` |
| Target | `Class` (0 = Legit, 1 = Fraud) |

>  The dataset is **highly imbalanced**. Techniques like SMOTE, undersampling, and class weighting are applied to address this.

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Install Dependencies

```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
```

### Requirements

```
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
lightgbm
matplotlib
seaborn
joblib
jupyter
```

---

## 🚀 Usage

### 1. Preprocess the Data

```bash
python src/preprocess.py --input data/raw/creditcard.csv --output data/processed/
```

### 2. Train the Model

```bash
python src/train.py --config config.yaml
```

### 3. Evaluate

```bash
python src/evaluate.py --model models/best_model.pkl --data data/processed/test.csv
```

### 4. Predict on New Transactions

```bash
python src/predict.py --input transactions.csv --output predictions.csv
```

---

##  Models & Approach

### Class Imbalance Handling

- **SMOTE** (Synthetic Minority Oversampling Technique)
- **Class weight balancing** in model hyperparameters
- **Threshold tuning** on decision boundary

### Models Evaluated

| Model | Description |
|---|---|
| Logistic Regression | Baseline linear classifier |
| Random Forest | Ensemble tree-based model |
| XGBoost | Gradient boosting (best performer) |
| LightGBM | Fast gradient boosting |
| Isolation Forest | Anomaly detection baseline |

### Best Model: XGBoost

Selected based on **F1 score** and **ROC-AUC** on the held-out test set.

---

## 📈 Results

| Metric | Score |
|---|---|
| ROC-AUC | 0.98 |
| Precision (Fraud) | 0.91 |
| Recall (Fraud) | 0.87 |
| F1 Score (Fraud) | 0.89 |

> Evaluation focuses on fraud class metrics, as minimizing false negatives (missed fraud) is the primary objective.

### Confusion Matrix

```
                  Predicted Legit   Predicted Fraud
Actual Legit          56,854              10
Actual Fraud              12              86
```

---

##  Feature Engineering

- `Amount_log` — log-transformed transaction amount (reduces skew)
- `Hour` — extracted from `Time` feature (cyclical encoding applied)
- `Amount_zscore` — standardized amount per hour bucket
- V1–V28 are retained as-is (already PCA-transformed in the source data)

---

##  Key Design Decisions

**Why F1 / Recall over Accuracy?**
With only 0.17% fraud, a model predicting everything as "legit" achieves 99.8% accuracy — but is useless. Recall matters most here: catching fraud is the goal.

**Why threshold tuning?**
The default 0.5 decision threshold is not optimal for imbalanced classes. Threshold is tuned on the validation set to maximize F1 on the minority class.

**Why not a neural network?**
Tree-based models (XGBoost, LightGBM) outperform deep learning on this tabular dataset and are faster to train and explain.

---

##  Testing

```bash
pytest tests/
```

---

## 📝 Future Work

- [ ] Deploy as a REST API using FastAPI or Flask
- [ ] Add real-time streaming support (Kafka / Spark)
- [ ] Implement model explainability (SHAP values)
- [ ] Add drift detection for production monitoring
- [ ] Experiment with AutoML (e.g., AutoGluon, H2O)

---

##  Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -m 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

##  Acknowledgements

- [ULB Machine Learning Group](https://mlg.ulb.ac.be/) for the dataset
- [Kaggle](https://www.kaggle.com/) for hosting the dataset
- [imbalanced-learn](https://imbalanced-learn.org/) for SMOTE and resampling utilities
