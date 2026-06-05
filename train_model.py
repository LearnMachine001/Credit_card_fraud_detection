# train_model.py
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score,f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

v_feature = [f"V{i}" for i in range(1,29)]
# ================= 1. Load Dataset =================
df = pd.read_csv("C:/creditcard.csv")

# Basic features
df["Time_hour"] = df["Time"] / 3600  # convert seconds to hours
df["Amount_log"] = np.log1p(df["Amount"])  # log-transform Amount
df["Amount_hour_interaction"] = df["Amount"] * df["Time_hour"]
df["is_night"] = df["Time_hour"].apply(lambda x: 1 if x<6 or x>22 else 0)


derived_features = ["Amount", "Time_hour", "Amount_log", "Amount_hour_interaction", "is_night"]
features = v_feature + derived_features
X = df[features]
y = df["Class"]

# ================= 2. Split Dataset =================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ================= 3. Handle Imbalance =================
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)


scale_pos_weight = len(y_train[y_train == 0])/ len(y_train[y_train == 1])
print(f"Original class distribution: {np.bincount(y_train)}")
print(f"After SMOTE: {np.bincount(y_train_res)}")

# ================= 4. Train Model =================
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight = scale_pos_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train_res, y_train_res)

# ================= 5. Evaluate Model =================
y_pred_prob = model.predict_proba(X_test)[:,1]

# Tune threshold for better precision/recall trade-off
threshold = 0.15
y_pred = (y_pred_prob >= threshold).astype(int)

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_prob))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print(f"F1-Score {f1_score(y_test,y_pred)}")

# ================= 6. Save Model + Features + Threshold =================
pickle.dump(model, open("fraud_model.pkl", "wb"))
pickle.dump(features, open("all_features.pkl", "wb"))
pickle.dump(threshold, open("best_threshold.pkl", "wb"))


