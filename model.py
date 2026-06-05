import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np

# ================= LOAD DATA =================
df = pd.read_csv('C:/creditcard.csv')

v_feature = [f"V{i}" for i in range(1,29)]

# ================= FEATURE ENGINEERING =================

df["Time_hour"] = df["Time"] / 3600
df["Amount_log"] = np.log1p(df["Amount"])
df["Amount_hour_interaction"] = df["Amount"] * df["Time_hour"]
df["is_night"] = df["Time_hour"].apply(lambda x: 1 if x < 6 or x > 22 else 0)

df["V_mean"] = df[v_feature].mean(axis=1)
df["V_std"] = df[v_feature].std(axis=1)
df["V_max"] = df[v_feature].max(axis=1)
df["V_min"] = df[v_feature].min(axis=1)
df["V_range"] = df["V_max"] - df["V_min"]
df["V_sum"] = df[v_feature].sum(axis=1)
df["V_abs_sum"] = df[v_feature].abs().sum(axis=1)

df["V_outlier_count"] = (df[v_feature].abs() > 2).sum(axis=1)
df["V_positive_count"] = (df[v_feature] > 0).sum(axis=1)
df["V_negative_count"] = (df[v_feature] < 0).sum(axis=1)
df["V_pos_neg_ratio"] = df["V_positive_count"] / (df["V_negative_count"] + 1)

# ================= FEATURES =================
features = v_feature + ["Time", "Amount", "Time_hour", "Amount_log",
                        "Amount_hour_interaction", "is_night",
                        "V_mean", "V_std", "V_range",
                        "V_sum", "V_abs_sum",
                        "V_outlier_count", "V_pos_neg_ratio"]

X = df[features]
y = df["Class"]

# ================= SPLIT =================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ================= FIX DATA LEAKAGE =================
mean = X_train["Amount"].mean()
std = X_train["Amount"].std()

X_train["Amount_zscore"] = (X_train["Amount"] - mean) / std
X_test["Amount_zscore"] = (X_test["Amount"] - mean) / std

# ================= AUTOENCODER =================
ae_features = v_feature + ["Amount", "Time"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[ae_features])
X_test_scaled = scaler.transform(X_test[ae_features])

input_dim = X_train_scaled.shape[1]

autoencoder = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(input_dim)
])

autoencoder.compile(optimizer='adam', loss='mse')

X_normal = X_train_scaled[y_train == 0]

autoencoder.fit(X_normal, X_normal,
                epochs=20,
                batch_size=256,
                validation_split=0.1)

# ================= RECON ERROR =================
train_pred = autoencoder.predict(X_train_scaled)
test_pred = autoencoder.predict(X_test_scaled)

X_train["recon_error"] = np.mean((X_train_scaled - train_pred)**2, axis=1)
X_test["recon_error"] = np.mean((X_test_scaled - test_pred)**2, axis=1)

# ================= MODEL =================
from xgboost import XGBClassifier

scale_pos_weight = (len(y_train[y_train==0]) / len(y_train[y_train==1]))

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss',
    random_state=42
)

model.fit(X_train, y_train)

# ================= PREDICTION =================
y_prob = model.predict_proba(X_test)[:,1]

# ================= THRESHOLD =================
from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
best_threshold = 0.25

y_pred = (y_prob > best_threshold).astype(int)

# ================= EVALUATION =================
from sklearn.metrics import classification_report, confusion_matrix,precision_score,recall_score,accuracy_score

print(f"F1-score{f1_scores}")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(f"Precision score {precision_score(y_test,y_pred)}")
print(f"Recall score {recall_score(y_test,y_pred)}")
print(f"Accuracy score {accuracy_score(y_test,y_pred)}")
print(best_threshold)

import pickle

v_feature = [f"V{i}" for i in range(1,29)]

v_means = df[v_feature].mean().to_dict()

pickle.dump(v_means, open("v_means.pkl", "wb"))
pickle.dump(model, open("model.pkl","wb"))
pickle.dump(X_train.columns.tolist(), open("all_features.pkl","wb"))
pickle.dump(best_threshold, open("best_threshold.pkl","wb"))
pickle.dump(scaler, open("scaler.pkl","wb"))
pickle.dump(autoencoder, open("autoencoder.pkl","wb"))
