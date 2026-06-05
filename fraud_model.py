import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np

v_feature = [f"V{i}" for i in range(1,29)]

df = pd.read_csv('C:/creditcard.csv')
print(df)


df['Amount'].count()

# Mean of all V features
df["V_mean"] = df[[f"V{i}" for i in range(1,29)]].mean(axis=1)

# Standard deviation
df["V_std"] = df[[f"V{i}" for i in range(1,29)]].std(axis=1)

# Max & Min
df["V_max"] = df[[f"V{i}" for i in range(1,29)]].max(axis=1)
df["V_min"] = df[[f"V{i}" for i in range(1,29)]].min(axis=1)

# Range
df["V_range"] = df["V_max"] - df["V_min"]

# Sum of all V features
df["V_sum"] = df[[f"V{i}" for i in range(1,29)]].sum(axis=1)

# Absolute sum
df["V_abs_sum"] = df[[f"V{i}" for i in range(1,29)]].abs().sum(axis=1)

df["V1_V2"] = df["V1"] * df["V2"]
df["V3_V4"] = df["V3"] * df["V4"]
df["V5_V6"] = df["V5"] * df["V6"]

for i in range(1, 6):
    df[f"V{i}_sq"] = df[f"V{i}"] ** 2

df["V_outlier_count"] = (df[[f"V{i}" for i in range(1,29)]].abs() > 2).sum(axis=1)

df["V_positive_count"] = (df[[f"V{i}" for i in range(1,29)]] > 0).sum(axis=1)
df["V_negative_count"] = (df[[f"V{i}" for i in range(1,29)]] < 0).sum(axis=1)

df["V_pos_neg_ratio"] = df["V_positive_count"] / (df["V_negative_count"] + 1)

df["Amount_zscore"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()

df["is_high_amount"] = (df["Amount"] > df["Amount"].quantile(0.95)).astype(int)
df["Amount_to_mean"] = df["Amount"] / (df["Amount"].mean() + 1)
df["Amount_diff"] = df["Amount"] - df["Amount"].median()

df.corr()['Class'].sort_values(ascending=True)

################################ Using tensorflow here ##############################################
X = df.drop('Class',axis = 1)
y = df['Class']
#User input feature
df["Time_hour"] = df["Time"] / 3600  # convert seconds to hours
df["Amount_log"] = np.log1p(df["Amount"])  # log-transform Amount
df["Amount_hour_interaction"] = df["Amount"] * df["Time_hour"]
df["is_night"] = df["Time_hour"].apply(lambda x: 1 if x < 6 or x > 22 else 0)


derived_features = ["Amount", "Time_hour", "Amount_log", "Amount_hour_interaction", "is_night"]

features = v_feature + derived_features + [
    "Amount_zscore", "is_high_amount",
    "Amount_to_mean", "Amount_diff",
    "V_mean", "V_std", "V_range",
    "V_sum", "V_abs_sum",
    "V_outlier_count", "V_pos_neg_ratio"
]
X = df[features]
y = df["Class"]


#scaled data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42,stratify=y)


ae_features = v_feature + ["Amount","Time"]
#Using Standard Scaler for scaled data....
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[ae_features])
X_test_scaled = scaler.transform(X_test[ae_features])

#Train only on model data....

input_dim = X_train_scaled.shape[1]

autoencoder= tf.keras.Sequential([
    tf.keras.layers.Dense(32,activation = 'relu',input_shape=(input_dim,)),
    tf.keras.layers.Dense(16,activation = 'relu'),
    tf.keras.layers.Dense(32,activation = 'relu'),
    tf.keras.layers.Dense(input_dim,activation = 'linear')
    ])

autoencoder.compile(optimizer = 'adam',loss = 'mse')

X_normal = X_train_scaled[y_train==0]
autoencoder.fit(X_normal,X_normal,epochs=10,batch_size=256,shuffle=True,validation_split = 0.1)

# Convert scaled arrays to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=ae_features)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=ae_features)

#Train
train_pred = autoencoder.predict(X_train_scaled)
test_pred = autoencoder.predict(X_test_scaled)

#Test
X_train_scaled['recon_error'] = np.mean((X_train_scaled - train_pred) **2,axis=1)
X_test_scaled['recon_error'] = np.mean((X_test_scaled - test_pred) **2,axis = 1)


X_train["recon_error"] = X_train_scaled["recon_error"].values
X_test["recon_error"] = X_test_scaled["recon_error"].values



###########Using XGBoost here...............................
#Using xgboost model here and selectkbest
from sklearn.feature_selection import SelectKBest,f_classif
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
model = XGBClassifier()

scale_pos_weight = 10 * (len(y_train[y_train==0]) / len(y_train[y_train==1]))

pipeline = Pipeline([
    ('scaler',StandardScaler()),
    ('model',XGBClassifier(scale_pos_weight = scale_pos_weight,eval_metric ='logloss',random_state = 42))
    ])


#Model tunning for find best parameter
grid_params ={
    'model__n_estimators':[300],
    'model__max_depth':[5],
    'model__learning_rate':[0.05],
    'model__subsample':[0.8,1],
    'model__colsample_bytree':[0.8,1]
}

from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(
    pipeline,
    param_distributions = grid_params,
    n_iter = 10,
    scoring='recall',
    cv = 3,
    n_jobs = -1,
    verbose = 2
)

model.fit(X_train_scaled,y_train)

print(f'Best Parameter {random_search.best_params_}')

best_model = random_search.best_estimator_

y_prob = best_model.predict_proba(X_test_scaled)[:,1]
#Select best threshold for detect fraud
X_test_scaled["Amount"] = X_test["Amount"].values


from sklearn.metrics import precision_recall_curve

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)



from sklearn.metrics import f1_score
thresholds = np.arange(0.1, 0.9, 0.05)
best_f1 = 0
best_threshold = 0.25

for t in thresholds:
    temp_pred = (y_prob > t).astype(int)
    f1 = f1_score(y_test, temp_pred)
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

y_pred = (y_prob > best_threshold).astype(int)


from sklearn.metrics import classification_report,recall_score,accuracy_score,roc_auc_score,precision_score,confusion_matrix
print(f'Classification Report {classification_report(y_test,y_pred)}')
print(f'Recall Score {recall_score(y_test,y_pred)}')
print(f'Accuracy Score {accuracy_score(y_test,y_pred)}')
print(f'ROC AUC Score {roc_auc_score(y_test,y_prob)}')
print(f'Precision {precision_score(y_test,y_pred)}')
print(f'Confusion Matrix {confusion_matrix(y_test,y_pred)}')

# import pickle

# pickle.dump(best_model, open("fraud_model.pkl", "wb"))
# pickle.dump(features, open("all_features.pkl", "wb"))
# pickle.dump(best_threshold, open("best_threshold.pkl", "wb"))
# pickle.dump(scaler,open("scaler.pkl","wb"))
# pickle.dump(autoencoder,open("autoencoder.pkl","wb"))


