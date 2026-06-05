import pandas as pd
df = pd.read_csv('C:/creditcard.csv')
print(df)

df.columns

df.head()

df.info()

df.describe()

df['Amount'].count()

# Mean of all V features
df["V_mean"] = df[[f"V{i}" for i in range(1,26)]].mean(axis=1)

# Standard deviation
df["V_std"] = df[[f"V{i}" for i in range(1,26)]].std(axis=1)

# Max & Min
df["V_max"] = df[[f"V{i}" for i in range(1,26)]].max(axis=1)
df["V_min"] = df[[f"V{i}" for i in range(1,26)]].min(axis=1)

# Range
df["V_range"] = df["V_max"] - df["V_min"]

# Sum of all V features
df["V_sum"] = df[[f"V{i}" for i in range(1,26)]].sum(axis=1)

# Absolute sum
df["V_abs_sum"] = df[[f"V{i}" for i in range(1,26)]].abs().sum(axis=1)

df["V1_V2"] = df["V1"] * df["V2"]
df["V3_V4"] = df["V3"] * df["V4"]
df["V5_V6"] = df["V5"] * df["V6"]

for i in range(1, 6):
    df[f"V{i}_sq"] = df[f"V{i}"] ** 2

df["V_outlier_count"] = (df[[f"V{i}" for i in range(1,26)]].abs() > 2).sum(axis=1)

df["V_positive_count"] = (df[[f"V{i}" for i in range(1,26)]] > 0).sum(axis=1)
df["V_negative_count"] = (df[[f"V{i}" for i in range(1,26)]] < 0).sum(axis=1)

df["V_pos_neg_ratio"] = df["V_positive_count"] / (df["V_negative_count"] + 1)

df.corr()['Class'].sort_values(ascending=True)
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from  sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

from sklearn.feature_selection import SelectKBest,f_classif
X = df.drop('Class',axis=1)
y = df['Class']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42,stratify = y)
scaler = StandardScaler()

X_train_scaler = scaler.fit_transform(X_train,y_train)
X_test_scaler = scaler.transform(X_test)



pipeline = Pipeline([
    ('smote',SMOTE(random_state = 42)),
    ('scaler',StandardScaler()),
    ('model',RandomForestClassifier(class_weight = 'balanced',random_state = 42))
    ])


grid_params ={
    'model__n_estimators':[200,300],
    'model__max_depth':[5,7],
    'model__min_samples_split':[2,5],
    'model__min_samples_leaf':[1,2]
}

from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(
    pipeline,
    param_distributions = grid_params,
    n_iter = 100,
    scoring='f1',
    cv = 3,
    n_jobs = -1,
    verbose = 2
)

random_search.fit(X_train,y_train)

print(f'Best Parameter {random_search.best_params_}')

best_model = random_search.best_estimator_

y_prob = best_model.predict_proba(X_test)[:,1]

y_pred = (y_prob > 0.2).astype(int)

from sklearn.metrics import classification_report,recall_score,accuracy_score,roc_auc_score,precision_score
print(f'Classification Report {classification_report(y_test,y_pred)}')
print(f'Recall Score {recall_score(y_test,y_pred)}')
print(f'Accuracy Score {accuracy_score(y_test,y_pred)}')
print(f'ROC AUC Score {roc_auc_score(y_test,y_prob)}')
print(f'Precision {precision_score(y_test,y_pred)}')

# import pickle
# with open('all_features.pkl','wb') as f:
#     pickle.dump(X.columns.to_list(),f)



# with open('fraud_model2.pkl','wb') as f:
#     pickle.dump(best_model,f)