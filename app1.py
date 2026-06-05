import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('C:/credit_card_fraud_dataset.csv')


df.head()

df.describe()

df["is_fraud"].value_counts()

df.copy()

df.info()

fraud_percentage = (df["is_fraud"].value_counts(normalize=True)*100)
print(fraud_percentage)

df["is_fraud"].value_counts().plot(kind="bar")
plt.title("Normal vs Fraud transaction")
plt.show()

df["is_fraud"].mean()

print("fraud transactions",df["is_fraud"].value_counts()[1])
print("Normal transactions",df["is_fraud"].value_counts()[0])
print("Fraud ratio ",df["is_fraud"].mean())

df["log_amount"] = np.log1p(df["amount"])

print(df["log_amount"])

plt.boxplot(x=df["amount"])
plt.title("Boxplot of transaction amount")
plt.show()

plt.figure(figsize=(10,8))
plt.hist(df["amount"],bins=50)
plt.title("Distribution of amount")
plt.xlabel("Amount")
plt.ylabel("frequency")
# plt.xticks(10,30)
plt.show()

plt.scatter(range(len(df)),df["amount"])
plt.title("scatter plot of amount")
plt.xlabel("frequency")
plt.ylabel("amount")
plt.show()

# plt.boxplot(x="is_fraud",y="amount",data=df)
# plt.title("amount vs fraud")
# plt.show()

#correaltion
cor_relation = df.corr(numeric_only = True)
print(cor_relation)

#feature engineering

df["high_amount"] = (df["amount"] > 200).astype(int)


df["hour"] = df["transaction_time"]//3600


df["is_night"] = df["hour"].apply(lambda x: 1 if x < 6 else 0)

df["user_transaction_count"] = df.groupby("user_id")["transaction_id"].transform("count")
df["avg_amount_per_user"] = df.groupby("user_id")["amount"].transform("mean")

df["fraud_risk"] = df["previous_fraud_count"]/(df["user_transaction_count"] +1)

df["location_risk"] = df.groupby("location")["is_fraud"].transform("mean")

df["is_high_risk"] = ((df["is_international"] ==1) & (df["amount"] > 200)).astype(int)

df["amount_per_fraud"] = df["amount"] * df["previous_fraud_count"]

df = pd.get_dummies(df,columns=["device_type"],drop_first=True)


df.corr(numeric_only=True)["is_fraud"].sort_values(ascending=True)

from sklearn.linear_model import LogisticRegression
important_features =["amount","hour","is_night","high_amount","is_international",
    "fraud_risk","location_risk","avg_amount_per_user","user_transaction_count"]

X = df[important_features]
y = df["is_fraud"]

#Train model on given data.
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#use Standard scaler for convert data mean =0 std =1
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state = 42)
X_train_smote,y_train_smote = smote.fit_resample(X_train,y_train)

#use model for prediction
# model = LogisticRegression(class_weight = "balanced",max_iter = 1000)
#use xgbclassifier for predict fraud
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train_smote,y_train_smote)


#use pipeline for advance prediction
from imblearn.pipeline import Pipeline
pipeline = Pipeline([
                    ('smote',SMOTE(k_neighbors =3,random_state = 42)),
                    ('model',XGBClassifier(scale_pos_weight = 15))
                    ])

# pipeline.fit(X_train,y_train)
#hyperparameter tunning using gridsearchcv
params_grid = {
            "model__n_estimators":[200,400],
            "model__max_depth":[3,5],
            "model__learning_rate":[0.05,0.1],
            "model__subsample":[0.8,1.0]
}

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(pipeline,params_grid,cv=3,scoring="f1",n_jobs=1)

grid.fit(X_train,y_train)

print("Best Parameter ", grid.best_params_)

best_model = grid.best_estimator_


from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,f1_score,recall_score
y_prob = best_model.predict_proba(X_test)[:,1]
y_pred = (y_prob > 0.3).astype(int)
print(f"Y_prob {y_prob}")
print(f"classification Report {classification_report(y_test,y_pred)}")
print(f"confusion_matrix {confusion_matrix(y_test,y_pred)}")
print(f"Accuracy {accuracy_score(y_test,y_pred)}")
print(f"F1_Score {f1_score(y_test,y_pred)}")
print(f"Recall score {recall_score(y_test,y_pred)}")
    

import pickle
with open("model.pkl","wb") as file:
    pickle.dump(best_model,file)
print("model save successfully")


print("model save succefully")

