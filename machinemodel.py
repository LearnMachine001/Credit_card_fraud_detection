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

X_temp = df.copy()
y_temp = df["is_fraud"]

from sklearn.model_selection import  train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_temp,y_temp,test_size=0.2,random_state=42,stratify=y_temp)

location_risk_map = X_train.groupby("location")["is_fraud"].mean()
df["location_risk"] = df["location"].map(location_risk_map)
df["location_risk"].fillna(df["location_risk"].mean(),inplace=True)

df["is_high_risk"] = ((df["is_international"] ==1) & (df["amount"] > 200)).astype(int)

df["amount_per_fraud"] = df["amount"] * df["previous_fraud_count"]

df = pd.get_dummies(df,columns=["device_type"],drop_first=True)

df["amount_deviation"] = df["amount"] - df["avg_amount_per_user"]
df["amount_ratio"] = df["amount"] / (df["avg_amount_per_user"] + 1)

df["transactions_per_hour"] = df["user_transaction_count"] / 24

df["user_fraud_ratio"] = df["previous_fraud_count"] / (df["user_transaction_count"] + 1)

df["is_weekend"] = (df["hour"] >= 5).astype(int)
df["is_peak_hour"] = ((df["hour"] >= 9) & (df["hour"] <= 18)).astype(int)

df["risk_score"] = (
    df["fraud_risk"] +
    df["location_risk"] +
    df["is_high_risk"]
)

df["amount_location_risk"] = df["amount"] * df["location_risk"]
df["amount_user_risk"] = df["amount"] * df["user_fraud_ratio"]



df.corr(numeric_only=True)["is_fraud"].sort_values(ascending=True)

important_features = [
    "amount","hour","is_night","high_amount",
    "is_international","fraud_risk","location_risk",
    "is_high_risk","avg_amount_per_user",
    "amount_deviation","amount_ratio",
    "transactions_per_hour","user_fraud_ratio",
    "is_weekend","is_peak_hour",
    "risk_score",
    "amount_location_risk","amount_user_risk"
]
X = df[important_features]
y = df["is_fraud"]

#Train model on given data.
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42, stratify= y)

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
model = XGBClassifier(class_weight = 'balanced')
model.fit(X_train,y_train)


#use pipeline for advance prediction
from imblearn.pipeline import Pipeline
pipeline = Pipeline([
                    ('model',XGBClassifier())
                    ])

pipeline.fit(X_train,y_train)

#hyperparameter tunning using gridsearchcv
params_grid = {
            "model__n_estimators":[400],
            "model__max_depth":[10],
            "model__learning_rate":[0.03],
            "model__subsample":[0.8],
            "model__colsample_bytree":[0.8],
            "model__scale_pos_weight":[25]
}

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(pipeline,params_grid,cv=5,scoring="recall",n_jobs=-1)

grid.fit(X_train,y_train)

print("Best Parameter ", grid.best_params_)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,f1_score,roc_curve,auc,recall_score
best_model = grid.best_estimator_
y_prob = best_model.predict_proba(X_test)[:,1]
fpr,tpr,threshold = roc_curve(y_test,y_prob)

roc_auc = auc(fpr,tpr)
print(f"ROC-AUC {roc_auc}")

plt.figure()
plt.plot(fpr,tpr,label=f"ROC CURVE (AUC = {roc_auc:.4f})")
plt.plot([0,1],[0,1],linestyle='--')
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curve")
plt.legend()
plt.show()

best_index = (tpr-fpr).argmax()
best_threshold = threshold[best_index]

print(f"Best threshold{best_threshold}")

y_pred = (y_prob > best_threshold).astype(int)


print(f"classification Report {classification_report(y_test,y_pred)}")
print(f"confusion_matrix {confusion_matrix(y_test,y_pred)}")
print(f"Accuracy {accuracy_score(y_test,y_pred)}")
print(f"F1_Score {f1_score(y_test,y_pred)}")
print(f"Racall Score {recall_score(y_test,y_pred)}")


import pickle
with open('model.pkl','wb') as file:
    pickle.dump(best_model,file)

print("Model save successfully ")


# print("model save succefully")
# amount=float(input("Enter amount"))
# hour=float(input("Enter hour"))
# is_night=float(input("Enter is_night"))
# high_amount=bool(input("Enter high_amount"))
# is_international=bool(input("Enter is_international"))
# fraud_risk=bool(input("Enter fraud_risk"))
# location_risk=bool(input("Enter location_risk"))
# is_high_risk=bool(input("Enter is_high_risk"))
# avg_amount_per_user=float(input("Enter avg_amount_per_user"))
# user_input =np.array([amount,hour,is_night,high_amount,is_international,fraud_risk,location_risk,is_high_risk,avg_amount_per_user].reshape(1,-1))
# prob = best_model.predict_proba(user_input)[0][1]

# user_prediction = best_model.predict([user_input])

# print(f"user prediction{user_prediction[0]}")
