# %%
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

### 나이브 베이즈 : 분류문제 , X 데이터들을 가지고 확률 기반해서 분류함.
# df["is_setosa"].value_counts(normalize=True) # 이거를 사전확률 이라고 정의함.
# model = GaussianNB().fit(X=df.iloc[:, :4], y=df["is_setosa"])
# model.class_prior_ # 입력으로 먹인 데이터 확률 정보
# model.theta_
# pred = model.predict_proba(df.iloc[:, :4])
# pred = pred[:, 1] # 여기 중요함. (뒤에거가 확률임)
# pred_class = (pred>0.999) + 0
# accuracy_score(y_true = df['is_setosa'], y_pred= pred_class)

### 파이썬에서 몫 구하기 9 // 4 = 2 임...

# %%
df = pd.read_csv("iris.csv")
df["is_setosa"] = (df["Species"] == "setosa") + 0
# %%
df["is_setosa"].value_counts(normalize=True)
# %%
model = GaussianNB().fit(X=df.iloc[:, :4], y=df["is_setosa"])
model.class_prior_
# %%
model.theta_

# %%
pred = model.predict_proba(df.iloc[:, :4])
pred = pred[:, 1]

# %%
pred_class = (pred > 0.999) + 0

# %%
accuracy_score(y_true=df["is_setosa"], y_pred=pred_class)
# %%
df = pd.read_csv("diabetes.csv")
model = GaussianNB().fit(
    X=df.loc[:, ["Glucose", "BloodPressure", "Age"]], y=df["Outcome"]
)
pred = model.predict_proba(df.loc[:, ["Glucose", "BloodPressure", "Age"]])
pred_class = (pred[:, 1] > 0.5) + 0
# %%
accuracy_score(y_true=df["Outcome"], y_pred=pred_class)

# %%

df = df.loc[df["BMI"] > 0,]
df["Age_g"] = (df["Age"] // 10) * 10
df["is_preg"] = (df["Pregnancies"] > 0) + 0
df.head(2)

# %%
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, train_size=0.8, random_state=123)

model = GaussianNB().fit(
    X=df_train.loc[:, ["is_preg", "Age_g", "BMI", "Glucose"]], y=df_train["Outcome"]
)
pred = model.predict_proba(df_test.loc[:, ["is_preg", "Age_g", "BMI", "Glucose"]])
accuracy_score(y_true=df_test["Outcome"], y_pred=(pred[:, 1] > 0.5) + 0)
# %%

from sklearn.linear_model import LogisticRegression

model2 = LogisticRegression().fit(
    X=df_train.loc[:, ["is_preg", "Age_g", "BMI", "Glucose"]], y=df_train["Outcome"]
)
pred2 = model2.predict_proba(df_test.loc[:, ["is_preg", "Age_g", "BMI", "Glucose"]])
pred2 = pred2[:, 1]
pred2_class = (pred2 > 0.5) + 0
accuracy_score(y_pred=pred2_class, y_true=df_test["Outcome"])
# %%
