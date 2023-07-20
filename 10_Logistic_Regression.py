# %%
import pandas as pd
import numpy as np
from statsmodels.api import Logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

### Logit 함수 : 회귀인데 분류 맞춤
# model = Logit(endog=df["is_setosa"], exog=df.iloc[:, :2]).fit() # endog : X, exog : y
# model.params
# np.exp(model.params) # odds ratio 구하라면 이거해야함.
# model.tvalues
# pred = model.predict(df.iloc[:3, :2])

### LogisticRegression 함수
# model = LogisticRegression(random_state=123)
# model.fit(X = df.iloc[:, :2], y = df['is_setosa'])
# model.coef_
# model.intercept_
# model.predict_proba(df.iloc[:3,:2]) ## 이거 pred = pred[:, 1] 로 해야함. 뒤엑게 확률임.
# model.predict(df.iloc[:3,:2])

### 로지스틱 평가
# roc_auc_score(y_true=df["is_setosa"], y_score=pred)
# accuracy_score(y_true=df['is_setosa'], y_pred= (pred>0.5) + 0)

# %%
df = pd.read_csv("iris.csv")
df["is_setosa"] = (df["Species"] == "setosa") + 0
# %%
df
# %%
model = Logit(endog=df["is_setosa"], exog=df.iloc[:, :2]).fit()
model.params
# %%
model.tvalues
# %%
pred = model.predict(df.iloc[:3, :2])

# %%
model = LogisticRegression(random_state=123)
model.fit(X=df.iloc[:, :2], y=df["is_setosa"])

# %%
model.coef_
# %%
model.intercept_
# %%
model.predict_proba(df.iloc[:3, :2])

# %%
pred = model.predict(df.iloc[:3, :2])
# %%

pred = model.predict_proba(df.iloc[:, :2])
pred = pred[:, 1]
pred
# %%
roc_auc_score(y_true=df["is_setosa"], y_score=pred)
# %%
accuracy_score(y_true=df["is_setosa"], y_pred=(pred > 0.5) + 0)
# %%


df = pd.read_csv("diabetes.csv")
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, train_size=0.8, random_state=123)

model = Logit(
    endog=df_train["Outcome"],
    exog=df_train.loc[:, ["BloodPressure", "Glucose", "BMI", "Insulin"]],
).fit()
# %%
pred = model.predict(df_test.loc[:, ["BloodPressure", "Glucose", "BMI", "Insulin"]])
# %%
pred[:4]
# %%
pred_class = (pred > 0.5) + 0
pred_class
# %%
accuracy_score(y_true=df_test["Outcome"], y_pred=pred_class)

# %%
model = Logit(endog=df["Outcome"], exog=df.loc[:, ["Glucose", "BMI", "Age"]]).fit()
model.params
np.exp(model.params)
# %%
