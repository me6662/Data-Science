# %%
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error as mse

### KNN 분류기 (회귀에도 씀)
# K-최근접 이웃(K-NN, K-Nearest Neighbor) 알고리즘은 가장 간단한 머신러닝 알고리즘으로,
# 분류(Classification) 알고리즘이다.
# 비슷한 특성을 가진 데이터는 비슷한 범주에 속하는 경향이 있다는 가정하에 사용한다.
# 주변의 가장 가까운 K개의 데이터를 보고 데이터가 속할 그룹을 판단하는 알고리즘이 K-NN 알고리즘이다.
# KMeans 는 비슷한 애들끼리 묶어서 군집을 만드는것 , 이거는 주변 이웃 데이터 경향성을 보고 나를 판단함
# 이웃은 유클리디안 거리가 가장 가까운 n 개를 고르게 됨
# 테스트 데이터가 들어오면 학습데이터들이랑 거리를 구한다음 가까운 n개 골라서 결과결정
# 적절한 K 선택이 필요하고, 학습은 빠른대신에 데이터가 많아지면 분류가 느림.
# 성능이 보통 조타

# 분류
# model = KNeighborsClassifier(n_neighbors=3)  # 최근접 이웃을 3개까지 본다
# model.fit(X=df.iloc[:, :4], y=df["is_setosa"])
# pred = model.predict(df.iloc[:, :4])
# accuracy_score(y_true=df['is_setosa'], y_pred=pred)

# 회귀
# model2 = KNeighborsRegressor(n_neighbors=3)
# model2.fit(X=df.iloc[:, :3], y=df["Petal.Width"])
# pred = model2.predict(df.iloc[:, :3])
# pred
# mse_ = mse(y_true=df["Petal.Width"], y_pred=pred)
# rmse_ = mse_**0.5


# %%
df = pd.read_csv("iris.csv")
df["is_setosa"] = (df["Species"] == "setosa") + 0

model = KNeighborsClassifier(n_neighbors=3)  # 최근접 이웃을 3개까지 본다
model.fit(X=df.iloc[:, :4], y=df["is_setosa"])
pred = model.predict(df.iloc[:, :4])
accuracy_score(y_true=df["is_setosa"], y_pred=pred)
# %%
model2 = KNeighborsRegressor(n_neighbors=3)
model2.fit(X=df.iloc[:, :3], y=df["Petal.Width"])
pred = model2.predict(df.iloc[:, :3])
pred
mse_ = mse(y_true=df["Petal.Width"], y_pred=pred)
rmse_ = mse_**0.5

# %%
df = pd.read_csv("diabetes.csv")
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, train_size=0.7, random_state=123)
model = KNeighborsClassifier()
model.fit(
    X=df_train.loc[:, ["Pregnancies", "Glucose", "BloodPressure"]],
    y=df_train["Outcome"],
)
# %%
pred = model.predict(df_test.loc[:, ["Pregnancies", "Glucose", "BloodPressure"]])
accuracy_score(y_true=df_test["Outcome"], y_pred=pred)
# %%
df = pd.read_csv("diabetes.csv")
df["is_preg"] = (df["Pregnancies"] > 0) + 0
# %%

df_train, df_test = train_test_split(df, train_size=0.8, random_state=123)
X_cols = ["is_preg", "Glucose", "BloodPressure", "Insulin", "BMI"]
neighbors = [3, 5, 10, 20]
accs = []
for n_n in neighbors:
    model = KNeighborsClassifier(n_neighbors=n_n)
    model.fit(X=df_train.loc[:, X_cols], y=df_train["Outcome"])
    pred = model.predict(df_test.loc[:, X_cols])
    acc_sub = accuracy_score(y_pred=pred, y_true=df_test["Outcome"])
    accs = accs + [acc_sub]

df_score = pd.DataFrame({"neighbors": neighbors, "accs": accs})
df_score["accs"] = df_score["accs"].round(2)
df_score
# %%
from sklearn.metrics import mean_squared_error

X_cols = ["is_preg", "Glucose", "BloodPressure", "Insulin"]
neighbors = [3, 5, 10, 20]
rmses = []
for n_n in neighbors:
    model = KNeighborsRegressor(n_neighbors=n_n)
    model.fit(X=df_train.loc[:, X_cols], y=df_train["BMI"])
    pred = model.predict(df_test.loc[:, X_cols])
    rmse_sub = mean_squared_error(y_pred=pred, y_true=df_test["BMI"]) ** 0.5
    rmses = rmses + [rmse_sub]

df_score = pd.DataFrame({"neighbors": neighbors, "rmses": rmses})
df_score["rmses"] = df_score["rmses"].round(3)
df_score
# %%
