# %%
import pandas as pd
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

### LinearReression , ols 활용
# model = ols(formula = "PL ~ PW", data = df).fit()  # PL : X, PL : Y
# model.summary()
# df['pred'] = model.predict(df)

### LinearReression , sklearn 활용
# model = LinearRegression().fit(X = df[['PL']] ,y = df[['PW']]) # data frame 형식으로 넣어줘야 해서 [[]] 로함
# model.coef_ # 기울기
# model.intercept_ # 절편
# model.predict(df[['PL']])

### 에러계산
# mean_absolute_error(y_true = df["PL"], y_pred= df["PW"])
# mean_squared_error(y_true = df["PL"], y_pred=df["PW"])
# mean_squared_error(y_true = df["PL"], y_pred=df["PW"]) ** 0.5 # RMSE


# %%
df = pd.read_csv("iris.csv")
df.columns = ["SL", "SW", "PL", "PW", "species"]
df
# %%
model = ols(formula="SL~SW", data=df).fit()
model.summary()
# %%
model = ols(formula="PL ~ PW", data=df).fit()
model.summary()  # 좋은모델
# %%
model.predict(df.iloc[:6, :])
df["pred"] = model.predict(df)
df
# %%

model = LinearRegression().fit(
    X=df[["PL"]], y=df[["PW"]]
)  # data frame 형식으로 넣어줘야 해서 [[]] 로함
model.coef_
# %%
model.intercept_

# %%
model.predict(df[["PL"]])
# %%
mean_absolute_error(y_true=df["PL"], y_pred=df["PW"])
# %%
mean_squared_error(y_true=df["PL"], y_pred=df["PW"])
# %%
mean_squared_error(y_true=df["PL"], y_pred=df["PW"]) ** 0.5
# %%
from sklearn.model_selection import train_test_split

df = pd.read_csv("bike.csv")
df_train, df_test = train_test_split(df, train_size=0.7, random_state=123)
model = ols("registered ~ temp", data=df_train).fit()

# %%
model = ols("casual ~ atemp", data=df_train).fit()
pred = model.predict(df_test)
pred[:4]
mean_squared_error(y_pred=pred, y_true=df_test["casual"]) ** 0.5
# %%
