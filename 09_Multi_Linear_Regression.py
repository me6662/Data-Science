# %%
import pandas as pd
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif

### vif 구하기 (10이상으로 나오면 변수들끼리 연관성이 있는것이므로 안쓴다.)
# formular = "casual ~ " + " + ".join(df_sub.columns[:-1])
# y, X = dmatrices(formular, data=df_sub, return_type="dataframe")
# df_vif = pd.DataFrame()
# df_vif['colname'] = X.columns
# df_vif['VIF'] = [vif(X.values, i) for i in range(X.shape[1]) ]
# df_vif
## 꼭 이렇게 해야 intercept 라는 변수까지 나오고 계산되므로 꼭 dmatrices 써야함.

### df.values -> 값을 array 로 변환함.

# %%
df = pd.read_csv("bike.csv")
df_sub = df.loc[:, "season":"casual"]
# %%
formular = "casual ~ " + " + ".join(df_sub.columns[:-1])
# %%
y, X = dmatrices(formular, data=df_sub, return_type="dataframe")
# %%
df_vif = pd.DataFrame()
df_vif["colname"] = X.columns
df_vif["VIF"] = [vif(X.values, i) for i in range(X.shape[1])]
df_vif

# %%
df_sub = pd.concat([df.loc[:, "season":"temp"], df.loc[:, "humidity":"casual"]], axis=1)
df_sub
# %%
formular = "casual ~ " + " + ".join(df_sub.columns[:-1])
y, X = dmatrices(formular, df_sub, return_type="dataframe")
# %%
df_vif = pd.DataFrame()
df_vif["colname"] = X.columns
df_vif["VIF"] = [vif(X.values, i) for i in range(X.shape[1])]
df_vif
# %%
df = pd.read_csv("diamonds.csv")
df_sub = df.iloc[:, [6, 0, 4, 5, 7, 8, 9]]

# %%
df_sub
# %%

y, X = dmatrices(
    "price ~ " + " + ".join(df_sub.columns[1:]), data=df_sub, return_type="dataframe"
)

df_vif = pd.DataFrame()
df_vif["vars"] = X.columns
df_vif["VIF"] = [vif(X.values, i) for i in range(X.shape[1])]
df_vif
# %%
df_vif
# %%
