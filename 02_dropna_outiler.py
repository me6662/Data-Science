# %%

### 결측치 처리 isna, fillna, dropna
### 통계 함수 sum, max, min, mean, median, quantile, var, std , abs, round
### 조건으로 이상치 제거  df.loc[cond1 | cond2, ] , df.loc[cond1 & cond2, ]

import pandas as pd
import numpy as np

# %%
df = pd.read_csv("iris_missing.csv")
df.head(2)
# %%
df_na = df.head(7)
df_na
# %%
df_na["Sepal_Length"].isna()
# %%
df_na["Sepal_Length"].notna()
# %%
df_na["Sepal_Length"].isna().sum()
# %%
df_na["Sepal_Length"].notna().sum()
# %%
df_na.isna().sum(axis=1)
df_na
# %%
df_na = df_na.fillna(value={"Sepal_Length": 99999})
df_na
# %%
df_na = df_na.fillna(value={"Sepal_Length": 99999, "Sepal_Width": 88888})
df_na

# %%
df_na = df_na.fillna(value=12345)
df_na
# %%
df_na = df.head(10)
df_na
# %%
df_na = df_na.fillna(value={"Sepal_Length": df_na["Sepal_Length"].mean()})
df_na
# %%
df_na.dropna()
# %%
df_na.dropna(how="all")  # how='any'
# %%
df_na.iloc[:, :-1]  # 마지막 칼럼 제거방법
# %%
df_na.iloc[:, :-1].dropna(how="all")
# %%
SL = "Sepal_Length"
df[SL].max()
# %%
df[SL].min()
# %%
df[SL].mean()
# %%
print(df[SL].median())
print(df[SL].quantile(0.5))
print(df[SL].quantile(q=0.25))
# %%
print(df[SL].quantile(q=0.25))
print(df[SL].quantile(q=0.5))
print(df[SL].quantile(q=0.75))
# %%
df.iloc[:, :-1].isna().sum()
# %%
df.iloc[:, :-1].isna().sum().sum()
# %%
df_swidth = df["Sepal_Width"]
df_swidth
# %%
df_swidth = df_swidth.fillna(value=df["Sepal_Width"].mean())
# %%
print(df_swidth.var())
print(df_swidth.std())

# %%
# 이상치제거
df = pd.read_csv("iris.csv")
df.head(5)
# %%
sl_mean = df["Sepal.Length"].mean()
sl_std = df["Sepal.Length"].std()
print(sl_mean)
print(sl_std)
# %%
cond1 = df["Sepal.Length"] < (sl_mean - 1.5 * sl_std)
cond2 = df["Sepal.Length"] > (sl_mean + 1.5 * sl_std)
df_out = df.loc[cond1 | cond2, :]
len(df_out)
# %%
