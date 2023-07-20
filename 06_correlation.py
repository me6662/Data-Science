# %%
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau

### df.corr() : 상관계수 피어슨
# df[["casual", "registered", "count"]].corr(method="kendall")
# df[["casual", "registered", "count"]].corr(method="spearman")

### pearsonr(df['casual'], df['registered']), stats 상관분석

# %%
df = pd.read_csv("bike.csv")
df
# %%
df.corr()
# %%
df[["casual", "registered", "count"]].corr()

# %%
pearsonr(df["casual"], df["registered"])
# %%
df[["temp", "atemp", "humidity", "casual"]].corr().round(2)
# %%
df[["temp", "atemp", "humidity", "casual"]].corr().round(2).min().max()
# %%
df_corr = df[["season", "atemp", "casual"]].groupby("season").corr()
df_corr = df_corr.reset_index()
# %%
df_corr
# %%
df_corr.loc[df_corr["atemp"] < 1, :]

# %%
df["is_sunny"] = (df["weather"] == 1) + 0
# %%
df
# %%
df["is_sunny"].unique()
# %%
df_corr = df.groupby("is_sunny")[["temp", "casual"]].corr()
# %%
df_corr
# %%
