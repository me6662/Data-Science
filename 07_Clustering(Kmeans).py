# %%
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, StandardScaler

### 정규화 : MinMaxScaler
# nor_minmax = MinMaxScaler()
# nor_minmax.fit(df_1.iloc[:, :-1])
# minmax_arr = nor_minmax.transform(df_1.iloc[:, :-1])
# df_minmax = pd.DataFrame(minmax_arr, columns=df_1.columns[:4])

### KMeans
# model = KMeans(n_clusters=3, random_state=123).fit(df.iloc[:,:-1])
# model.labels_
# model.cluster_centers_
# df['cluster'] = model.labels_
# df.groupby('cluster').mean()

### value counts : 칼럼 데이터 별 갯수 세줌
# df_sub["cluster"].value_counts()

### transpose : 행 열 교환
# df_centers = df_centers.transpose()

# %%
df = pd.read_csv("iris.csv")
df_1 = df.head()
df_2 = df.tail(1)
# %%
nor_minmax = MinMaxScaler()
nor_minmax.fit(df_1.iloc[:, :-1])
# %%
minmax_arr = nor_minmax.transform(df_1.iloc[:, :-1])
# %%
df_minmax = pd.DataFrame(minmax_arr, columns=df_1.columns[:4])
df_minmax

# %%

model = KMeans(n_clusters=3, random_state=123).fit(df.iloc[:, :-1])
# %%
model.labels_
# %%
model.cluster_centers_
# %%
df["cluster"] = model.labels_

# %%
df.groupby("cluster").mean()
# %%
df = pd.read_csv("diabetes.csv")
df
# %%
df_sub = df.loc[df["BMI"] != 0,]
df_sub
# %%
model = KMeans(n_clusters=4, random_state=123).fit(df_sub)

# %%
df_sub["cluster"] = model.labels_
# %%
df_sub
# %%
df_sub["cluster"].value_counts()
# %%
df_sub.groupby("cluster")["Insulin"].mean().reset_index()
# %%

nor_minmax = MinMaxScaler().fit(df_sub)
df_sub_nor = nor_minmax.transform(df_sub)
df_sub_nor = pd.DataFrame(df_sub_nor, columns=df_sub.columns)
df_sub_nor

# %%
model = KMeans(n_clusters=4, random_state=123).fit(df_sub_nor)
df_sub["cluster"] = model.labels_
df_sub
# %%
df_sub["cluster"].value_counts()
# %%
df_sub.groupby("cluster")["Age"].mean().reset_index()
# %%
df_sub = df.loc[df["BMI"] != 0,]
model = KMeans(n_clusters=3, random_state=123).fit(df_sub)
df_centers = pd.DataFrame(model.cluster_centers_, columns=df_sub.columns)
df_centers
# %%
model.cluster_centers_
# %%
df_centers = df_centers.transpose()
# %%
df_centers
# %%
