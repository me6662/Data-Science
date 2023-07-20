# %%
### %config Completer.use_jedi = False
###  df.sample 사용법
### df.groupby 사용법
### train_test_split 사용법

import pandas as pd
import numpy as np

# %%


from sklearn.model_selection import train_test_split

df = pd.read_csv("bike.csv")
df.head(2)

# %%

df.sample(n=2, random_state=34)


# %%
df["season"].unique()
# %%
df["season"].nunique()
# %%
df.groupby("season").sample(n=2, random_state=34)

# %%
df.sample(frac=0.005, random_state=34)
# %%
df.info()
# %%
len(df.sample(frac=0.005, random_state=34))
# %%
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, train_size=0.7, random_state=123)
df_train.head(2)
# %%
len(df_test)
# %%
len(df_train)
# %%
len(df.sample(frac=0.0123))
# %%
df.groupby("season").sample(frac=0.05).shape
# %%
df_train, df_test = train_test_split(df, train_size=0.8, random_state=123)
max_temp = df_train["temp"].max()
max_temp
# %%
