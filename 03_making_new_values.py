# %%
import pandas as pd
import numpy as np

### np.where 를 활용 'is_A' 0 - 1 변수 만들기
### pd.crosstab 으로 가능한 조합 갯수 찾기
### (조건) + 0 , (조건) * 1 로 np.where 와 같은 효과 적용
### df.rename 으로 칼럼명 변경
### df.apply 로 커스텀 함수 적용 bike_sub.apply(func=lambda x: round(x.mean()))
### astype 으로 자료형 변환 bike_sub["casual"].astype("str") + "대"
### 날짜 변수 자르기 bike['datetime'].str.slice(0,4)
### pd.to_datetime 함수 활용 bike_time = pd.to_datetime(bike["datetime"]), bike_time.dt.year
### get_dummies 통한 one hot encoding
# bike_dum = pd.get_dummies(data= bike, columns=['season'] , drop_first= True)

### pd.series 일때, 결측치 제거
# bike_agg = bike.groupby("date")["casual"].max()
# bike_agg
# bike_agg_up25 = bike_agg[bike_agg > 25]

### series to dataframe
# df = pd.DataFrame(my_series)
# 또는 시리즈에 바로 reset_index() 하면 dataframe 됨.

### idxmax()
# bike_hour.loc[bike_hour["registered"].idxmax(), : ]

# apply 커스텀  람다식
# bike_sub.apply(func=lambda x: round(x.mean()))
# bike_sub.apply(func=lambda x: x.mean())
# bike.sub.apply(custon_func)
# def custom_func (x) :  return x + 1

# %%
df = pd.read_csv("iris.csv")
df.head()
# %%
df["is_setosa"] = np.where(df["Species"] == "setosa", 1, 0)
df.head()
# %%
df.tail()
# %%
pd.crosstab(df["Species"], df["is_setosa"])
# %%
df["is_setosa2"] = (df["Species"] == "setosa") + 0
df["is_setosa3"] = (df["Species"] == "setosa") * 1
df.head()
# %%
df = df.rename(columns={"Sepal.Length": "SL"})
df
# %%
bike = pd.read_csv("bike.csv")
bike.head()
# %%
bike.columns
# %%
bike.iloc[:4, 9:11]  # only integer
# %%
bike.loc[:4, ["casual", "registered"]]
# %%
bike.loc[:4, "casual":"registered"]
# %%
bike_sub = bike.loc[:4, "casual":"registered"]
bike_sub.sum()
# %%
bike_sub.sum(axis=1)
# %%
bike_sub.apply(func=sum)
# %%
bike_sub.apply(func=sum, axis=1)

# %%
# 커스텀  람다식

bike_sub.apply(func=lambda x: round(x.mean()))
bike_sub.apply(func=lambda x: x.mean())
# %%
bike_sub["casual"].astype("str") + "대"
# %%
bike = pd.read_csv("bike.csv")
bike.head(2)
# %%
bike["datetime"][:3]
# %%
bike["datetime"].str.slice(0, 4)
# %%
bike_time = pd.to_datetime(bike["datetime"])
bike_time
# %%
print(bike_time.dt.year)
print(bike_time.dt.month)
print(bike_time.dt.hour)
print(bike_time.dt.weekday)
# %%
bike_dum = pd.get_dummies(data=bike, columns=["season"])
bike_dum.head()
# %%
bike_dum = pd.get_dummies(data=bike, columns=["season"], drop_first=True)
bike_dum.head()
# %%
bike["diff"] = bike["temp"] - bike["atemp"]
bike.head(2)
# %%
bike["diff"].abs().mean()
# %%
bike["datetime"] = pd.to_datetime(bike["datetime"])
bike["date"] = bike["datetime"].dt.date
# %%
bike.head()
# %%
bike_agg = bike.groupby("date")["casual"].max()
bike_agg
# %%
bike_agg_up25 = bike_agg[bike_agg > 25]
bike_agg_up25
# %%
bike2 = pd.DataFrame(bike_agg).reset_index()
bike2
# %%
bike2.loc[bike2["casual"] > 25, :]
# %%
bike.info()
# %%
bike["hour"] = bike["datetime"].dt.hour
bike.head(2)
# %%
bike_hour = bike.groupby("hour")["registered"].mean().reset_index()
bike_hour
# %%
bike_hour.loc[bike_hour["registered"] == bike_hour["registered"].max(), :]
# %%
bike_hour.loc[bike_hour["registered"].idxmax(), :]

# %%
