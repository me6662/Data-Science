# %%
import pandas as pd
import numpy as np

### dataframe drop (특정 행, 열 제거)
# row : bike_drop = bike_sub.drop(index=558)
# column : bike_drop2 = bike_sub.drop(labels="season", axis=1)

### set_index : 칼럼을 인덱스로 사용
# bike_sub = bike_sub.set_index("datetime")

### pd.concat : 바인딩
# bike_1 = bike.iloc[:3, :4]
# bike_2 = bike.iloc[5:8, :4]
# pd.concat([bike_1, bike_2]) # 세로로 바인딩
# pd.concat([bike_1, bike_2.reset_index()],  axis= 1) # 가로로 바인딩

### pd.merge : 조인
# pd.merge(left= df_A, right= df_B, left_on='member', right_on='name', how='inner')
# lefton, right on 조인할 칼럼 명 써줌
# how => inner : 공통된 행만, left / right : 해당 df 의 행은 모두유지
# 칼럼명이 같으면 left 에 _x, right 에 _y 가 붙는다!

# %%
bike = pd.read_csv("bike.csv")
bike.head(2)
# %%
bike_sub = bike.sample(n=4, random_state=123)
bike_sub.reset_index()
# %%

# %%
bike_sub
bike_drop = bike_sub.drop(index=558)
bike_drop
# %%
bike_drop2 = bike_sub.drop(labels="season", axis=1)
bike_drop2
# %%
bike_sub = bike_sub.reset_index(drop=True)
# %%
bike_sub
# %%
bike_sub = bike_sub.set_index("datetime")
bike_sub
# %%
bike_sub = bike_sub.reset_index()
# %%
bike_sub
# %%
bike_1 = bike.iloc[:3, :4]
bike_2 = bike.iloc[5:8, :4]
pd.concat([bike_1, bike_2])
# %%
pd.concat([bike_1, bike_2.reset_index()], axis=1)


# %%
df_A = pd.read_csv("join_data_group_members.csv")
df_B = pd.read_csv("join_data_member_room.csv")

pd.merge(left=df_A, right=df_B, left_on="member", right_on="name", how="inner")
# %%
pd.merge(left=df_A, right=df_B, left_on="member", right_on="name", how="right")

# %%
df_join = pd.merge(left=df_A, right=df_B, left_on="member", right_on="name", how="left")
df_join.isna().sum(axis=1)
# %%
bike = pd.read_csv("bike.csv")
bike["datetime"] = pd.to_datetime(bike["datetime"])
bike["hour"] = bike["datetime"].dt.hour
bike.head(2)

# %%
bike_s2 = bike.loc[bike["season"] == 2, :]
bike_s4 = bike.loc[bike["season"] == 4, :]

# %%
bike_s2_agg = bike_s2.groupby("hour")["registered"].mean().reset_index()
bike_s4_agg = bike_s4.groupby("hour")["registered"].mean().reset_index()

# %%
bike_agg_bind = pd.concat([bike_s2_agg, bike_s4_agg], axis=1)
bike_agg_bind
# %%
bike_agg_bind = bike_agg_bind.iloc[:, [0, 1, 3]]
bike_agg_bind
# %%
bike_agg_bind.columns = ["hour", "regS2", "regS4"]
# %%
bike_agg_bind
# %%
bike_agg_bind["diff"] = bike_agg_bind["regS2"] - bike_agg_bind["regS4"]
bike_agg_bind
# %%
bike_agg_bind.iloc[[bike_agg_bind["diff"].idxmax()], :]
# %%
bike_agg_bind["diffabs"] = bike_agg_bind["diff"].abs()

# %%
bike_agg_bind.iloc[[bike_agg_bind["diffabs"].idxmax()], :]
# %%
bike["date"] = bike["datetime"].dt.date
bike
# %%
bike_h100 = bike.groupby("date")["humidity"].max().reset_index()
bike_h100
# %%
bike_h100 = bike_h100.loc[bike_h100["humidity"] == 100, :]
bike_h100
# %%
len(bike_h100)
# %%
bike_join = pd.merge(
    left=bike, right=bike_h100, left_on="date", right_on="date", how="inner"
)
bike_join
# %%
bike_join_temp_up30 = bike_join.loc[bike_join["temp"] > 30,]

bike_join_temp_up30["count"].mean()
# %%
