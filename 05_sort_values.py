# %%
import pandas as pd
import numpy as np

### pd.crosstab : 여러 칼럼 종류 조합이 몇개고 속하는 데이터가 몇개인지 확인가능
# pd.crosstab(dia['cut'],dia['clarity'])
# pd.crosstab(dia["cut"], dia["clarity"], normalize = True) # 정규화는 공짜임
# pd.crosstab(dia["cut"], dia["clarity"], normalize = 0) # row  별 비중
# pd.crosstab(dia["cut"], dia["clarity"], normalize = 1) # col 별 비중
# pd.crosstab(dia['cut'], dia['clarity'], values=dia['price'], aggfunc=pd.Series.mean) # 해당되는 데이터들의 3번 쨰 속성값 계산

### gropuby 여러개 : pd.crosstab 과 동일한 효과
# dia.groupby(['cut', 'clarity'])['price'].mean().reset_index()

### sort values : 오름 내림 차순 정렬
# dia_agg.sort_values('price', ascending=False)
# dia_agg.sort_values('price')
# dia_agg.sort_values(['cut', 'clarity'], ascending=[True, False])

### melt : 칼럼이 많은 경우 칼럼을 데이터로 녹여냄
# elec_melt = elec.melt(id_vars = ["YEAR", "MONTH", "DAY"]) # idvars : 기준이될 변수

### pivot : row 가 많은 경우 칼럼으로 변경 (melt 의 반대)
# elec_melt.pivot(index = ["YEAR", "MONTH", "DAY"], columns="variable", values = "value").reset_index()
# %%
dia = pd.read_csv("diamonds.csv")
dia
# %%
pd.crosstab(dia["cut"], dia["clarity"])
# %%
pd.crosstab(dia["cut"], dia["clarity"], normalize=True)

# %%
pd.crosstab(dia["cut"], dia["clarity"], normalize=True).round(2)
# %%
pd.crosstab(dia["cut"], dia["clarity"], normalize=0)
pd.crosstab(dia["cut"], dia["clarity"], normalize=1)

# %%
pd.crosstab(dia["cut"], dia["clarity"], values=dia["price"], aggfunc=pd.Series.mean)
# %%
dia.groupby(["cut", "clarity"])["price"].mean().reset_index()
# %%
dia_agg = dia.groupby(["cut", "clarity"])["price"].mean().reset_index()
dia_agg.head()
# %%
dia_agg.sort_values("price")
# %%
dia_agg.sort_values("price", ascending=False)
# %%
dia_agg.sort_values(["cut", "clarity"], ascending=[True, False])
# %%
bike = pd.read_csv("bike.csv")
bike.head(2)
# %%
pd.crosstab(bike["workingday"], bike["holiday"])
# %%
pd.crosstab(bike["workingday"], bike["holiday"], normalize=True)
# %%
dia = pd.read_csv("diamonds.csv")
cross = pd.crosstab(dia["cut"], dia["color"])
cross
# %%
cross.reset_index().melt(id_vars="cut")
# %%
dia_agg = dia.groupby(["cut", "color"])[["price", "carat"]].mean().reset_index()
dia_agg.head(2)

# %%
dia_agg["ratio"] = dia_agg["price"] / dia_agg["carat"]
dia_agg = dia_agg.sort_values("ratio", ascending=False)
dia_agg.head()
# %%
# dia_agg["test"] = "yes" if (dia_agg["price"] >= 4000) else "no" # 이런건 에러가난다. if 문을 직접 써서 하면 안된다!!★
dia_agg["test"] = "yes"
dia_agg.loc[dia_agg["price"] < 4000, "test"] = "no"
dia_agg.head()
# %%
