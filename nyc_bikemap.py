import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

districts = gpd.read_file("20221006-Final-Plan-Districts.json")
districts["DISTRICT"] = districts["DISTRICT"].astype(int)

lane_share = pd.read_csv("Citywide Protected Bike Lanes by City Council.csv")
lane_share["DISTRICT"] = (
    lane_share["City Council District"].str.split(" ").str[1].astype(int)
)

districts = districts.merge(lane_share)

boroughs = gpd.read_file("boroughs.json").make_valid()

legend_kwds = {
    "label": "Percent of street miles with a protected bike lane",
    "orientation": "vertical",
}

districts.clip(boroughs).plot(
    column="Percent of street miles with a protected bike lane",
    legend=True,
    legend_kwds=legend_kwds,
)

plt.show()

district_income = pd.read_csv("district_income.tsv", sep="\t")
district_income["DISTRICT"] = (
    district_income["District"].str.split(" ").str[1].astype(int)
)
district_income = district_income.drop("District", axis=1)
district_income["Median Income"] = (
    district_income["District Value"].str.replace(r"[$,]", "", regex=True).astype(int)
)


districts = districts.merge(district_income, how="left", on="DISTRICT")

z = np.polyfit(
    districts["Median Income"],
    districts["Percent of street miles with a protected bike lane"],
    1,
)
p = np.poly1d(z)

districts.plot.scatter(
    x="Median Income", y="Percent of street miles with a protected bike lane"
)
plt.plot(
    districts["Median Income"],
    p(districts["Median Income"]),
    color="red",
    label="Trend line",
)

plt.show()
