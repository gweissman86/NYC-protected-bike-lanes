import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats as stats

# ==========================================
# 1. Load and Clean Spatial & Bike Lane Data
# ==========================================

# Load City Council district boundaries and cast district IDs to integers
districts = gpd.read_file("20221006-Final-Plan-Districts.json")
districts["DISTRICT"] = districts["DISTRICT"].astype(int)

# Load protected bike lane share data and parse the district integer
lane_share = pd.read_csv("Citywide Protected Bike Lanes by City Council.csv")
lane_share["Percent of street miles with a protected bike lane"] = lane_share["Percent of street miles with a protected bike lane"] * 100
lane_share["DISTRICT"] = (
    lane_share["City Council District"].str.split(" ").str[1].astype(int)
)

# Merge bike lane metrics into the spatial districts dataframe
districts = districts.merge(lane_share)

# Load borough boundaries and repair any invalid geometries
boroughs = gpd.read_file("boroughs.json").make_valid()


# ==========================================
# 2. Map Visualisation: Bike Lane Density
# ==========================================

# Set up a clean, high-DPI figure for spatial plotting
fig, ax = plt.subplots(figsize=(10, 10))

# Clip districts to borough boundaries and plot using a stylish continuous colormap
districts.clip(boroughs).plot(
    column="Percent of street miles with a protected bike lane",
    cmap="YlGnBu",
    linewidth=0.5,
    edgecolor="0.5",
    legend=True,
    legend_kwds={
        "label": "Percent of Street Miles with Protected Bike Lanes (%)",
        "orientation": "vertical",
        "shrink": 0.7,
    },
    ax=ax,
)

# Styling details for map
ax.set_title(
    "Protected Bike Lane Density by NYC Council District",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
ax.axis("off")  # Remove axis ticks and frame for a cleaner map
plt.tight_layout()
plt.savefig("bike_lane_map.png")


# ==========================================
# 3. Load and Merge Income Data
# ==========================================

# Read median income data and clean currency formatting
district_income = pd.read_csv("district_income.tsv", sep="\t")
district_income["DISTRICT"] = (
    district_income["District"].str.split(" ").str[1].astype(int)
)
district_income = district_income.drop("District", axis=1)
district_income["Median Income"] = (
    district_income["District Value"].str.replace(r"[$,]", "", regex=True).astype(int)
)

# Merge income data with existing dataset
districts = districts.merge(district_income, how="left", on="DISTRICT")


# ==========================================
# 4. Calculate & Print Relationship Statistics
# ==========================================

# Clean subset containing complete cases for statistical analysis
valid_data = districts.dropna(
    subset=["Median Income", "Percent of street miles with a protected bike lane"]
)

x = valid_data["Median Income"]
y = valid_data["Percent of street miles with a protected bike lane"]

# Linear fit coefficients
slope, intercept = np.polyfit(x, y, 1)

# Correlation coefficients
pearson_r, pearson_p = stats.pearsonr(x, y)
spearman_r, spearman_p = stats.spearmanr(x, y)

print("=" * 65)
print("STATISTICAL SUMMARY: Median Income vs. Protected Bike Lanes")
print("=" * 65)
print(f"Sample Size (Districts): {len(valid_data)}")
print(f"Mean Median Income:       ${x.mean():,.2f} (Std: ${x.std():,.2f})")
print(f"Mean Protected Lane %:    {y.mean():.2f}% (Std: {y.std():.2f}%)")
print("-" * 65)
print(f"Pearson Correlation (r):  {pearson_r:.4f} (p-value: {pearson_p:.4f})")
print(f"Spearman Rank Corr (r):  {spearman_r:.4f} (p-value: {spearman_p:.4f})")
print(f"Linear Trend Line Equation: y = {slope:.6f} * x + {intercept:.4f}")
print("=" * 65)


# ==========================================
# 5. Chart Visualisation: Income vs. Lanes
# ==========================================

fig, ax = plt.subplots(figsize=(9, 6))

# Scatter plot of district points
ax.scatter(
    x,
    y,
    color="#2b5c8f",
    alpha=0.75,
    edgecolors="w",
    s=70,
    zorder=3,
    label="Council District",
)

# Trend line evaluation
x_vals = np.linspace(x.min(), x.max(), 100)
y_vals = slope * x_vals + intercept

# Plot trend line
ax.plot(
    x_vals,
    y_vals,
    color="#d95f02",
    linewidth=2,
    linestyle="--",
    zorder=4,
    label=f"Trend Line (r = {pearson_r:.2f})",
)

# Polish formatting
ax.set_title(
    "NYC Council Districts: Median Income vs. Protected Bike Lane Share",
    fontsize=13,
    fontweight="bold",
    pad=12,
)
ax.set_xlabel("Median Income ($)", fontsize=11, labelpad=8)
ax.set_ylabel(
    "Protected Bike Lanes (% of Street Miles)", fontsize=11, labelpad=8
)

# Format X axis as currency values
ax.xaxis.set_major_formatter("${x:,.0f}")

# Layout tweaks
ax.grid(True, linestyle=":", alpha=0.6, zorder=1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=True, facecolor="white", edgecolor="none")

plt.tight_layout()
plt.savefig("bike_lane_chart.png")