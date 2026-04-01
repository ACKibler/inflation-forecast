import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from fredapi import Fred
from dotenv import load_dotenv

load_dotenv()
fred = Fred(api_key=os.getenv("FRED_API_KEY"))

series = {
    "CPI":      "CPIAUCSL",
    "UNRATE":   "UNRATE",
    "M2":       "M2SL",
    "OIL":      "DCOILWTICO",
    "FEDFUNDS": "FEDFUNDS",
}

# Pull raw data WITHOUT dropna so gaps are visible
dfs = []
for name, sid in series.items():
    data = fred.get_series(sid, observation_start="2000-01-01")
    data.name = name
    dfs.append(data)

df_raw = pd.concat(dfs, axis=1).resample("MS").mean()

# Build a boolean missing matrix (True = missing)
missing = df_raw.isnull()
missing_rows = missing[missing.any(axis=1)]

fig, axes = plt.subplots(2, 1, figsize=(14, 8),
                         gridspec_kw={"height_ratios": [3, 1]})

# --- Top: heatmap of all data presence ---
ax = axes[0]
cmap = mcolors.ListedColormap(["#d73027", "#4dac26"])  # red=missing, green=present
ax.imshow(~missing.T.values, aspect="auto", cmap=cmap,
          interpolation="none", vmin=0, vmax=1)

ax.set_yticks(range(len(df_raw.columns)))
ax.set_yticklabels(df_raw.columns, fontsize=11)

# X-axis: show a tick every 3 years
years = df_raw.index[df_raw.index.month == 1]
tick_positions = [df_raw.index.get_loc(y) for y in years[::3]]
tick_labels = [y.year for y in years[::3]]
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=45, ha="right")

ax.set_title("Data Presence by Series (green = present, red = missing)", fontsize=13)
ax.set_xlabel("Date")

# Color bar legend
import matplotlib.patches as mpatches
present_patch = mpatches.Patch(color="#4dac26", label="Present")
missing_patch = mpatches.Patch(color="#d73027", label="Missing")
ax.legend(handles=[present_patch, missing_patch], loc="upper right", fontsize=10)

# --- Bottom: count of missing values per month ---
ax2 = axes[1]
missing_per_month = missing.sum(axis=1)
missing_per_month[missing_per_month > 0].plot(
    kind="bar", ax=ax2, color="#d73027", width=1.0
)
ax2.set_title("Number of Missing Series per Month", fontsize=12)
ax2.set_ylabel("# Missing")
ax2.set_xlabel("Date")
ax2.tick_params(axis="x", labelsize=7, rotation=45)

# Only label x-ticks where there are gaps
gap_dates = missing_per_month[missing_per_month > 0].index.strftime("%Y-%m")
ax2.set_xticklabels(gap_dates)

plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/missing_data.png", dpi=150, bbox_inches="tight")
print("Saved to outputs/missing_data.png")
plt.show()
