# app/streamlit_app.py

from pathlib import Path
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="Airbnb Occupancy & Revenue Forecasting",
    page_icon="🏙️",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FORECAST_BASELINE_PATH = PROJECT_ROOT / "data" / "forecasts" / "neighborhood_forecast_90d.parquet"
FORECAST_SUPPLY_PATH = PROJECT_ROOT / "data" / "forecasts" / "neighborhood_forecast_90d_supply.parquet"

# -------------------------
# Cached loaders
# -------------------------
@st.cache_data
def load_forecast(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.rename(columns={"neighbourhood_cleansed": "neighborhood"})
    # numeric safety
    for c in ["supply_nights", "occupancy_rate", "booked_nights", "avg_price_est", "revenue_est", "lat", "lon", "demand_cluster_id"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def format_money(x: float) -> str:
    try:
        return f"${x:,.0f}"
    except Exception:
        return str(x)

# -------------------------
# Header
# -------------------------
st.title("🏙️ Airbnb Forecasting: Occupancy + Revenue (Neighborhood-level)")

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Controls")

source_choice = st.sidebar.radio(
    "Forecast source",
    options=["Baseline (constant supply)", "Supply-aware (predicted supply)"],
    index=1
)

forecast_path = FORECAST_SUPPLY_PATH if "Supply-aware" in source_choice else FORECAST_BASELINE_PATH

if not forecast_path.exists():
    st.error(
        f"Forecast file not found:\n{forecast_path}\n\n"
        f"Run the appropriate script:\n"
        f"- Baseline: python src/pipeline/forecast.py\n"
        f"- Supply-aware: python src/pipeline/forecast_supply.py"
    )
    st.stop()

df = load_forecast(forecast_path)

neighborhoods = sorted(df["neighborhood"].dropna().unique().tolist())
if not neighborhoods:
    st.error("No neighborhoods found in forecast data.")
    st.stop()

selected_nb = st.sidebar.selectbox("Select Neighborhood", neighborhoods, index=0)

st.sidebar.subheader("What-if scenario")
price_mult = st.sidebar.slider("Price multiplier", 0.8, 1.5, 1.0, 0.05)
demand_mult = st.sidebar.slider("Event demand multiplier (occupancy)", 1.0, 1.5, 1.0, 0.05)

st.sidebar.caption("Adjusted revenue formula:")
st.sidebar.code("revenue_adj = occ_adj × avg_price_adj × supply_nights", language="text")

show_raw = st.sidebar.checkbox("Show raw forecast table", value=False)

# -------------------------
# Derived what-if columns for selected neighborhood
# -------------------------
nb_df = df[df["neighborhood"] == selected_nb].copy().sort_values("date")
nb_df["occ_adj"] = np.clip(nb_df["occupancy_rate"] * demand_mult, 0.0, 1.0)
nb_df["avg_price_adj"] = nb_df["avg_price_est"] * price_mult
nb_df["revenue_adj"] = nb_df["occ_adj"] * nb_df["avg_price_adj"] * nb_df["supply_nights"]

# Baseline formula revenue (for consistent comparison)
nb_df["revenue_formula_base"] = nb_df["occupancy_rate"] * nb_df["avg_price_est"] * nb_df["supply_nights"]

# -------------------------
# Overview KPIs
# -------------------------
st.subheader("Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Neighborhoods", f"{df['neighborhood'].nunique()}")

date_min = df["date"].min().date()
date_max = df["date"].max().date()
col2.metric("Forecast start", f"{date_min}")
col3.metric("Forecast end", f"{date_max}")

first7 = nb_df.head(7)["revenue_formula_base"].mean()
last7 = nb_df.tail(7)["revenue_formula_base"].mean()
growth_pct = ((last7 - first7) / first7 * 100) if first7 and first7 > 0 else 0.0
col4.metric("90 day revenue growth (baseline)", f"{growth_pct:.1f}%")

st.divider()

# -------------------------
# Charts
# -------------------------
st.subheader(f"Forecast for: {selected_nb}")

left, right = st.columns(2)

occ_chart = alt.Chart(nb_df).mark_line().encode(
    x=alt.X("date:T", title="Date"),
    y=alt.Y("occupancy_rate:Q", title="Occupancy rate"),
    tooltip=["date:T", alt.Tooltip("occupancy_rate:Q", format=".3f")]
).properties(height=300)

occ_adj_chart = alt.Chart(nb_df).mark_line(strokeDash=[4, 4]).encode(
    x="date:T",
    y=alt.Y("occ_adj:Q", title="Occupancy rate"),
    tooltip=["date:T", alt.Tooltip("occ_adj:Q", format=".3f")]
)

left.altair_chart(occ_chart + occ_adj_chart, use_container_width=True)
left.caption("Solid = forecast occupancy | Dashed = what-if adjusted occupancy")

rev_base_chart = alt.Chart(nb_df).mark_line().encode(
    x=alt.X("date:T", title="Date"),
    y=alt.Y("revenue_formula_base:Q", title="Revenue ($)"),
    tooltip=["date:T", alt.Tooltip("revenue_formula_base:Q", format=",.0f")]
).properties(height=300)

rev_adj_chart = alt.Chart(nb_df).mark_line(strokeDash=[4, 4]).encode(
    x="date:T",
    y=alt.Y("revenue_adj:Q", title="Revenue ($)"),
    tooltip=["date:T", alt.Tooltip("revenue_adj:Q", format=",.0f")]
)

right.altair_chart(rev_base_chart + rev_adj_chart, use_container_width=True)
right.caption("Solid = baseline revenue formula | Dashed = what-if adjusted revenue")

st.divider()

# -------------------------
# What-if impact summary
# -------------------------
st.subheader("What-if Scenario Impact")

base_total = float(nb_df["revenue_formula_base"].sum())
adj_total = float(nb_df["revenue_adj"].sum())
delta = adj_total - base_total
delta_pct = (delta / base_total * 100) if base_total > 0 else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Baseline total revenue (formula)", format_money(base_total))
c2.metric("Adjusted total revenue", format_money(adj_total))
c3.metric("Change", f"{format_money(delta)} ({delta_pct:.1f}%)")
c4.metric("Avg price (baseline)", f"${nb_df['avg_price_est'].mean():.0f}")

# Export selected neighborhood to CSV
csv_df = nb_df[[
    "date", "neighborhood", "demand_cluster_id",
    "lat", "lon",
    "supply_nights",
    "occupancy_rate", "occ_adj",
    "avg_price_est", "avg_price_adj",
    "revenue_formula_base", "revenue_adj"
]].copy()

csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download selected neighborhood (CSV)",
    data=csv_bytes,
    file_name=f"{selected_nb.replace(' ', '_').lower()}_forecast.csv",
    mime="text/csv"
)

if show_raw:
    st.dataframe(csv_df, use_container_width=True)

st.divider()

# -------------------------
# Map view by demand clusters
# -------------------------
st.subheader("Map view: neighborhoods by demand cluster")

# One point per neighborhood (mean lat/lon)
geo = (
    df.groupby(["neighborhood"], as_index=False)
      .agg(
          lat=("lat", "mean"),
          lon=("lon", "mean"),
          demand_cluster_id=("demand_cluster_id", "median"),
          avg_revenue=("revenue_est", "mean"),
          avg_occ=("occupancy_rate", "mean"),
      )
)

# Streamlit map expects columns: lat, lon
# We'll show cluster via tooltip and size by avg revenue
map_chart = alt.Chart(geo).mark_circle().encode(
    longitude="lon:Q",
    latitude="lat:Q",
    size=alt.Size("avg_revenue:Q", title="Avg revenue", scale=alt.Scale(zero=False)),
    color=alt.Color("demand_cluster_id:Q", title="Demand cluster"),
    tooltip=[
        "neighborhood:N",
        alt.Tooltip("demand_cluster_id:Q", title="Cluster"),
        alt.Tooltip("avg_occ:Q", title="Avg occ", format=".3f"),
        alt.Tooltip("avg_revenue:Q", title="Avg revenue", format=",.0f"),
    ]
).properties(height=450)

st.altair_chart(map_chart, use_container_width=True)
st.caption("Circle size = avg forecasted revenue. Color = demand_cluster_id. Hover to see details.")

st.divider()

# -------------------------
# Leaderboard: Top 10 neighborhoods by predicted revenue growth
# -------------------------
st.subheader("Top 10 neighborhoods by predicted revenue growth")

tmp = df.copy()
tmp["revenue_formula_base"] = tmp["occupancy_rate"] * tmp["avg_price_est"] * tmp["supply_nights"]

def growth_for_group(g: pd.DataFrame) -> float:
    g = g.sort_values("date")
    first7 = g.head(7)["revenue_formula_base"].mean()
    last7 = g.tail(7)["revenue_formula_base"].mean()
    if first7 and first7 > 0:
        return float((last7 - first7) / first7)
    return 0.0

leader = (
    tmp.groupby("neighborhood", as_index=False)
       .apply(lambda g: pd.Series({
           "revenue_growth_pct": growth_for_group(g) * 100,
           "avg_revenue_first7": g.sort_values("date").head(7)["revenue_formula_base"].mean(),
           "avg_revenue_last7": g.sort_values("date").tail(7)["revenue_formula_base"].mean(),
       }))
       .reset_index(drop=True)
       .sort_values("revenue_growth_pct", ascending=False)
       .head(10)
)        

st.dataframe(leader, use_container_width=True)
st.caption("Growth compares average revenue of first 7 forecast days vs last 7 forecast days (baseline formula).")

st.success("Streamlit (final version). ")
