"""
Raw Data Dashboard — Streamlit
Visualizes each raw dataset separately on one page using tabs.

Run with:
    streamlit run src/analysis/raw_data_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"

# ── helpers ────────────────────────────────────────────────────────────────
MONTH_MAP = {
    "Januar": 1, "Februar": 2, "März": 3, "April": 4, "Mai": 5, "Juni": 6,
    "Juli": 7, "August": 8, "September": 9, "Oktober": 10, "November": 11, "Dezember": 12,
}

def fix_enc(v):
    if isinstance(v, str):
        return (v.replace("Ã¤", "ä").replace("Ã¼", "ü")
                 .replace("Ã¶", "ö").replace("ÃŸ", "ß"))
    return v

def parse_destatis_month(df, month_col, year_col="time"):
    """Add a 'date' column from a German month label + year column."""
    df = df.copy()
    df[month_col] = df[month_col].apply(fix_enc)
    df["_month_num"] = df[month_col].map(MONTH_MAP)
    df = df.dropna(subset=["_month_num"])
    df["date"] = pd.to_datetime(
        df[year_col].astype(str) + "-" + df["_month_num"].astype(int).astype(str) + "-01"
    )
    return df.drop(columns=["_month_num"])

def parse_semester_date(s):
    """2022-10P6M → 2022-10-01, 2022-04P6M → 2022-04-01"""
    if not isinstance(s, str):
        return None
    try:
        year = s[:4]
        month = "10" if "-10" in s else "04"
        return pd.to_datetime(f"{year}-{month}-01")
    except Exception:
        return None

def read_destatis(fname, **kwargs):
    return pd.read_csv(DATA_RAW / fname, sep=";", decimal=",",
                       encoding="utf-8", low_memory=False, **kwargs)

def num_col(df, col="value"):
    df = df.copy()
    if df[col].dtype == object:
        df[col] = pd.to_numeric(df[col].str.replace(",", "."), errors="coerce")
    else:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def stat_row(series, label):
    """Return a dict of key statistics for st.metric display."""
    s = series.dropna()
    if s.empty:
        return {}
    return {
        "label": label,
        "min": f"{s.min():.2f}",
        "max": f"{s.max():.2f}",
        "mean": f"{s.mean():.2f}",
        "latest": f"{s.iloc[-1]:.2f}",
    }

def metrics_row(stats_list):
    cols = st.columns(len(stats_list) * 4)
    ci = 0
    for s in stats_list:
        cols[ci].metric("Min", s["min"])
        cols[ci+1].metric("Max", s["max"])
        cols[ci+2].metric("Mean", s["mean"])
        cols[ci+3].metric("Latest", s["latest"])
        ci += 4

# ── loaders ────────────────────────────────────────────────────────────────

@st.cache_data
def load_wages():
    df = read_destatis("data_wages.csv")
    df = num_col(df)
    df = parse_destatis_month(df, "1_variable_attribute_label")
    df["value_variable_label"] = df["value_variable_label"].apply(fix_enc)
    nominal = (df[df["value_variable_label"] == "Nominallohnindex"]
               [["date", "value"]].rename(columns={"value": "Nominallohnindex"})
               .sort_values("date").drop_duplicates("date"))
    real = (df[df["value_variable_label"] == "Reallohnindex"]
            [["date", "value"]].rename(columns={"value": "Reallohnindex"})
            .sort_values("date").drop_duplicates("date"))
    out = pd.merge(nominal, real, on="date", how="outer").sort_values("date")
    return out

@st.cache_data
def load_rent():
    """Price index for rent from data_price_index.csv (61111 CPI 3-digit)."""
    df = read_destatis("data_price_index.csv")
    df = num_col(df)
    df = parse_destatis_month(df, "1_variable_attribute_label")
    df["3_variable_attribute_label"] = df["3_variable_attribute_label"].apply(fix_enc)
    # Tatsächliche Nettokaltmiete = actual net cold rent
    rent = df[df["3_variable_attribute_label"].str.contains(
        "Nettokaltmiete|Miete", na=False, case=False
    )][["date", "3_variable_attribute_label", "value"]]
    # Keep the most granular "Tatsächliche Nettokaltmiete" if available
    pref = rent[rent["3_variable_attribute_label"].str.contains("Tatsächliche", na=False)]
    if pref.empty:
        pref = rent
    out = (pref.groupby("date")["value"].mean().reset_index()
           .rename(columns={"value": "rent_index"}).sort_values("date"))
    return out

@st.cache_data
def load_students():
    df = read_destatis("data_students.csv")
    df = num_col(df)
    # Filter Insgesamt across all attribute columns
    attr_cols = [c for c in df.columns if "_variable_attribute_label" in c]
    for col in attr_cols:
        if df[col].str.contains("Insgesamt", na=False, case=False).any():
            df = df[df[col].str.contains("Insgesamt", na=False, case=False) | df[col].isna()]
    df["date"] = df["time"].apply(parse_semester_date)
    df = df.dropna(subset=["date", "value"])
    out = (df.groupby("date")["value"].sum().reset_index()
           .rename(columns={"value": "elite_students"}).sort_values("date"))
    return out

@st.cache_data
def load_gdp():
    df = read_destatis("data_gdp_quarterly.csv")
    df = num_col(df)
    # GDP volume index (chain index 2020=100)
    vol = df[df["value_variable_code"] == "VGR014"][["time", "value"]].copy()
    vol["date"] = pd.to_datetime(vol["time"].astype(str) + "-01-01")
    # GDP growth %
    chg = df[df["value_variable_code"] == "BIP005"][["time", "value"]].copy()
    chg["date"] = pd.to_datetime(chg["time"].astype(str) + "-01-01")
    vol = vol.groupby("date")["value"].mean().reset_index().rename(columns={"value": "gdp_index"})
    chg = chg.groupby("date")["value"].mean().reset_index().rename(columns={"value": "gdp_growth_pct"})
    out = pd.merge(vol, chg, on="date", how="outer").sort_values("date")
    return out

@st.cache_data
def load_cpi_general():
    df = read_destatis("data_cpi_general.csv")
    df = num_col(df)
    df = parse_destatis_month(df, "1_variable_attribute_label")
    # Overall CPI: use 2-digit level total basket (CC13-00 or first 2-digit entry)
    # data_cpi_general uses CC13A2 (2-Steller) — total = 00 or aggregation
    if "3_variable_attribute_code" in df.columns:
        # Try to find CC13-00 (Insgesamt) or the widest category
        total_mask = df["3_variable_attribute_code"].str.match(r"^CC13-\d{2}$", na=False)
        if total_mask.any():
            df = df[total_mask]
    # Deduplicate: average per date
    out = (df.groupby("date")["value"].mean().reset_index()
           .rename(columns={"value": "cpi_general"}).sort_values("date"))
    return out

@st.cache_data
def load_food_prices():
    df = read_destatis("data_food_prices.csv")
    df = num_col(df)
    df = parse_destatis_month(df, "1_variable_attribute_label")
    # Food = CC13-01 (Nahrungsmittel und alkoholfreie Getränke)
    if "3_variable_attribute_code" in df.columns:
        food = df[df["3_variable_attribute_code"] == "CC13-01"]
        if food.empty:
            food = df
    else:
        food = df
    out = (food.groupby("date")["value"].mean().reset_index()
           .rename(columns={"value": "food_price_index"}).sort_values("date"))
    return out

@st.cache_data
def load_youth():
    df = read_destatis("data_youth_annual.csv")
    df = num_col(df)
    # Ages 15–24: attribute codes ALT015 … ALT024
    if "2_variable_attribute_code" in df.columns:
        mask = df["2_variable_attribute_code"].str.contains(
            r"ALT01[5-9]|ALT02[0-4]", na=False, regex=True
        )
        df = df[mask]
    # Dates: time column is like "2024-12-31"
    df["date"] = pd.to_datetime(df["time"].astype(str).str[:4] + "-01-01", errors="coerce")
    df = df.dropna(subset=["date", "value"])
    out = (df.groupby("date")["value"].sum().reset_index()
           .rename(columns={"value": "youth_15_24"}).sort_values("date"))
    return out

@st.cache_data
def load_tax():
    df = read_destatis("data_tax_revenue.csv")
    df = num_col(df)
    # Group all entries by year, then sum to get total state revenue proxy
    df["date"] = pd.to_datetime(df["time"].astype(str) + "-01-01", errors="coerce")
    df = df.dropna(subset=["date", "value"])
    # Total: sum of all tax categories per year (crude but shows trend)
    out = (df.groupby("date")["value"].sum().reset_index()
           .rename(columns={"value": "total_tax_revenue_k_eur"}).sort_values("date"))
    return out

@st.cache_data
def load_civil_servants():
    df = read_destatis("data_civil_servants.csv")
    df = num_col(df)
    # Total headcount: filter for "Insgesamt"
    if "2_variable_attribute_label" in df.columns:
        total = df[df["2_variable_attribute_label"].str.contains("Insgesamt", na=False)]
        if total.empty:
            total = df
    else:
        total = df
    # Dates: time = "2022-06-30"
    total = total.copy()
    total["date"] = pd.to_datetime(total["time"].astype(str).str[:4] + "-06-01", errors="coerce")
    total = total.dropna(subset=["date", "value"])
    out = (total.groupby("date")["value"].mean().reset_index()
           .rename(columns={"value": "civil_servants_k"}).sort_values("date"))
    return out

@st.cache_data
def load_strikes():
    df = pd.read_csv(DATA_RAW / "data_strikes_wsi.csv")
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-01-01")
    return df[["date", "strike_days_k"]].sort_values("date")

@st.cache_data
def load_holders():
    df = pd.read_csv(DATA_RAW / "data_holders.csv")
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-01-01")
    return df[["date", "holders_k"]].sort_values("date")

@st.cache_data
def load_google_trends():
    df = pd.read_csv(DATA_RAW / "google_trends_mobilization.csv", parse_dates=["date"])
    return df.sort_values("date")

# ── chart helpers ──────────────────────────────────────────────────────────

COLORS = px.colors.qualitative.Plotly

def line_chart(df, x, ys, title, yaxis_title="", colors=None):
    fig = go.Figure()
    for i, y in enumerate(ys):
        if y not in df.columns:
            continue
        color = (colors[i] if colors else COLORS[i % len(COLORS)])
        fig.add_trace(go.Scatter(
            x=df[x], y=df[y], mode="lines", name=y,
            line=dict(color=color, width=2)
        ))
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title=yaxis_title,
        hovermode="x unified", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40, l=60, r=20),
    )
    return fig

def bar_chart(df, x, y, title, yaxis_title="", color=None):
    fig = go.Figure(go.Bar(
        x=df[x], y=df[y], name=y,
        marker_color=color or COLORS[0]
    ))
    fig.update_layout(
        title=title, xaxis_title="Date", yaxis_title=yaxis_title,
        height=420, margin=dict(t=60, b=40, l=60, r=20),
    )
    return fig

# ── page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Cliodynamics — Raw Data Explorer",
    page_icon="📊",
    layout="wide",
)

st.title("Cliodynamics — Raw Dataset Explorer")
st.caption(
    "Interactive visualization of each raw data source used in the PSI v4 pipeline."
)

# ── tabs ───────────────────────────────────────────────────────────────────

TABS = [
    "Wages",
    "Rent Index",
    "Students",
    "GDP",
    "CPI General",
    "Food Prices",
    "Youth Population",
    "Tax Revenue",
    "Civil Servants",
    "Strikes",
    "Holders",
    "Google Trends",
]

tabs = st.tabs(TABS)

# ── 1. Wages ───────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Wage Indices — Destatis 62361-0001")
    st.markdown(
        "**Source:** `data/raw/data_wages.csv` · Monthly nominal and real wage index (2022=100). "
        "Used to compute `wealth_pump = rent / wage_nominal`."
    )
    df = load_wages()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(df))
    c2.metric("Period start", str(df["date"].min().date()))
    c3.metric("Period end", str(df["date"].max().date()))
    c4.metric("Latest nominal", f"{df['Nominallohnindex'].dropna().iloc[-1]:.1f}")

    ma_window = st.slider("Moving average window (months)", 2, 24, 12, key="wages_ma")
    fig = go.Figure()
    for col, color in [("Nominallohnindex", COLORS[0]), ("Reallohnindex", COLORS[1])]:
        if col not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df["date"], y=df[col], mode="lines", name=col,
            line=dict(color=color, width=1.5), opacity=0.45
        ))
        ma = df[col].rolling(ma_window, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=df["date"], y=ma, mode="lines", name=f"{col} {ma_window}m MA",
            line=dict(color=color, width=2.5)
        ))
    fig.update_layout(
        title="Nominal vs Real Wage Index (2022=100)",
        xaxis_title="Date", yaxis_title="Index",
        hovermode="x unified", height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40, l=60, r=20),
    )
    st.plotly_chart(fig, width='stretch')

    with st.expander("Raw data"):
        st.dataframe(df, width='stretch')

# ── 2. Rent / Price Index ──────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Rent Price Index — Destatis 61111-0004")
    st.markdown(
        "**Source:** `data/raw/data_price_index.csv` · Monthly CPI for rent category "
        "(*Tatsächliche Nettokaltmiete*, base 2020=100). "
        "Forms the numerator of `wealth_pump`."
    )
    df = load_rent()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(df))
    c2.metric("Period start", str(df["date"].min().date()))
    c3.metric("Period end", str(df["date"].max().date()))
    c4.metric("Latest", f"{df['rent_index'].dropna().iloc[-1]:.1f}")

    fig = line_chart(df, "date", ["rent_index"],
                     "Rent Price Index (2020=100)", "Index", colors=["#EF553B"])
    st.plotly_chart(fig, width='stretch')

    with st.expander("Raw data"):
        st.dataframe(df, width='stretch')

# ── 3. Students ────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("University Students (Elite Fields) — Destatis 21311-0003")
    st.markdown(
        "**Source:** `data/raw/data_students.csv` · Semi-annual (winter/summer semester). "
        "Aggregated total of students in status-seeking fields. "
        "Used as `elite_candidates` in `elite_pressure`."
    )
    df = load_students()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Semesters", len(df))
    c2.metric("Period start", str(df["date"].min().date()))
    c3.metric("Period end", str(df["date"].max().date()))
    c4.metric("Latest count", f"{df['elite_students'].iloc[-1]:,.0f}")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["date"], y=df["elite_students"],
        marker_color="#00CC96", name="Elite Students"
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["elite_students"],
        mode="lines+markers", line=dict(color="#636EFA", width=2),
        name="Trend"
    ))
    fig.update_layout(
        title="Elite Candidates per Semester",
        xaxis_title="Semester", yaxis_title="Students",
        height=420, hovermode="x unified",
        margin=dict(t=60, b=40, l=60, r=20),
    )
    st.plotly_chart(fig, width='stretch')

    with st.expander("Raw data"):
        st.dataframe(df, width='stretch')

# ── 4. GDP ─────────────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("GDP — Destatis 81111-0001")
    st.markdown(
        "**Source:** `data/raw/data_gdp_quarterly.csv` · Annual GDP chain index (2020=100) "
        "and YoY growth %. Used to derive `m_econ = gdp_growth − real_wage_growth`."
    )
    df = load_gdp()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Years", len(df))
    c2.metric("Period start", str(df["date"].min().date()))
    c3.metric("Period end", str(df["date"].max().date()))
    c4.metric("Latest growth %", f"{df['gdp_growth_pct'].dropna().iloc[-1]:.1f}%")

    fig = line_chart(df.dropna(subset=["gdp_index"]), "date", ["gdp_index"],
                     "GDP Volume Index (2020=100)", "Index (2020=100)",
                     colors=["#FFA15A"])
    st.plotly_chart(fig, width='stretch')

    yoy_gdp = df.set_index("date")["gdp_index"].pct_change() * 100
    fig2 = go.Figure(go.Bar(
        x=yoy_gdp.index, y=yoy_gdp.values,
        marker_color=["#EF553B" if v < 0 else "#00CC96" for v in yoy_gdp.fillna(0)],
        name="YoY Change %"
    ))
    fig2.update_layout(title="GDP Volume Index YoY Change (%)", xaxis_title="Year",
                       yaxis_title="%", height=320,
                       margin=dict(t=60, b=40, l=60, r=20))
    st.plotly_chart(fig2, width='stretch')

    with st.expander("Raw data"):
        st.dataframe(df, width='stretch')

# ── 5. CPI General ─────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Consumer Price Index (General) — Destatis 61111-0004")
    st.markdown(
        "**Source:** `data/raw/data_cpi_general.csv` · Monthly CPI by 2-digit category. "
        "Used as the reference index to compute `food_pump = food_inflation − CPI`."
    )
    df = load_cpi_general()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(df))
    c2.metric("Period start", str(df["date"].min().date()))
    c3.metric("Period end", str(df["date"].max().date()))
    c4.metric("Latest", f"{df['cpi_general'].dropna().iloc[-1]:.1f}")

    fig = line_chart(df, "date", ["cpi_general"],
                     "Consumer Price Index — Average across 2-digit categories",
                     "Index (2020=100)", colors=["#19D3F3"])
    st.plotly_chart(fig, width='stretch')

    with st.expander("Raw data"):
        st.dataframe(df, width='stretch')

# ── 6. Food Prices ─────────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("Food Price Index — Destatis 61111-0004 (CC13-01)")
    st.markdown(
        "**Source:** `data/raw/data_food_prices.csv` · Monthly CPI for food & "
        "non-alcoholic beverages (*Nahrungsmittel und alkoholfreie Getränke*, 2020=100). "
        "Used in `food_pump`."
    )
    df = load_food_prices()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(df))
    c2.metric("Period start", str(df["date"].min().date()))
    c3.metric("Period end", str(df["date"].max().date()))
    c4.metric("Latest", f"{df['food_price_index'].dropna().iloc[-1]:.1f}")

    fig = line_chart(df, "date", ["food_price_index"],
                     "Food Price Index (CC13-01, 2020=100)",
                     "Index (2020=100)", colors=["#FF6692"])
    st.plotly_chart(fig, width='stretch')

    with st.expander("Raw data"):
        st.dataframe(df, width='stretch')

# ── 7. Youth Population ────────────────────────────────────────────────────
with tabs[6]:
    st.subheader("Youth Population (ages 15–24) — Destatis 12411-0005")
    st.markdown(
        "**Source:** `data/raw/data_youth_annual.csv` · Annual population count for ages 15–24. "
        "Normalized to mean → `youth_bulge` PSI component."
    )
    df = load_youth()
    mean_val = df["youth_15_24"].mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Years", len(df))
    c2.metric("Period start", str(df["date"].min().date()))
    c3.metric("Period end", str(df["date"].max().date()))
    c4.metric("Mean youth pop", f"{mean_val:,.0f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["youth_15_24"],
        mode="lines+markers", name="Youth 15-24",
        line=dict(color="#FECB52", width=2), fill="tozeroy",
        fillcolor="rgba(254,203,82,0.15)"
    ))
    fig.add_hline(y=mean_val, line_dash="dash", line_color="gray",
                  annotation_text=f"Mean: {mean_val:,.0f}")
    fig.update_layout(
        title="Youth Population (15–24 years)", xaxis_title="Year",
        yaxis_title="Population", height=420,
        margin=dict(t=60, b=40, l=60, r=20),
    )
    st.plotly_chart(fig, width='stretch')

    with st.expander("Raw data"):
        st.dataframe(df, width='stretch')

# ── 8. Tax Revenue ─────────────────────────────────────────────────────────
with tabs[7]:
    st.subheader("Tax Revenue — Destatis 71211-0001")
    st.markdown(
        "**Source:** `data/raw/data_tax_revenue.csv` · Annual tax revenues by category (Tsd. EUR). "
        "Sum across all categories shown as total proxy. Used for `s_capacity` (tax stability)."
    )
    df = load_tax()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Years", len(df))
    c2.metric("Period start", str(df["date"].min().date()))
    c3.metric("Period end", str(df["date"].max().date()))
    c4.metric("Latest total (Tsd. EUR)", f"{df['total_tax_revenue_k_eur'].iloc[-1]:,.0f}")

    fig = bar_chart(df, "date", "total_tax_revenue_k_eur",
                    "Total Tax Revenue Sum (all categories, Tsd. EUR)",
                    "Tsd. EUR", color="#636EFA")
    st.plotly_chart(fig, width='stretch')

    yoy = df.set_index("date")["total_tax_revenue_k_eur"].pct_change() * 100
    fig2 = go.Figure(go.Bar(
        x=yoy.index, y=yoy.values,
        marker_color=["#EF553B" if v < 0 else "#00CC96" for v in yoy.fillna(0)],
        name="YoY Change %"
    ))
    fig2.update_layout(title="Tax Revenue YoY Change (%)", xaxis_title="Year",
                       yaxis_title="%", height=320,
                       margin=dict(t=60, b=40, l=60, r=20))
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Raw data"):
        st.dataframe(df, width='stretch')

# ── 9. Civil Servants ──────────────────────────────────────────────────────
with tabs[8]:
    st.subheader("Civil Servants (Public Sector) — Destatis 74111-0001")
    st.markdown(
        "**Source:** `data/raw/data_civil_servants.csv` · Annual headcount (June snapshot, ×1000). "
        "Total public sector employees used in `s_capacity` composite."
    )
    df = load_civil_servants()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Years", len(df))
    c2.metric("Period start", str(df["date"].min().date()))
    c3.metric("Period end", str(df["date"].max().date()))
    c4.metric("Latest (×1000)", f"{df['civil_servants_k'].dropna().iloc[-1]:,.1f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["civil_servants_k"],
        mode="lines+markers", name="Civil Servants",
        line=dict(color="#B6E880", width=2)
    ))
    fig.update_layout(
        title="Total Public Sector Employees (×1000)",
        xaxis_title="Year", yaxis_title="Headcount (×1000)",
        height=420, margin=dict(t=60, b=40, l=60, r=20),
    )
    st.plotly_chart(fig, width='stretch')

    with st.expander("Raw data"):
        st.dataframe(df, width='stretch')

# ── 10. Strikes ────────────────────────────────────────────────────────────
with tabs[9]:
    st.subheader("Strike Days — WSI Arbeitskampfbilanz")
    st.markdown(
        "**Source:** `data/raw/data_strikes_wsi.csv` · Annual days lost to strikes (×1000). "
        "Interpolated monthly as `strike_days` PSI component."
    )
    df = load_strikes()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Years", len(df))
    c2.metric("Period start", str(df["date"].min().date()))
    c3.metric("Period end", str(df["date"].max().date()))
    c4.metric("Latest (k days)", f"{df['strike_days_k'].iloc[-1]:,}")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["date"], y=df["strike_days_k"],
        marker_color="#FF97FF", name="Strike Days (k)"
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["strike_days_k"].rolling(3, min_periods=1).mean(),
        mode="lines", name="3-yr MA",
        line=dict(color="#636EFA", width=2, dash="dot")
    ))
    fig.update_layout(
        title="Annual Days Lost to Strikes (×1000)",
        xaxis_title="Year", yaxis_title="Strike Days (×1000)",
        height=420, hovermode="x unified",
        margin=dict(t=60, b=40, l=60, r=20),
    )
    st.plotly_chart(fig, width='stretch')

    with st.expander("Raw data"):
        st.dataframe(df, width='stretch')

# ── 11. Holders ────────────────────────────────────────────────────────────
with tabs[10]:
    st.subheader("Elite Position Holders — Mikrozensus Führungskräfte")
    st.markdown(
        "**Source:** `data/raw/data_holders.csv` · Annual employed managers/executives "
        "(Berufshauptgruppe 1, ×1000). Used in `frustrated_fraction` calculation."
    )
    df = load_holders()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Years", len(df))
    c2.metric("Period start", str(df["date"].min().date()))
    c3.metric("Period end", str(df["date"].max().date()))
    c4.metric("Latest (k)", f"{df['holders_k'].iloc[-1]:,}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["holders_k"],
        mode="lines+markers", name="Holders",
        line=dict(color="#00CC96", width=2), fill="tozeroy",
        fillcolor="rgba(0,204,150,0.12)"
    ))
    fig.update_layout(
        title="Employed Managers / Executives (×1000)",
        xaxis_title="Year", yaxis_title="Holders (×1000)",
        height=420, margin=dict(t=60, b=40, l=60, r=20),
    )
    st.plotly_chart(fig, width='stretch')

    with st.expander("Raw data"):
        st.dataframe(df, width='stretch')

# ── 12. Google Trends ──────────────────────────────────────────────────────
with tabs[11]:
    st.subheader("Google Trends — Mobilization Index")
    st.markdown(
        "**Source:** `data/raw/google_trends_mobilization.csv` · Monthly relative search volume "
        "for protest/mobilization keywords in Germany. Composite `mobilization_index` = average."
    )
    df = load_google_trends()
    keywords = [c for c in df.columns if c not in ("date",)]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(df))
    c2.metric("Period start", str(df["date"].min().date()))
    c3.metric("Period end", str(df["date"].max().date()))
    if "mobilization_index" in df.columns:
        c4.metric("Latest index", f"{df['mobilization_index'].iloc[-1]:.1f}")

    # Individual keyword trends
    kw_cols = [c for c in keywords if c != "mobilization_index"]
    fig = go.Figure()
    for i, kw in enumerate(kw_cols):
        fig.add_trace(go.Scatter(
            x=df["date"], y=df[kw],
            mode="lines", name=kw,
            line=dict(color=COLORS[i % len(COLORS)], width=1.5),
            opacity=0.7,
        ))
    fig.update_layout(
        title="Individual Search Terms (Relative Search Volume, 0–100)",
        xaxis_title="Date", yaxis_title="RSV",
        height=360, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40, l=60, r=20),
    )
    st.plotly_chart(fig, width='stretch')

    if "mobilization_index" in df.columns:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["date"], y=df["mobilization_index"],
            mode="lines", name="Mobilization Index",
            line=dict(color="#EF553B", width=2), fill="tozeroy",
            fillcolor="rgba(239,85,59,0.12)"
        ))
        fig2.add_trace(go.Scatter(
            x=df["date"],
            y=df["mobilization_index"].rolling(12, min_periods=1).mean(),
            mode="lines", name="12-month MA",
            line=dict(color="#636EFA", width=2, dash="dash")
        ))
        fig2.update_layout(
            title="Composite Mobilization Index + 12-Month Moving Average",
            xaxis_title="Date", yaxis_title="Index",
            height=360, hovermode="x unified",
            margin=dict(t=60, b=40, l=60, r=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Raw data"):
        st.dataframe(df, width='stretch')
