"""
Cliodynamics Dashboard — Streamlit
Two views: Raw Data (one tab per source) and Processed Data (PSI components).

Run with:
    streamlit run src/analysis/raw_data_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

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
    df = df.copy()
    df[month_col] = df[month_col].apply(fix_enc)
    df["_month_num"] = df[month_col].map(MONTH_MAP)
    df = df.dropna(subset=["_month_num"])
    df["date"] = pd.to_datetime(
        df[year_col].astype(str) + "-" + df["_month_num"].astype(int).astype(str) + "-01"
    )
    return df.drop(columns=["_month_num"])

def parse_semester_date(s):
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
    df = read_destatis("data_rent_index.csv")
    df = num_col(df)
    df = parse_destatis_month(df, "1_variable_attribute_label")
    df["3_variable_attribute_label"] = df["3_variable_attribute_label"].apply(fix_enc)
    rent = df[df["3_variable_attribute_label"].str.contains(
        "Nettokaltmiete|Miete", na=False, case=False
    )][["date", "3_variable_attribute_label", "value"]]
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
    attr_cols = [c for c in df.columns if "_variable_attribute_label" in c]
    for col in attr_cols:
        if df[col].str.contains("Insgesamt", na=False, case=False).any():
            df = df[df[col].str.contains("Insgesamt", na=False, case=False) | df[col].isna()]
    df["date"] = df["time"].apply(parse_semester_date)
    df = df.dropna(subset=["date", "value"])
    out = (df.groupby("date")["value"].sum().reset_index()
           .rename(columns={"value": "elite_candidates"}).sort_values("date"))
    return out

@st.cache_data
def load_gdp():
    df = read_destatis("data_gdp_quarterly.csv")
    df = num_col(df)
    # Ensure consistent unit: VGRPVK (price-adjusted chained volume)
    mask_bip = (df["value_variable_code"] == "VGR014") & (df["2_variable_attribute_code"] == "VGRPVK")
    vol = df[mask_bip][["time", "value"]].copy()
    vol["date"] = pd.to_datetime(vol["time"].astype(str) + "-01-01")
    
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
    if "3_variable_attribute_code" in df.columns:
        total_mask = df["3_variable_attribute_code"].str.match(r"^CC13-\d{2}$", na=False)
        if total_mask.any():
            df = df[total_mask]
    out = (df.groupby("date")["value"].mean().reset_index()
           .rename(columns={"value": "cpi_general"}).sort_values("date"))
    return out

@st.cache_data
def load_food_prices():
    df = read_destatis("data_food_prices.csv")
    df = num_col(df)
    df = parse_destatis_month(df, "1_variable_attribute_label")
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
    
    # 1. Total Population
    total = df[df["2_variable_attribute_label"] == "Insgesamt"].groupby("time")["value"].sum()
    
    # 2. Youth Cohort (15-24)
    if "2_variable_attribute_code" in df.columns:
        mask = df["2_variable_attribute_code"].str.contains(
            r"ALT01[5-9]|ALT02[0-4]", na=False, regex=True
        )
        youth = df[mask].groupby("time")["value"].sum()
    else:
        youth = total * 0.1 # Fallback
        
    share = (youth / total).reset_index().rename(columns={"value": "youth_share"})
    share["date"] = pd.to_datetime(share["time"].astype(str).str[:4] + "-01-01", errors="coerce")
    share["youth_count_k"] = youth.values / 1000
    
    return share.dropna(subset=["date"]).sort_values("date")

@st.cache_data
def load_tax():
    df = read_destatis("data_tax_revenue.csv")
    df = num_col(df)
    df["date"] = pd.to_datetime(df["time"].astype(str) + "-01-01", errors="coerce")
    df = df.dropna(subset=["date", "value"])
    out = (df.groupby("date")["value"].sum().reset_index()
           .rename(columns={"value": "total_tax_revenue_k_eur"}).sort_values("date"))
    return out

@st.cache_data
def load_tax_by_type():
    df = read_destatis("data_tax_revenue.csv")
    df = num_col(df)
    df["date"] = pd.to_datetime(df["time"].astype(str) + "-01-01", errors="coerce")
    df = df.dropna(subset=["date", "value"])
    tax_col = "2_variable_attribute_label"
    out = (df.groupby(["date", tax_col])["value"].sum().reset_index()
           .rename(columns={tax_col: "tax_type", "value": "revenue_k_eur"})
           .sort_values(["tax_type", "date"]))
    return out

@st.cache_data
def load_civil_servants():
    df = read_destatis("data_civil_servants.csv")
    df = num_col(df)
    if "2_variable_attribute_label" in df.columns:
        total = df[df["2_variable_attribute_label"].str.contains("Insgesamt", na=False)]
        if total.empty:
            total = df
    else:
        total = df
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

@st.cache_data
def load_studentflow():
    path = DATA_RAW / "data_studienanfaenger.csv"
    if not path.exists():
        return None
    df = read_destatis("data_studienanfaenger.csv")
    df = num_col(df)
    def parse_sf_time(t):
        s = str(t)
        if "S" in s: return pd.Timestamp(s[:4] + "-04-01")
        if "W" in s: return pd.Timestamp(s[:4] + "-10-01")
        return pd.Timestamp(s[:4] + "-01-01")
    df["date"] = df["time"].apply(parse_sf_time)
    df = df.dropna(subset=["date", "value"])
    out = (df.groupby("date")["value"].sum().reset_index()
           .rename(columns={"value": "studentflow"}).sort_values("date"))
    # Resample to annual if semi-annual
    if (out["date"].dt.month != 1).any():
        out = (out.set_index("date")["studentflow"]
               .resample("YS").sum().reset_index())
    return out

@st.cache_data
def load_processed_v4(mtime):
    path = DATA_PROCESSED / "master_cliodynamics_v4.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)

# ── chart helpers ──────────────────────────────────────────────────────────

COLORS = px.colors.qualitative.Plotly
_CLIP_START = pd.Timestamp("2022-01-01")


def xaxis_range(dates):
    """Return xaxis_range kwarg if data has no pre-2022 values, else empty dict."""
    if dates.dropna().empty:
        return {}
    if pd.to_datetime(dates.dropna()).min() >= _CLIP_START:
        return {"xaxis_range": [_CLIP_START, pd.Timestamp.today()]}
    return {}


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
        **xaxis_range(df[x]),
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
        **xaxis_range(df[x]),
    )
    return fig

# ── page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Cliodynamics Dashboard",
    page_icon="📊",
    layout="wide",
)

# ── session state ─────────────────────────────────────────────────────────

# Handle ?view=processed query param (set by inline HTML links in raw tabs)
if "view" in st.query_params and "view" not in st.session_state:
    if st.query_params["view"] == "processed":
        st.session_state.view = "Processed Data"
        del st.query_params["view"]

if "view" not in st.session_state:
    st.session_state.view = "Raw Data"
if "proc_tab" not in st.session_state:
    st.session_state.proc_tab = "Wealth Pump"

# ── header ─────────────────────────────────────────────────────────────────

st.title("Cliodynamics — PSI v4 Dashboard")
st.caption("Political Stress Index for Germany · SDT / Turchin framework")

# ── Floating Reload Button ─────────────────────────────────────────────────
st.markdown('<div id="reload-marker"></div>', unsafe_allow_html=True)
if st.button("🔄", help="Clear data cache and reload from disk"):
    st.cache_data.clear()
    st.rerun()

st.markdown("""
    <style>
    /* Use the marker to find and pin the button container */
    div[data-testid="stElementContainer"]:has(#reload-marker) + div[data-testid="stElementContainer"],
    div.element-container:has(#reload-marker) + div.element-container {
        position: fixed;
        top: 20px;
        right: 120px;
        z-index: 1000000;
        width: auto !important;
    }
    
    /* Make the button look nice and square */
    div.element-container:has(#reload-marker) + div.element-container button {
        border-radius: 5px;
        padding: 0 !important;
        background-color: transparent;
        border: 1px solid rgba(128,128,128,0.3);
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        height: 38px;
        width: 38px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    div.element-container:has(#reload-marker) + div.element-container button:hover {
        border-color: #EF553B;
    }
    </style>
""", unsafe_allow_html=True)

view = st.segmented_control(
    "View",
    ["Raw Data", "Processed Data"],
    key="view",
    label_visibility="collapsed",
)

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# RAW DATA VIEW
# ══════════════════════════════════════════════════════════════════════════════

if view == "Raw Data":

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

    # ── 1. Wages ───────────────────────────────────────────────────────────
    with tabs[0]:
        st.subheader("Wage Indices — Destatis 62361-0001")
        st.markdown(
            "**Source:** `data/raw/data_wages.csv` · Monthly nominal and real wage index (2022=100). "
            "Used as denominator: "
            "<a href='?view=processed' target='_self' style='color:#00CC96;font-family:monospace;"
            "text-decoration:none;background:rgba(0,204,150,0.10);padding:1px 5px;border-radius:3px;"
            "cursor:pointer;'>wealth_pump = rent_index / wage_nominal</a>.",
            unsafe_allow_html=True,
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
            **xaxis_range(df["date"]),
        )
        st.plotly_chart(fig, width='stretch')

        with st.expander("Raw data"):
            st.dataframe(df, width='stretch')

    # ── 2. Rent / Price Index ──────────────────────────────────────────────
    with tabs[1]:
        st.subheader("Rent Price Index — Destatis 61111-0004")
        st.markdown(
            "**Source:** `data/raw/data_rent_index.csv` · Monthly CPI for rent category "
            "(*Tatsächliche Nettokaltmiete*, base 2020=100). "
            "Forms the numerator of "
            "<a href='?view=processed' target='_self' style='color:#00CC96;font-family:monospace;"
            "text-decoration:none;background:rgba(0,204,150,0.10);padding:1px 5px;border-radius:3px;"
            "cursor:pointer;'>wealth_pump = rent_index / wage_nominal</a>.",
            unsafe_allow_html=True,
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

    # ── 3. Students ────────────────────────────────────────────────────────
    with tabs[2]:
        st.subheader("University Students (Elite Fields) — Destatis 21311-0003 + 21311-0012")
        st.markdown(
            "**Source:** `data/raw/data_students.csv` · Semi-annual (winter/summer semester). "
            "Total enrolled in 13 elite-status fields: Rechtswissenschaft, Politikwissenschaft, "
            "BWL, VWL, Internationale BWL, Wirtschaftswissenschaften, Soziologie, Philosophie, "
            "Geschichte, Alte Geschichte, Wissenschaftsgeschichte, Medienwissenschaft, Kommunikationswissenschaft.  \n"
            "**Studentflow** (21311-0012): annual first-semester entrants — pipeline inflow. "
            "Graduates estimated as `entrants × 0.70` (30% dropout). "
            "Run `python src/loaders/load_studienanfaenger.py` to fetch Studentflow data."
        )
        df = load_students()
        sf_df = load_studentflow()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Semesters (enrolled)", len(df))
        c2.metric("Period start", str(df["date"].min().date()))
        c3.metric("Period end", str(df["date"].max().date()))
        c4.metric("Latest enrolled", f"{df['elite_candidates'].iloc[-1]:,.0f}")

        # ── Enrolled stock ──────────────────────────────────────────────
        st.markdown("#### Enrolled Stock (elite_candidates)")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df["date"], y=df["elite_candidates"],
            marker_color="#00CC96", name="Enrolled"
        ))
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["elite_candidates"],
            mode="lines+markers", line=dict(color="#636EFA", width=2),
            name="Trend"
        ))
        fig.update_layout(
            title="Elite Candidates (enrolled stock) per Semester",
            xaxis_title="Semester", yaxis_title="Students",
            height=380, hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=60, b=40, l=60, r=20),
            **xaxis_range(df["date"]),
        )
        st.plotly_chart(fig, width='stretch')

        # ── Studentflow (annual inflow) ─────────────────────────────────
        st.markdown("#### Studentflow — Annual Pipeline Inflow (21311-0012)")
        if sf_df is None:
            st.info("Studentflow data not available. Run `python src/loaders/load_studienanfaenger.py` to fetch it.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Years", len(sf_df))
            c2.metric("Period start", str(sf_df["date"].min().date()))
            c3.metric("Period end", str(sf_df["date"].max().date()))
            c4.metric("Latest entrants", f"{sf_df['studentflow'].iloc[-1]:,.0f}")

            sf_df["graduates_est"] = sf_df["studentflow"] * 0.70
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=sf_df["date"], y=sf_df["studentflow"],
                name="New Entrants (Studentflow)", marker_color="rgba(65,105,225,0.5)"
            ))
            fig2.add_trace(go.Scatter(
                x=sf_df["date"], y=sf_df["graduates_est"],
                mode="lines+markers", name="Est. Graduates (×0.70)",
                line=dict(color="#EF553B", width=2, dash="dash")
            ))
            fig2.update_layout(
                title="Studentflow — Annual First-Semester Entrants in Elite Fields",
                xaxis_title="Year", yaxis_title="Persons",
                height=380, hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=60, b=40, l=60, r=20),
                **xaxis_range(sf_df["date"]),
            )
            st.plotly_chart(fig2, width='stretch')

            with st.expander("Studentflow raw data"):
                st.dataframe(sf_df, width='stretch')

        with st.expander("Enrolled raw data"):
            st.dataframe(df, width='stretch')

    # ── 4. GDP ─────────────────────────────────────────────────────────────
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
                           margin=dict(t=60, b=40, l=60, r=20),
                           **xaxis_range(yoy_gdp.index.to_series()))
        st.plotly_chart(fig2, width='stretch')

        with st.expander("Raw data"):
            st.dataframe(df, width='stretch')

    # ── 5. CPI General ─────────────────────────────────────────────────────
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

    # ── 6. Food Prices ─────────────────────────────────────────────────────
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

    # ── 7. Youth Population ────────────────────────────────────────────────
    with tabs[6]:
        st.subheader("Youth Population (ages 15–24) — Destatis 12411-0005")
        st.markdown(
            "**Source:** `data/raw/data_youth_annual.csv` · Annual population share for ages 15–24. "
            "Calculated as `youth_15_24 / total_population`. "
            "Normalized to mean → `youth_bulge` PSI component."
        )
        df = load_youth()
        avg_share = df["youth_share"].mean()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Years", len(df))
        c2.metric("Period start", str(df["date"].min().date()))
        c3.metric("Latest Share", f"{df['youth_share'].iloc[-1]*100:.2f}%")
        c4.metric("Avg Share", f"{avg_share*100:.2f}%")
        
        st.markdown("#### Youth Share vs Absolute Count")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["date"], y=df["youth_share"],
            mode="lines+markers", name="Youth Share (%)",
            line=dict(color="#FECB52", width=3),
            yaxis="y1"
        ))
        fig.add_trace(go.Bar(
            x=df["date"], y=df["youth_count_k"],
            name="Youth Count (k)",
            marker_color="rgba(99,110,250,0.2)",
            yaxis="y2"
        ))
        
        fig.update_layout(
            title="Youth Profile (15–24 years)",
            xaxis_title="Year",
            yaxis=dict(title="Share of Total Population", tickformat=".1%", range=[0, max(df["youth_share"])*1.2]),
            yaxis2=dict(title="Absolute Count (k)", overlaying="y", side="right", showgrid=False),
            height=450, hovermode="x unified",
            margin=dict(t=60, b=40, l=60, r=60),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            **xaxis_range(df["date"]),
        )
        st.plotly_chart(fig, width='stretch')

        with st.expander("Raw data"):
            st.dataframe(df, width='stretch')

    # ── 8. Tax Revenue ─────────────────────────────────────────────────────
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
                           margin=dict(t=60, b=40, l=60, r=20),
                           **xaxis_range(yoy.index.to_series()))
        st.plotly_chart(fig2, width='stretch')

        # ── By tax type ────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Revenue by Tax Type")
        df_types = load_tax_by_type()
        all_types = sorted(df_types["tax_type"].unique())

        # Pre-select the largest types by total revenue for readability
        totals = df_types.groupby("tax_type")["revenue_k_eur"].sum().sort_values(ascending=False)
        top_defaults = totals.head(8).index.tolist()

        selected_types = st.multiselect(
            "Select tax types to display:",
            options=all_types,
            default=top_defaults,
            key="tax_type_select",
        )

        if selected_types:
            df_sel = df_types[df_types["tax_type"].isin(selected_types)]

            fig3 = go.Figure()
            colors = px.colors.qualitative.Plotly
            for i, ttype in enumerate(selected_types):
                sub = df_sel[df_sel["tax_type"] == ttype].sort_values("date")
                fig3.add_trace(go.Bar(
                    x=sub["date"], y=sub["revenue_k_eur"],
                    name=ttype,
                    marker_color=colors[i % len(colors)],
                ))
            fig3.update_layout(
                title="Tax Revenue by Type (Tsd. EUR)",
                xaxis_title="Year", yaxis_title="Tsd. EUR",
                barmode="group",
                height=450,
                margin=dict(t=60, b=40, l=60, r=20),
                legend=dict(orientation="h", y=-0.25),
                **xaxis_range(df_sel["date"]),
            )
            st.plotly_chart(fig3, width='stretch')

            # YoY change per selected type
            fig4 = go.Figure()
            for i, ttype in enumerate(selected_types):
                sub = df_sel[df_sel["tax_type"] == ttype].sort_values("date").set_index("date")
                yoy_t = sub["revenue_k_eur"].pct_change() * 100
                fig4.add_trace(go.Scatter(
                    x=yoy_t.index, y=yoy_t.values,
                    mode="lines+markers",
                    name=ttype,
                    line=dict(color=colors[i % len(colors)]),
                ))
            fig4.update_layout(
                title="YoY Change (%) by Tax Type",
                xaxis_title="Year", yaxis_title="%",
                height=380,
                margin=dict(t=60, b=40, l=60, r=20),
                legend=dict(orientation="h", y=-0.3),
                **xaxis_range(df_sel["date"]),
            )
            st.plotly_chart(fig4, width='stretch')
        else:
            st.info("Select at least one tax type above to render the comparison chart.")

        with st.expander("Raw data"):
            st.dataframe(df, width='stretch')

    # ── 9. Civil Servants ──────────────────────────────────────────────────
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
            **xaxis_range(df["date"]),
        )
        st.plotly_chart(fig, width='stretch')

        with st.expander("Raw data"):
            st.dataframe(df, width='stretch')

    # ── 10. Strikes ────────────────────────────────────────────────────────
    with tabs[9]:
        st.subheader("Strike Days — WSI Arbeitskampfbilanz")
        st.markdown(
            "**Source:** `data/raw/data_strikes_wsi.csv` · Annual days lost to strikes (×1000). "
            "Annual value carried flat across all months of the year as `strike_days` PSI component."
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
            **xaxis_range(df["date"]),
        )
        st.plotly_chart(fig, width='stretch')

        with st.expander("Raw data"):
            st.dataframe(df, width='stretch')

    # ── 11. Holders ────────────────────────────────────────────────────────
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
            **xaxis_range(df["date"]),
        )
        st.plotly_chart(fig, width='stretch')

        with st.expander("Raw data"):
            st.dataframe(df, width='stretch')

    # ── 12. Google Trends ──────────────────────────────────────────────────
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
            **xaxis_range(df["date"]),
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
                **xaxis_range(df["date"]),
            )
            st.plotly_chart(fig2, width='stretch')

        with st.expander("Raw data"):
            st.dataframe(df, width='stretch')

# ══════════════════════════════════════════════════════════════════════════════
# PROCESSED DATA VIEW
# ══════════════════════════════════════════════════════════════════════════════

elif view == "Processed Data":

    PROC_TABS = [
        "Wealth Pump",
        "Elite Pressure",
        "Macro Stress",
        "Food Pump",
        "Youth Bulge",
        "Strike Days",
        "State Capacity",
        "PSI v4 Final",
    ]

    proc_tabs = st.tabs(PROC_TABS)

    # ── Load PSI data ───────────────────────────────────────────────────────
    path_v4 = DATA_PROCESSED / "master_cliodynamics_v4.csv"
    mtime_v4 = os.path.getmtime(path_v4) if path_v4.exists() else 0
    
    try:
        psi_df = load_processed_v4(mtime_v4)
    except Exception as e:
        st.error(
            f"Error loading `data/processed/master_cliodynamics_v4.csv`: {e}  \n"
            "Run the pipeline first: `python src/preprocessors/process_final_psi.py`"
        )
        st.stop()

    # ── Wealth Pump ─────────────────────────────────────────────────────────
    with proc_tabs[0]:
        st.subheader("Wealth Pump — rent / wage ratio")
        st.markdown(
            "**Formula:** `wealth_pump = rent_index / wage_nominal`  \n"
            "Rising values = housing costs outpacing wages (SDT inequality driver).  \n"
            "**Source data:** Destatis 61111-0004 (rent CPI) + 62361-0001 (nominal wage index)."
        )

        wp = psi_df[["date", "wealth_pump"]].dropna(subset=["wealth_pump"])

        if wp.empty:
            st.warning("No wealth_pump data found. Run the pipeline first.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Period start", str(wp["date"].min().date()))
            c2.metric("Period end", str(wp["date"].max().date()))
            c3.metric("Min", f"{wp['wealth_pump'].min():.1f}")
            c4.metric("Latest", f"{wp['wealth_pump'].iloc[-1]:.1f}")

            ma_wp = st.slider("Moving average window (months)", 1, 24, 6, key="wp_ma")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=wp["date"], y=wp["wealth_pump"],
                mode="lines", name="Wealth Pump",
                line=dict(color="#EF553B", width=1.5), opacity=0.45,
                fill="tozeroy", fillcolor="rgba(239,85,59,0.08)"
            ))
            fig.add_trace(go.Scatter(
                x=wp["date"],
                y=wp["wealth_pump"].rolling(ma_wp, min_periods=1).mean(),
                mode="lines", name=f"{ma_wp}m MA",
                line=dict(color="#636EFA", width=2.5)
            ))
            fig.update_layout(
                title="Wealth Pump (rent / wage)",
                xaxis_title="Date", yaxis_title="Ratio",
                hovermode="x unified", height=440,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=60, b=40, l=60, r=20),
                **xaxis_range(wp["date"]),
            )
            st.plotly_chart(fig, width='stretch')

            with st.expander("Processed data"):
                st.dataframe(wp, width='stretch')

    # ── Elite Pressure ──────────────────────────────────────────────────────
    with proc_tabs[1]:
        st.subheader("Elite Pressure — overproduction of aspirants")
        st.markdown(
            "**Formula:** `elite_pressure = nm(elite_candidates) × (1 + frustrated_fraction)`  \n"
            "`frustrated_fraction = max(0, annual_graduates − annual_openings) / annual_graduates`  \n"
            "`annual_graduates = elite_candidates × 0.70 / 5` · `annual_openings = holders × 0.05`  \n"
            "**Source data:** Destatis 21311-0003 (enrolled) + Mikrozensus (Führungskräfte)."
        )

        ep_cols = ["elite_candidates", "holders", "frustrated_fraction", "elite_pressure"]
        ep = psi_df[["date"] + [c for c in ep_cols if c in psi_df.columns]].dropna(
            subset=[c for c in ["elite_candidates", "elite_pressure"] if c in psi_df.columns]
        )

        if ep.empty:
            st.warning("No elite pressure data found.")
        else:
            latest = ep.iloc[-1]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Period start", str(ep["date"].min().date()))
            c2.metric("Period end", str(ep["date"].max().date()))
            if "elite_candidates" in ep.columns:
                c3.metric("Latest enrolled", f"{latest['elite_candidates']:,.0f}")
            if "frustrated_fraction" in ep.columns:
                c4.metric("Latest frustrated fraction", f"{latest['frustrated_fraction']:.3f}")

            # Annual flow comparison: graduates estimate vs openings
            if "elite_candidates" in ep.columns and "holders" in ep.columns:
                annual_grad = ep["elite_candidates"] * 0.70 / 5
                annual_open = ep["holders"] * 0.05
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=ep["date"], y=annual_grad,
                    mode="lines", name="Est. Annual Graduates (enrolled×0.7/5)",
                    line=dict(color="#636EFA", width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=ep["date"], y=annual_open,
                    mode="lines", name="Annual Elite Openings (holders×5%)",
                    line=dict(color="#00CC96", width=2, dash="dash")
                ))
                fig.add_trace(go.Scatter(
                    x=ep["date"], y=annual_grad - annual_open,
                    mode="lines", name="Surplus Aspirants",
                    line=dict(color="#EF553B", width=1.5, dash="dot"),
                    fill="tozeroy", fillcolor="rgba(239,85,59,0.08)"
                ))
                fig.update_layout(
                    title="Annual Elite Graduate Flow vs Available Openings",
                    xaxis_title="Date", yaxis_title="Persons / yr",
                    hovermode="x unified", height=380,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(t=60, b=40, l=60, r=20),
                    **xaxis_range(ep["date"]),
                )
                st.plotly_chart(fig, width='stretch')

            # Frustrated fraction
            if "frustrated_fraction" in ep.columns:
                ff = ep[["date", "frustrated_fraction"]].dropna()
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=ff["date"], y=ff["frustrated_fraction"],
                    mode="lines", name="Frustrated Fraction",
                    line=dict(color="#FFA15A", width=2),
                    fill="tozeroy", fillcolor="rgba(255,161,90,0.12)"
                ))
                fig2.update_layout(
                    title="Frustrated Fraction — share of graduates without elite positions",
                    xaxis_title="Date", yaxis_title="Fraction [0–1]",
                    hovermode="x unified", height=300,
                    margin=dict(t=60, b=40, l=60, r=20),
                    **xaxis_range(ff["date"]),
                )
                st.plotly_chart(fig2, width='stretch')

            # Elite pressure composite
            if "elite_pressure" in ep.columns:
                epp = ep[["date", "elite_pressure"]].dropna()
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(
                    x=epp["date"], y=epp["elite_pressure"],
                    mode="lines", name="Elite Pressure",
                    line=dict(color="#EF553B", width=2.5),
                    fill="tozeroy", fillcolor="rgba(239,85,59,0.10)"
                ))
                fig3.update_layout(
                    title="Elite Pressure Composite (nm(candidates) × (1 + frustrated_fraction))",
                    xaxis_title="Date", yaxis_title="Index",
                    hovermode="x unified", height=300,
                    margin=dict(t=60, b=40, l=60, r=20),
                    **xaxis_range(epp["date"]),
                )
                st.plotly_chart(fig3, width='stretch')

            with st.expander("Processed data"):
                st.dataframe(ep, width='stretch')

    # ── Macro Stress ────────────────────────────────────────────────────────
    with proc_tabs[2]:
        st.subheader("Macroeconomic Stress — GDP growth minus real wage growth")
        st.markdown(
            "**Formula:** `m_econ = gdp_growth − real_wage_growth`  \n"
            "Positive = GDP growing faster than wages (workers losing share of output).  \n"
            "**Source data:** Destatis 81111-0001 (GDP) + 62361-0001 (real wage index)."
        )

        me_cols = [c for c in ["gdp_growth", "m_econ"] if c in psi_df.columns]
        me = psi_df[["date"] + me_cols].copy()
        if "m_econ" in me.columns:
            me["m_econ"] = me["m_econ"].replace(0, pd.NA)
        me = me.dropna(subset=[c for c in ["m_econ"] if c in me.columns])

        if me.empty:
            st.warning("No m_econ data found.")
        else:
            latest_me = me.dropna().iloc[-1]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Period start", str(me["date"].min().date()))
            c2.metric("Period end", str(me["date"].max().date()))
            if "gdp_growth" in me.columns:
                c3.metric("Latest GDP growth", f"{me['gdp_growth'].dropna().iloc[-1]:.3f}")
            if "m_econ" in me.columns:
                c4.metric("Latest m_econ", f"{me['m_econ'].dropna().iloc[-1]:.3f}")

            ma_me = st.slider("Moving average window (months)", 1, 24, 6, key="me_ma")

            fig = go.Figure()
            if "m_econ" in me.columns:
                me_s = me[["date", "m_econ"]].dropna()
                fig.add_trace(go.Scatter(
                    x=me_s["date"], y=me_s["m_econ"],
                    mode="lines", name="m_econ (raw)",
                    line=dict(color="#FFA15A", width=1.5), opacity=0.5
                ))
                fig.add_trace(go.Scatter(
                    x=me_s["date"],
                    y=me_s["m_econ"].rolling(ma_me, min_periods=1).mean(),
                    mode="lines", name=f"m_econ {ma_me}m MA",
                    line=dict(color="#EF553B", width=2.5)
                ))
            fig.add_hline(y=0, line_dash="dash", line_color="gray",
                          annotation_text="neutral")
            fig.update_layout(
                title="Macroeconomic Stress (m_econ)",
                xaxis_title="Date", yaxis_title="Stress",
                hovermode="x unified", height=440,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=60, b=40, l=60, r=20),
                xaxis_range=[me["date"].min(), me["date"].max()],
            )
            st.plotly_chart(fig, width='stretch')

            with st.expander("Processed data"):
                st.dataframe(me, width='stretch')

    # ── Food Pump ───────────────────────────────────────────────────────────
    with proc_tabs[3]:
        st.subheader("Food Pump — food inflation vs general CPI")
        st.markdown(
            "**Formula:** `food_pump = food_CPI_yoy − general_CPI_yoy`  \n"
            "Positive = food prices rising faster than general inflation (regressive shock).  \n"
            "**Source data:** Destatis 61111-0004 (CC13-01 food vs general)."
        )

        fp = psi_df[["date", "food_pump"]].dropna(subset=["food_pump"]) if "food_pump" in psi_df.columns else pd.DataFrame()

        if fp.empty:
            st.warning("No food_pump data found.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Period start", str(fp["date"].min().date()))
            c2.metric("Period end", str(fp["date"].max().date()))
            c3.metric("Max", f"{fp['food_pump'].max():.3f}")
            c4.metric("Latest", f"{fp['food_pump'].iloc[-1]:.3f}")

            ma_fp = st.slider("Moving average window (months)", 1, 24, 6, key="fp_ma")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=fp["date"], y=fp["food_pump"],
                mode="lines", name="Food Pump (raw)",
                line=dict(color="#FF6692", width=1.5), opacity=0.5
            ))
            fig.add_trace(go.Scatter(
                x=fp["date"],
                y=fp["food_pump"].rolling(ma_fp, min_periods=1).mean(),
                mode="lines", name=f"{ma_fp}m MA",
                line=dict(color="#B6E880", width=2.5)
            ))
            fig.add_hline(y=0, line_dash="dash", line_color="gray",
                          annotation_text="neutral")
            fig.update_layout(
                title="Food Pump (food inflation − general CPI)",
                xaxis_title="Date", yaxis_title="Differential",
                hovermode="x unified", height=440,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=60, b=40, l=60, r=20),
                **xaxis_range(fp["date"]),
            )
            st.plotly_chart(fig, width='stretch')

            with st.expander("Processed data"):
                st.dataframe(fp, width='stretch')

    # ── Youth Bulge ─────────────────────────────────────────────────────────
    with proc_tabs[4]:
        st.subheader("Youth Bulge — relative share, normalized")
        st.markdown(
            "**Formula:** `youth_bulge = (youth_pop / total_pop) / mean(youth_share)`  \n"
            "Represented as a ratio of the current demographic weight to the historical average.  \n"
            "This accounts for aging societies where absolute counts might fall while structural weight remains.  \n"
            "**Source data:** Destatis 12411-0005."
        )

        yb = psi_df[["date", "youth_bulge"]].dropna(subset=["youth_bulge"]) if "youth_bulge" in psi_df.columns else pd.DataFrame()

        if yb.empty:
            st.warning("No youth_bulge data found.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Period start", str(yb["date"].min().date()))
            c2.metric("Period end", str(yb["date"].max().date()))
            c3.metric("Max", f"{yb['youth_bulge'].max():.3f}")
            c4.metric("Latest", f"{yb['youth_bulge'].iloc[-1]:.3f}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=yb["date"], y=yb["youth_bulge"],
                mode="lines", name="Youth Bulge",
                line=dict(color="#FECB52", width=2),
                fill="tozeroy", fillcolor="rgba(254,203,82,0.12)"
            ))
            fig.add_hline(y=1.0, line_dash="dash", line_color="gray",
                          annotation_text="Mean (1.0)")
            fig.update_layout(
                title="Youth Bulge (normalized to mean)",
                xaxis_title="Date", yaxis_title="Ratio to mean",
                hovermode="x unified", height=440,
                margin=dict(t=60, b=40, l=60, r=20),
                **xaxis_range(yb["date"]),
            )
            st.plotly_chart(fig, width='stretch')

            with st.expander("Processed data"):
                st.dataframe(yb, width='stretch')

    # ── Strike Days ─────────────────────────────────────────────────────────
    with proc_tabs[5]:
        st.subheader("Strike Days — annual values (flat per year)")
        st.markdown(
            "**Source:** WSI Arbeitskampfbilanz (annual) — same value carried across all months of each year.  \n"
            "Higher = more labour conflict, raising PSI via `nm(strike_days)`."
        )

        sd = psi_df[["date", "strike_days"]].dropna(subset=["strike_days"]) if "strike_days" in psi_df.columns else pd.DataFrame()
        if not sd.empty:
            sd = sd.copy()
            sd["year"] = sd["date"].dt.year
            sd = sd.groupby("year", as_index=False)["strike_days"].first()
            sd["date"] = pd.to_datetime(sd["year"].astype(str) + "-01-01")

        if sd.empty:
            st.warning("No strike_days data found.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Period start", str(sd["date"].min().date()))
            c2.metric("Period end", str(sd["date"].max().date()))
            c3.metric("Max (k days)", f"{sd['strike_days'].max():,.1f}")
            c4.metric("Latest (k days)", f"{sd['strike_days'].iloc[-1]:,.1f}")

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=sd["date"], y=sd["strike_days"],
                marker_color="#FF97FF", name="Strike Days (annual)"
            ))
            fig.update_layout(
                title="Strike Days (annual WSI value)",
                xaxis_title="Date", yaxis_title="Days lost (×1000)",
                hovermode="x unified", height=440,
                margin=dict(t=60, b=40, l=60, r=20),
                **xaxis_range(sd["date"]),
            )
            st.plotly_chart(fig, width='stretch')

            with st.expander("Processed data"):
                st.dataframe(sd, width='stretch')

    # ── State Capacity ──────────────────────────────────────────────────────
    with proc_tabs[6]:
        st.subheader("State Capacity — tax stability × civil servant headcount")
        st.markdown(
            "**Formula:** `s_capacity = tax_stability × civil_servant_factor`  \n"
            "`tax_stability = 1 / (1 + |ΔtaxRevenue YoY|)` · `civil_servant_factor = nm(civil_servants)`  \n"
            "Higher = stronger state (acts as PSI divisor, dampening stress).  \n"
            "**Source data:** Destatis 71211-0001 (tax) + 74111-0001 (civil servants)."
        )

        sc_cols = [c for c in ["civil_servants", "s_capacity"] if c in psi_df.columns]
        sc = psi_df[["date"] + sc_cols].dropna(subset=[c for c in ["s_capacity"] if c in psi_df.columns])

        if sc.empty:
            st.warning("No s_capacity data found.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Period start", str(sc["date"].min().date()))
            c2.metric("Period end", str(sc["date"].max().date()))
            if "s_capacity" in sc.columns:
                c3.metric("Min s_capacity", f"{sc['s_capacity'].min():.3f}")
                c4.metric("Latest s_capacity", f"{sc['s_capacity'].dropna().iloc[-1]:.3f}")

            if "civil_servants" in sc.columns:
                cs_s = sc[["date", "civil_servants"]].dropna()
                fig_cs = go.Figure()
                fig_cs.add_trace(go.Scatter(
                    x=cs_s["date"], y=cs_s["civil_servants"],
                    mode="lines+markers", name="Civil Servants",
                    line=dict(color="#B6E880", width=2)
                ))
                fig_cs.update_layout(
                    title="Civil Servants Headcount (normalized, used in s_capacity)",
                    xaxis_title="Date", yaxis_title="Normalized count",
                    hovermode="x unified", height=340,
                    margin=dict(t=60, b=40, l=60, r=20),
                    **xaxis_range(cs_s["date"]),
                )
                st.plotly_chart(fig_cs, width='stretch')

            if "s_capacity" in sc.columns:
                sc_s = sc[["date", "s_capacity"]].dropna()
                fig_sc = go.Figure()
                fig_sc.add_trace(go.Scatter(
                    x=sc_s["date"], y=sc_s["s_capacity"],
                    mode="lines", name="State Capacity",
                    line=dict(color="#19D3F3", width=2.5),
                    fill="tozeroy", fillcolor="rgba(25,211,243,0.10)"
                ))
                fig_sc.update_layout(
                    title="State Capacity Composite (PSI divisor)",
                    xaxis_title="Date", yaxis_title="s_capacity",
                    hovermode="x unified", height=340,
                    margin=dict(t=60, b=40, l=60, r=20),
                    **xaxis_range(sc_s["date"]),
                )
                st.plotly_chart(fig_sc, width='stretch')

            with st.expander("Processed data"):
                st.dataframe(sc, width='stretch')

    # ── PSI v4 Final ────────────────────────────────────────────────────────
    with proc_tabs[7]:
        st.subheader("PSI v4 — Political Stress Index (final)")
        st.markdown(
            "**Formula:**  \n"
            "`psi_v4 = rolling_mean_12(nm(wealth_pump) × elite_pressure × nm(m_econ)`  \n"
            "`        × nm(food_pump) × nm(youth_bulge) × nm(strike_days) / s_capacity)`  \n\n"
            "Values approaching 1 = maximum modelled stress. "
            "The 12-month rolling mean smooths monthly noise."
        )

        psi_cols = [c for c in ["psi_v4_raw", "psi_v4"] if c in psi_df.columns]
        psi = psi_df[["date"] + psi_cols].dropna(subset=[c for c in ["psi_v4"] if c in psi_df.columns])

        if psi.empty:
            st.warning("No psi_v4 data found.")
        else:
            latest_psi = psi["psi_v4"].dropna().iloc[-1]
            max_psi = psi["psi_v4"].max()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Period start", str(psi["date"].min().date()))
            c2.metric("Period end", str(psi["date"].max().date()))
            c3.metric("All-time max", f"{max_psi:.4f}")
            c4.metric("Latest PSI v4", f"{latest_psi:.4f}")

            fig = go.Figure()
            if "psi_v4_raw" in psi.columns:
                raw_s = psi[["date", "psi_v4_raw"]].dropna()
                fig.add_trace(go.Scatter(
                    x=raw_s["date"], y=raw_s["psi_v4_raw"],
                    mode="lines", name="PSI v4 raw",
                    line=dict(color="#636EFA", width=1), opacity=0.35
                ))
            psi_s = psi[["date", "psi_v4"]].dropna()
            fig.add_trace(go.Scatter(
                x=psi_s["date"], y=psi_s["psi_v4"],
                mode="lines", name="PSI v4 (12m rolling mean)",
                line=dict(color="#EF553B", width=3),
                fill="tozeroy", fillcolor="rgba(239,85,59,0.10)"
            ))

            # Highlight forecast window 2026-2027
            fig.add_vrect(
                x0="2026-01-01", x1="2027-12-31",
                fillcolor="rgba(255,200,0,0.10)",
                line_width=0,
                annotation_text="Forecast window 2026–27",
                annotation_position="top left",
                annotation_font_size=11,
            )

            fig.update_layout(
                title="Political Stress Index v4 — Germany",
                xaxis_title="Date", yaxis_title="PSI v4",
                hovermode="x unified", height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=60, b=40, l=60, r=20),
                **xaxis_range(psi["date"]),
            )
            st.plotly_chart(fig, width='stretch')

            # Component correlation heatmap
            component_cols = [c for c in [
                "wealth_pump", "elite_pressure", "m_econ", "food_pump",
                "youth_bulge", "strike_days", "s_capacity", "psi_v4"
            ] if c in psi_df.columns]
            corr_df = psi_df[component_cols].dropna()
            if not corr_df.empty and len(corr_df) > 5:
                corr_matrix = corr_df.corr()
                fig2 = go.Figure(go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns.tolist(),
                    y=corr_matrix.index.tolist(),
                    colorscale="RdBu_r", zmid=0,
                    text=corr_matrix.round(2).values,
                    texttemplate="%{text}",
                    hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>",
                ))
                fig2.update_layout(
                    title="Component Correlation Matrix",
                    height=420,
                    margin=dict(t=60, b=40, l=120, r=20),
                )
                st.plotly_chart(fig2, width='stretch')

            with st.expander("Full processed dataset (v4)"):
                st.dataframe(psi_df, width='stretch')
