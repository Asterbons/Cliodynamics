import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Set aesthetic
# (Using Plotly white theme in layout)

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_DIR = PROJECT_ROOT / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MON = {
    'Januar':1, 'Februar':2, 'März':3, 'April':4, 'Mai':5, 'Juni':6,
    'Juli':7, 'August':8, 'September':9, 'Oktober':10, 'November':11, 'Dezember':12
}

def get_pts(df, name):
    if df.empty: return pd.Series()
    df = df.copy()
    
    # 1. Geo (Deutschland) - optimized search
    mask = pd.Series([False]*len(df), index=df.index)
    for c in df.columns:
        if 'attribute' in str(c):
            mask |= (df[c].astype(str).str.strip() == 'DG')
            mask |= df[c].astype(str).str.contains('Deutschland insgesamt|insgesamt', na=False, case=False)
    
    df_f = df[mask] if mask.any() else df
    
    # 2. Months
    df_f['m_idx'] = 1
    for c in df_f.columns:
        if 'label' in str(c):
            m_map = df_f[c].map(MON)
            if m_map.notna().any():
                df_f['m_idx'] = m_map.fillna(1)
                break
            
    # 3. Final Build
    df_f['yr'] = pd.to_numeric(df_f['time'], errors='coerce').fillna(2020).astype(int)
    df_f['dt'] = pd.to_datetime(df_f['yr'].astype(str) + '-' + df_f['m_idx'].astype(int).astype(str) + '-01')
    df_f['val'] = pd.to_numeric(df_f['value'], errors='coerce')
    
    # Apply string replacement for German decimals if column is object type
    val_str = df_f['value'].astype(str).str.replace(',', '.', regex=False)
    # The Destatis datasets often put '-' or '...' for missing values
    df_f['val'] = pd.to_numeric(val_str, errors='coerce')
    
    # Take mean over all rows for the same timestamp (robust for 5-digit CPI aggregations)
    s = df_f.dropna(subset=['val']).groupby('dt')['val'].mean().sort_index()
    print(f"    - Series {name} Result: {len(s)} pts.")
    return s

def nm(s):
    v = pd.to_numeric(s, errors='coerce').fillna(s.median() if s.notna().any() else 0.5)
    mi, ma = v.min(), v.max()
    if mi == ma: return v * 0 + 0.5
    return (v - mi) / (ma - mi + 1e-9) + 0.1

def main():
    print("PSI v4 Pipeline (Building Output Dashboard)...")
    v2 = pd.read_csv(DATA_PROCESSED / 'master_cliodynamics_v2.csv', parse_dates=['date']).sort_values('date')
    v2 = v2.set_index('date')
    idx = v2.index

    # Data
    y_raw = pd.read_csv(DATA_RAW / 'data_youth_annual.csv', sep=';', decimal=',', low_memory=False)
    c_raw = pd.read_csv(DATA_RAW / 'data_cpi_general.csv', sep=';', decimal=',', low_memory=False)
    f_raw = pd.read_csv(DATA_RAW / 'data_food_prices.csv', sep=';', decimal=',', low_memory=False)
    g_raw = pd.read_csv(DATA_RAW / 'data_gdp_quarterly.csv', sep=';', decimal=',', low_memory=False)
    t_raw = pd.read_csv(DATA_RAW / 'data_tax_revenue.csv', sep=';', decimal=',', low_memory=False)

    # CPI overall (c_raw contains 5-digit items, so we mean aggregate the whole table)
    c_ser = get_pts(c_raw, "Main CPI Aggregate")
    f_ser = get_pts(f_raw[f_raw['3_variable_attribute_code'] == 'CC13-01'], "Food CPI")
    g_ser = get_pts(g_raw[g_raw['value_variable_code'] == 'VGR014'], "GDP Index")
    t_ser = get_pts(t_raw, "Tax Revenue")

    # Youth
    y_sum = y_raw[y_raw['2_variable_attribute_code'].str.contains('ALT01[5-9]|ALT02[0-4]', na=False)].groupby('time')['value'].sum()
    y_df = pd.DataFrame({'date': pd.to_datetime(y_sum.index.astype(str).str[:4] + '-01-01'), 'youth': y_sum.values})
    v2 = v2.reset_index()
    v2 = pd.merge_asof(v2, y_df, on='date', direction='backward').set_index('date')
    v2['youth_bulge'] = (v2['youth'].interpolate().fillna(method='bfill') / v2['youth'].mean())

    # Indicators
    v2['gdp_growth'] = g_ser.reindex(idx).interpolate(method='time').fillna(method='bfill').pct_change(12)*100 if not g_ser.empty else 0.0
    if not c_ser.empty and not f_ser.empty:
        v2['food_pump'] = (f_ser.pct_change(12)*100).reindex(idx).interpolate().fillna(0) - (c_ser.pct_change(12)*100).reindex(idx).interpolate().fillna(0)
    else:
        v2['food_pump'] = 0.0

    v2['m_econ'] = v2['gdp_growth'] - (v2['wage_real'].pct_change(12).fillna(0)*100)
    t_mon = t_ser.reindex(idx).interpolate() if not t_ser.empty else v2['gdp_growth']
    t_mon = t_mon.fillna(method='bfill') # Fix missing data before 2022
    v2['s_capacity'] = 1.0 / (1.0 + t_mon.pct_change(12).rolling(12).std().fillna(0.1))
    v2['s_capacity'] = v2['s_capacity'].fillna(method='bfill').fillna(1.0)

    # Stress model
    v2['psi_v4_raw'] = (nm(v2['wealth_pump']) * nm(v2['elite_candidates']) * 
                        nm(v2['m_econ']) * nm(v2['food_pump']) * nm(v2['youth_bulge'])) / v2['s_capacity'].fillna(1.0)
    v2['psi_v4'] = v2['psi_v4_raw'].rolling(12, center=True, min_periods=1).mean()
    
    v2.to_csv(DATA_PROCESSED / 'master_cliodynamics_v4.csv')

    # Visual (DASHBOARD) using Plotly
    fig = make_subplots(
        rows=4, cols=2,
        shared_xaxes=False,
        vertical_spacing=0.1,
        horizontal_spacing=0.08,
        subplot_titles=(
            "POLITICAL STRESS INDEX (PSI) v4 - Germany",
            "Mass Mobilization Factors (M)", "Elite Pressure (E)",
            "Macroeconomic Trends", "Demographic Context (Youth Bulge)",
            "State Institutional Stability (S)"
        ),
        specs=[[{"colspan": 2}, None],
               [{}, {}],
               [{}, {}],
               [{"colspan": 2}, None]]
    )
    
    # 1. PSI
    fig.add_trace(go.Scatter(
        x=idx, y=v2['psi_v4'],
        name='PSI v4',
        line=dict(color='firebrick', width=4),
        fill='tozeroy', fillcolor='rgba(178, 34, 34, 0.1)'
    ), row=1, col=1)

    # 2. M
    fig.add_trace(go.Scatter(
        x=idx, y=v2['m_econ'], name='Relative Deprivation', line=dict(color='#FF4B2B')
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=idx, y=v2['food_pump'], name='Food Pump (Inflation Diff)', line=dict(color='#FFD200')
    ), row=2, col=1)

    # 3. E
    fig.add_trace(go.Scatter(
        x=idx, y=nm(v2['wealth_pump']), name='Wealth Pump Index', line=dict(color='#9D50BB')
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=idx, y=nm(v2['elite_candidates']), name='Elite Candidates', line=dict(color='royalblue')
    ), row=2, col=2)

    # 4. GDP
    fig.add_trace(go.Scatter(
        x=idx, y=v2['gdp_growth'], name='Real GDP Growth', line=dict(color='seagreen')
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=idx, y=v2['wage_real'].pct_change(12)*100, name='Real Wage Growth', line=dict(color='orange')
    ), row=3, col=1)

    # 5. Youth
    fig.add_trace(go.Scatter(
        x=idx, y=v2['youth_bulge'], name='Youth Bulge Index', line=dict(color='goldenrod'),
        fill='tozeroy', fillcolor='rgba(218, 165, 32, 0.1)'
    ), row=3, col=2)

    # 6. S
    fig.add_trace(go.Scatter(
        x=idx, y=v2['s_capacity'], name='Extraction Stability', line=dict(color='slateblue', width=3),
        fill='tozeroy', fillcolor='rgba(106, 90, 205, 0.1)'
    ), row=4, col=1)

    # Layout
    fig.update_layout(
        template='plotly_white',
        height=1400,
        hovermode="x unified",
        showlegend=True,
        title_text="Dashboard: PSI v4 & Structural-Demographic Drivers",
        title_font_size=24
    )
    
    # Update axes specific logic
    fig.update_yaxes(range=[0.8, 1.1], row=4, col=1)
    
    html_out = OUTPUT_DIR / 'psi_v4_dashboard.html'
    fig.write_html(html_out)
    print(f"DONE: Visualized Dashboard securely in {html_out}")
    fig.show()

if __name__ == "__main__":
    main()
