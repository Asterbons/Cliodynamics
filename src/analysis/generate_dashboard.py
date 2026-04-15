import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing

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
    cs_raw = pd.read_csv(DATA_RAW / 'data_civil_servants.csv', sep=';', decimal=',')
    strikes_raw = pd.read_csv(DATA_RAW / 'data_strikes_wsi.csv')
    holders_raw = pd.read_csv(DATA_RAW / 'data_holders.csv')

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
    s_tax = 1.0 / (1.0 + t_mon.pct_change(12).rolling(12).std().fillna(0.1))

    # Civil servants component: more staff = higher capacity
    cs_total = cs_raw[cs_raw['2_variable_attribute_label'].str.contains('Insgesamt', na=False)]
    cs_df = pd.DataFrame({
        'date': pd.to_datetime(cs_total['time'].astype(str).str[:4] + '-06-01'),
        'cs_val': pd.to_numeric(cs_total['value'], errors='coerce')
    }).dropna().sort_values('date').set_index('date')
    cs_monthly = cs_df['cs_val'].reindex(idx).interpolate(method='time').fillna(method='bfill').fillna(method='ffill')
    v2['civil_servants'] = cs_monthly
    cs_norm = (cs_monthly - cs_monthly.min()) / (cs_monthly.max() - cs_monthly.min() + 1e-9) * 0.5 + 0.5
    v2['s_capacity'] = (s_tax * cs_norm).fillna(method='bfill').fillna(1.0)

    # Strike data (Mass Mobilization)
    strikes_df = pd.DataFrame({
        'date': pd.to_datetime(strikes_raw['year'].astype(str) + '-07-01'),
        'strike_days': strikes_raw['strike_days_k'].astype(float)
    }).set_index('date')
    strike_monthly = strikes_df['strike_days'].reindex(idx).interpolate(method='time').fillna(method='bfill').fillna(method='ffill')
    v2['strike_days'] = strike_monthly

    # Holders modeling (Elite Overproduction Pressure)
    holders_df = pd.DataFrame({
        'date': pd.to_datetime(holders_raw['year'].astype(str) + '-01-01'),
        'holders': holders_raw['holders_k'].astype(float) * 1000
    }).set_index('date')
    holders_monthly = holders_df['holders'].reindex(idx).interpolate(method='time').fillna(method='bfill').fillna(method='ffill')
    v2['holders'] = holders_monthly

    # frustrated_fraction: annual graduates vs annual elite position openings
    DROPOUT_RATE = 0.30
    AVG_STUDY_YEARS = 5
    TURNOVER_RATE = 0.05
    annual_graduates = v2['elite_candidates'] * (1 - DROPOUT_RATE) / AVG_STUDY_YEARS
    annual_openings = holders_monthly * TURNOVER_RATE
    v2['frustrated_fraction'] = np.clip(
        (annual_graduates - annual_openings) / annual_graduates.replace(0, np.nan),
        0, 1
    ).fillna(0)
    v2['elite_pressure'] = nm(v2['elite_candidates']) * (1 + v2['frustrated_fraction'])

    # Stress model (uses elite_pressure instead of raw elite_candidates)
    v2['psi_v4_raw'] = (nm(v2['wealth_pump']) * v2['elite_pressure'] * 
                        nm(v2['m_econ']) * nm(v2['food_pump']) * nm(v2['youth_bulge']) *
                        nm(v2['strike_days'])) / v2['s_capacity'].fillna(1.0)
    v2['psi_v4'] = v2['psi_v4_raw'].rolling(12, center=True, min_periods=1).mean()

    # ═══ FORECAST PSI to 2027 ═══
    psi_hist = v2['psi_v4'].dropna()
    forecast_start = psi_hist.index[-1] + pd.DateOffset(months=1)
    forecast_end = pd.Timestamp('2027-12-01')
    forecast_idx = pd.date_range(forecast_start, forecast_end, freq='MS')
    n_fc = len(forecast_idx)

    try:
        # Holt-Winters exponential smoothing for trend + seasonal forecast
        model = ExponentialSmoothing(
            psi_hist.values, trend='add', seasonal=None,
            initialization_method='estimated'
        ).fit(optimized=True)
        fc_values = model.forecast(n_fc)
        fc_series = pd.Series(fc_values, index=forecast_idx)
        # Confidence band: ±1 std of last 12 months residuals
        residual_std = (psi_hist - model.fittedvalues).std()
        fc_upper = fc_series + 1.96 * residual_std
        fc_lower = (fc_series - 1.96 * residual_std).clip(lower=0)
        print(f"  Forecast: {n_fc} months ({forecast_start.strftime('%Y-%m')} to {forecast_end.strftime('%Y-%m')})")
    except Exception as e:
        # Fallback: simple linear extrapolation from last 24 months
        print(f"  Holt-Winters failed ({e}), using linear extrapolation...")
        last_24 = psi_hist.iloc[-24:]
        x = np.arange(len(last_24))
        slope, intercept = np.polyfit(x, last_24.values, 1)
        fc_x = np.arange(len(last_24), len(last_24) + n_fc)
        fc_values = slope * fc_x + intercept
        fc_series = pd.Series(fc_values, index=forecast_idx)
        residual_std = last_24.std() * 0.5
        fc_upper = fc_series + 1.96 * residual_std
        fc_lower = (fc_series - 1.96 * residual_std).clip(lower=0)
    
    v2.to_csv(DATA_PROCESSED / 'master_cliodynamics_v4.csv')

    # Load Studentflow (first-semester entrants, 21311-0012) for flow panel — optional
    studentflow_ser = None
    try:
        sf_raw = pd.read_csv(DATA_RAW / 'data_studienanfaenger.csv', sep=';', decimal=',', low_memory=False)
        # Sum all elite fields per year/semester
        sf_raw['val'] = pd.to_numeric(sf_raw['value'].astype(str).str.replace(',', '.', regex=False), errors='coerce')
        # Group by time (year or semester), sum across fields, annual-ise
        sf_grouped = sf_raw.dropna(subset=['val']).groupby('time')['val'].sum()
        # Convert time index to datetime; semester codes (e.g. 2022S → 2022-04, 2022W → 2022-10) or plain year
        def parse_sf_time(t):
            s = str(t)
            if 'S' in s:   return pd.Timestamp(s[:4] + '-04-01')
            if 'W' in s:   return pd.Timestamp(s[:4] + '-10-01')
            return pd.Timestamp(s[:4] + '-01-01')
        sf_idx = pd.DatetimeIndex([parse_sf_time(t) for t in sf_grouped.index])
        studentflow_ser = pd.Series(sf_grouped.values, index=sf_idx).sort_index()
        # If semi-annual (winter+summer), resample to annual sum
        if (studentflow_ser.index.month != 1).any():
            studentflow_ser = studentflow_ser.resample('YS').sum()
        print(f"  Studentflow loaded: {len(studentflow_ser)} annual points")
    except FileNotFoundError:
        print("  data_studienanfaenger.csv not found — run load_studienanfaenger.py (load_studentflow) first. Flow panel will be skipped.")
    except Exception as e:
        print(f"  Studentflow load error: {e}")

    # Visual (DASHBOARD) using Plotly
    fig = make_subplots(
        rows=6, cols=2,
        shared_xaxes=False,
        vertical_spacing=0.06,
        horizontal_spacing=0.08,
        subplot_titles=(
            "POLITICAL STRESS INDEX (PSI) v4 - Germany",
            "Mass Mobilization Factors (M)", "Elite Pressure (E)",
            "Elite Pipeline Flow — Studentflow (Annual Inflow)",
            "Macroeconomic Trends", "Demographic Context (Youth Bulge)",
            "State Institutional Stability (S)", None,
            "PSI FORECAST 2026–2027 (Instability Window)"
        ),
        specs=[[{"colspan": 2}, None],
               [{}, {}],
               [{"colspan": 2}, None],
               [{}, {}],
               [{"colspan": 2}, None],
               [{"colspan": 2}, None]],
        row_heights=[0.20, 0.16, 0.14, 0.16, 0.14, 0.20]
    )
    
    # 1. PSI
    fig.add_trace(go.Scatter(
        x=idx, y=v2['psi_v4'],
        name='PSI v4',
        line=dict(color='firebrick', width=4),
        fill='tozeroy', fillcolor='rgba(178, 34, 34, 0.1)'
    ), row=1, col=1)

    # 2. M (Mass Mobilization)
    fig.add_trace(go.Scatter(
        x=idx, y=v2['m_econ'], name='Relative Deprivation', line=dict(color='#FF4B2B')
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=idx, y=v2['food_pump'], name='Food Pump (Inflation Diff)', line=dict(color='#FFD200')
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=idx, y=nm(v2['strike_days']), name='Strike Intensity (WSI)', line=dict(color='#FF6B6B', dash='dot', width=2)
    ), row=2, col=1)

    # 3. E (Elite Pressure)
    fig.add_trace(go.Scatter(
        x=idx, y=nm(v2['wealth_pump']), name='Wealth Pump Index', line=dict(color='#9D50BB')
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=idx, y=v2['elite_pressure'], name='Elite Pressure (with Holders)', line=dict(color='royalblue', width=2)
    ), row=2, col=2)
    fig.add_trace(go.Scatter(
        x=idx, y=v2['frustrated_fraction'], name='Frustrated Fraction', line=dict(color='crimson', dash='dash')
    ), row=2, col=2)

    # 3. Elite Pipeline Flow — Studentflow
    annual_graduates_est = v2['elite_candidates'] * (1 - 0.30) / 5
    annual_openings_est  = v2['holders'] * 0.05
    fig.add_trace(go.Scatter(
        x=idx, y=annual_graduates_est,
        name='Est. Annual Graduates (enrolled×0.7/5)',
        line=dict(color='steelblue', width=2)
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=idx, y=annual_openings_est,
        name='Annual Elite Openings (holders×5%)',
        line=dict(color='tomato', width=2, dash='dash')
    ), row=3, col=1)
    if studentflow_ser is not None:
        fig.add_trace(go.Bar(
            x=studentflow_ser.index, y=studentflow_ser.values,
            name='Studentflow (new entrants/yr)',
            marker_color='rgba(65,105,225,0.45)',
            width=1000*60*60*24*300  # ~10 months wide bars
        ), row=3, col=1)
        # Estimated graduates from Studentflow: entrants × (1 - dropout)
        sf_graduates = studentflow_ser * (1 - 0.30)
        fig.add_trace(go.Scatter(
            x=sf_graduates.index, y=sf_graduates.values,
            name='Est. Graduates from Entrants (×0.7)',
            line=dict(color='royalblue', width=2, dash='dot')
        ), row=3, col=1)

    # 4. GDP
    fig.add_trace(go.Scatter(
        x=idx, y=v2['gdp_growth'], name='Real GDP Growth', line=dict(color='seagreen')
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=idx, y=v2['wage_real'].pct_change(12)*100, name='Real Wage Growth', line=dict(color='orange')
    ), row=4, col=1)

    # 5. Youth
    fig.add_trace(go.Scatter(
        x=idx, y=v2['youth_bulge'], name='Youth Bulge Index', line=dict(color='goldenrod'),
        fill='tozeroy', fillcolor='rgba(218, 165, 32, 0.1)'
    ), row=4, col=2)

    # 6. S
    fig.add_trace(go.Scatter(
        x=idx, y=v2['s_capacity'], name='Extraction Stability', line=dict(color='slateblue', width=3),
        fill='tozeroy', fillcolor='rgba(106, 90, 205, 0.1)'
    ), row=5, col=1)

    # 7. FORECAST panel
    fig.add_trace(go.Scatter(
        x=idx, y=v2['psi_v4'],
        name='PSI v4 (Historical)',
        line=dict(color='firebrick', width=3),
        showlegend=False
    ), row=6, col=1)
    fig.add_trace(go.Scatter(
        x=fc_series.index, y=fc_series.values,
        name='PSI Forecast',
        line=dict(color='#FF6347', width=3, dash='dash')
    ), row=6, col=1)
    fig.add_trace(go.Scatter(
        x=fc_upper.index, y=fc_upper.values,
        name='95% CI Upper',
        line=dict(width=0),
        showlegend=False
    ), row=6, col=1)
    fig.add_trace(go.Scatter(
        x=fc_lower.index, y=fc_lower.values,
        name='95% Confidence Interval',
        line=dict(width=0),
        fill='tonexty',
        fillcolor='rgba(255, 99, 71, 0.15)'
    ), row=6, col=1)
    fig.add_vrect(
        x0='2026-06-01', x1='2027-06-01',
        fillcolor='rgba(255, 0, 0, 0.08)',
        line=dict(color='red', width=1, dash='dot'),
        annotation_text='INSTABILITY WINDOW',
        annotation_position='top left',
        annotation_font=dict(size=14, color='red', family='Arial Black'),
        row=6, col=1
    )
    fig.add_vline(
        x=psi_hist.index[-1], line_width=2, line_dash='solid',
        line_color='gray', row=6, col=1
    )
    fig.add_annotation(
        x=psi_hist.index[-1], y=psi_hist.iloc[-1],
        text='NOW', showarrow=True, arrowhead=2,
        font=dict(size=12, color='gray'),
        row=6, col=1
    )

    # Layout
    fig.update_layout(
        template='plotly_white',
        height=2100,
        hovermode="x unified",
        showlegend=True,
        title_text="Dashboard: PSI v4 & Structural-Demographic Drivers — Germany",
        title_font_size=24
    )

    fig.update_yaxes(range=[0.8, 1.1], row=5, col=1)
    fig.update_xaxes(title_text='', row=6, col=1)
    fig.update_yaxes(title_text='PSI', row=6, col=1)
    fig.update_yaxes(title_text='Persons/yr', row=3, col=1)
    
    html_out = OUTPUT_DIR / 'psi_v4_dashboard.html'
    fig.write_html(html_out)
    print(f"DONE: Visualized Dashboard securely in {html_out}")
    fig.show()

if __name__ == "__main__":
    main()
