import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'

MON = {'Januar':1, 'Februar':2, 'März':3, 'April':4, 'Mai':5, 'Juni':6,
       'Juli':7, 'August':8, 'September':9, 'Oktober':10, 'November':11, 'Dezember':12}

def get_pts(df):
    if df.empty: return pd.Series()
    # Find DG or Deutschland
    mask = df.stack().astype(str).str.contains('DG|Deutschland insgesamt', na=False).unstack().any(axis=1)
    df = df[mask] if mask.any() else df
    
    # Month
    def find_m(r):
        for val in r:
            if val in MON: return MON[val]
        return 1
    
    df['m_idx'] = df.apply(find_m, axis=1)
    df['dt'] = pd.to_datetime(df['time'].astype(str).str[:4] + '-' + df['m_idx'].astype(str) + '-01')
    df['val'] = pd.to_numeric(df['value'], errors='coerce')
    return df.dropna(subset=['val']).groupby('dt')['val'].first().sort_index()

def nm(s):
    if s is None or s.empty: return pd.Series([0.5])
    v = s.replace([np.inf, -np.inf], np.nan).fillna(s.median() if s.notna().any() else 0.5)
    mi, ma = v.min(), v.max()
    return (v - mi) / (ma - mi + 1e-9) + 0.1 if mi != ma else v*0 + 0.5

def process_v4():
    print("PSI v4 Pipeline (V2 -> V4 Final Stabilization)...")
    v2 = pd.read_csv(DATA_PROCESSED / 'master_cliodynamics_v2.csv', parse_dates=['date'])
    v2 = v2.sort_values('date').reset_index(drop=True)

    # Raw
    y_raw = pd.read_csv(DATA_RAW / 'data_youth_annual.csv', sep=';', decimal=',', comment=';')
    c_raw = pd.read_csv(DATA_RAW / 'data_cpi_general.csv', sep=';', decimal=',', comment=';')
    f_raw = pd.read_csv(DATA_RAW / 'data_food_prices.csv', sep=';', decimal=',', comment=';')
    g_raw = pd.read_csv(DATA_RAW / 'data_gdp_quarterly.csv', sep=';', decimal=',', comment=';')
    t_raw = pd.read_csv(DATA_RAW / 'data_tax_revenue.csv', sep=';', decimal=',', comment=';')
    cs_raw = pd.read_csv(DATA_RAW / 'data_civil_servants.csv', sep=';', decimal=',')
    strikes_raw = pd.read_csv(DATA_RAW / 'data_strikes_wsi.csv')
    holders_manual = pd.read_csv(DATA_RAW / 'data_holders.csv')
    try:
        holders_auto = pd.read_csv(DATA_RAW / 'data_holders_raw.csv', sep=';', decimal=',')
    except:
        holders_auto = pd.DataFrame()

    # 1. Youth (Relative Share approach)
    y_youth = y_raw[y_raw['2_variable_attribute_code'].str.contains('ALT01[5-9]|ALT02[0-4]', na=False)].groupby('time')['value'].sum()
    y_total = y_raw[y_raw['2_variable_attribute_label'] == 'Insgesamt'].groupby('time')['value'].sum()
    y_share = y_youth / y_total
    
    y_df = pd.DataFrame({
        'date': [pd.to_datetime(str(t)[:4]+'-01-01') for t in y_share.index], 
        'y_val': y_share.values
    }).sort_values('date')
    
    v2 = pd.merge_asof(v2, y_df, on='date', direction='backward')
    v2['youth_bulge'] = v2['y_val'].interpolate().fillna(method='bfill')
    # Normalize by mean to keep PSI scale consistent
    v2['youth_bulge'] /= v2['youth_bulge'].mean()

    # 2. Indicators
    v2 = v2.set_index('date')
    idx = v2.index
    
    # Extract food CPI (CC13-01) from food prices file
    f_ser = get_pts(f_raw[f_raw['3_variable_attribute_code'] == 'CC13-01'])

    # Compute overall CPI as mean of all 12 CC13 categories from food prices file
    # (data_cpi_general.csv has German decimal strings that fail pd.to_numeric)
    def build_overall_cpi(df):
        df = df.copy()
        mask = df.stack().astype(str).str.contains('DG|Deutschland insgesamt', na=False).unstack().any(axis=1)
        df = df[mask] if mask.any() else df
        def find_m(r):
            for val in r:
                if val in MON: return MON[val]
            return 1
        df['m_idx'] = df.apply(find_m, axis=1)
        df['dt'] = pd.to_datetime(df['time'].astype(str).str[:4] + '-' + df['m_idx'].astype(str) + '-01')
        df['val'] = pd.to_numeric(df['value'], errors='coerce')
        return df.groupby('dt')['val'].mean().sort_index()
    c_ser = build_overall_cpi(f_raw)
    g_ser = get_pts(g_raw[(g_raw['value_variable_code'] == 'VGR014') & (g_raw['2_variable_attribute_code'] == 'VGRPVK')])
    t_ser = get_pts(t_raw)

    # Always initialize columns
    v2['gdp_growth'] = 0.0
    v2['food_pump'] = 0.0
    
    if not g_ser.empty:
        v2['gdp_growth'] = g_ser.reindex(idx).interpolate(method='time').fillna(method='bfill').pct_change(12)*100
    
    if not c_ser.empty and not f_ser.empty:
        v2['food_pump'] = (f_ser.pct_change(12)*100).reindex(idx).interpolate() - (c_ser.pct_change(12)*100).reindex(idx).interpolate()

    # 3. Stress Logic
    v2['m_econ'] = v2['gdp_growth'].fillna(0) - v2['wage_real'].pct_change(12).fillna(0)*100
    t_mon = t_ser.reindex(idx).interpolate() if not t_ser.empty else v2['gdp_growth']
    s_tax = 1.0 / (1.0 + t_mon.pct_change(12).rolling(12).std().fillna(0))

    # Civil servants component: more staff = higher capacity
    cs_total = cs_raw[cs_raw['2_variable_attribute_label'].str.contains('Insgesamt', na=False)]
    cs_df = pd.DataFrame({
        'date': pd.to_datetime(cs_total['time'].astype(str).str[:4] + '-06-01'),
        'cs_val': pd.to_numeric(cs_total['value'], errors='coerce')
    }).dropna().sort_values('date').set_index('date')
    cs_monthly = cs_df['cs_val'].reindex(idx).interpolate(method='time').fillna(method='bfill').fillna(method='ffill')
    v2['civil_servants'] = cs_monthly
    # Normalize civil servants to [0.5, 1.0] — higher headcount = higher capacity
    cs_norm = (cs_monthly - cs_monthly.min()) / (cs_monthly.max() - cs_monthly.min() + 1e-9) * 0.5 + 0.5
    # Composite state capacity = tax stability * civil servant capacity
    v2['s_capacity'] = s_tax * cs_norm

    # 5. Strike data (Mass Mobilization) — annual values, same value for all months in a year
    strikes_annual = strikes_raw.set_index('year')['strike_days_k'].astype(float)
    v2['strike_days'] = idx.year.map(strikes_annual)

    # 6. Holders modeling (Elite Overproduction Pressure)
    # Base: historical manual data
    h_df = pd.DataFrame({
        'date': pd.to_datetime(holders_manual['year'].astype(str) + '-01-01'),
        'holders': holders_manual['holders_k'].astype(float) * 1000
    }).set_index('date')

    # Patch with automated data if available (Elite positions proxy: Law/Admin + Econ Sci)
    if not holders_auto.empty:
        # Correct mapping based on KldB 2010 groups:
        # KB10-73: Berufe in Recht und Verwaltung
        # KB10-91: Geistes-, Gesellschafts-, Wirtschaftswissensch.
        target_codes = ['KB10-73', 'KB10-91']
        
        ha = holders_auto[
            (holders_auto['4_variable_attribute_code'].isin(target_codes)) &
            (holders_auto['2_variable_attribute_label'] == 'Insgesamt') &
            (holders_auto['3_variable_attribute_label'] == 'Insgesamt')
        ]
        
        if not ha.empty:
            # Group by year and sum values for our target elite groups
            # Ensure numeric conversion to avoid string concatenation bug
            ha['val'] = pd.to_numeric(ha['value'], errors='coerce').fillna(0)
            ha_grouped = ha.groupby('time')['val'].sum().reset_index()
            ha_df = pd.DataFrame({
                'date': pd.to_datetime(ha_grouped['time'].astype(str) + '-01-01'),
                'holders': ha_grouped['val'].astype(float) * 1000
            }).set_index('date').dropna()
            
            # Update/Merge: prefer automated for overlapping years
            h_df = ha_df.combine_first(h_df)
            print(f"  Holders Auto-Patched: {len(ha_df)} points (latest val: {ha_df['holders'].iloc[-1]/1000:.1f}k)")

    holders_monthly = h_df['holders'].reindex(idx).interpolate(method='time').fillna(method='bfill').fillna(method='ffill')
    v2['holders'] = holders_monthly

    # frustrated_fraction: annual graduates vs annual elite position openings
    # Graduates = enrolled × (1 - dropout_rate) / avg_study_years
    # Openings   = total holders × annual turnover rate
    DROPOUT_RATE = 0.30   # 30% of enrolled students never graduate
    AVG_STUDY_YEARS = 5   # avg years to degree (Bachelor + Master in Germany)
    TURNOVER_RATE = 0.05  # ~5% annual turnover of elite positions
    annual_graduates = v2['elite_candidates'] * (1 - DROPOUT_RATE) / AVG_STUDY_YEARS
    annual_openings = holders_monthly * TURNOVER_RATE
    v2['frustrated_fraction'] = np.clip(
        (annual_graduates - annual_openings) / annual_graduates.replace(0, np.nan),
        0, 1
    ).fillna(0)
    # elite_pressure replaces raw elite_candidates in PSI
    # Even when frustrated_fraction is 0, base pressure from normalized elite count persists
    v2['elite_pressure'] = nm(v2['elite_candidates']) * (1 + v2['frustrated_fraction'])

    # 7. PSI Total (uses elite_pressure instead of raw elite_candidates)
    v2['psi_v4_raw'] = (nm(v2['wealth_pump']) * v2['elite_pressure'] * 
                        nm(v2['m_econ']) * nm(v2['food_pump']) * nm(v2['youth_bulge']) *
                        nm(v2['strike_days'])) / v2['s_capacity'].fillna(1.0)
    v2['psi_v4'] = v2['psi_v4_raw'].rolling(12, center=True, min_periods=1).mean()
    
    v2.to_csv(DATA_PROCESSED / 'master_cliodynamics_v4.csv')
    print("DONE. Processed v4.")

if __name__ == "__main__":
    process_v4()
