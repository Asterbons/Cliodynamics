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

    # 1. Youth
    y_ser = y_raw[y_raw['2_variable_attribute_code'].str.contains('ALT01[5-9]|ALT02[0-4]', na=False)].groupby('time')['value'].sum()
    y_df = pd.DataFrame({'date': [pd.to_datetime(str(t)[:4]+'-01-01') for t in y_ser.index], 'y_val': y_ser.values}).sort_values('date')
    v2 = pd.merge_asof(v2, y_df, on='date', direction='backward')
    v2['youth_bulge'] = v2['y_val'].interpolate().fillna(method='bfill')
    v2['youth_bulge'] /= v2['youth_bulge'].mean()

    # 2. Indicators
    v2 = v2.set_index('date')
    idx = v2.index
    
    # Extract points using robust search
    c_ser = get_pts(c_raw[c_raw.stack().astype(str).str.contains('Verbraucherpreisindex', na=False, case=False).unstack().any(axis=1)])
    f_ser = get_pts(f_raw[f_raw['3_variable_attribute_code'] == 'CC13-01'])
    g_ser = get_pts(g_raw[g_raw['value_variable_code'] == 'VGR014'])
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
    v2['s_capacity'] = 1.0 / (1.0 + t_mon.pct_change(12).rolling(12).std().fillna(0))

    # 4. PSI Total
    v2['psi_v4_raw'] = (nm(v2['wealth_pump']) * nm(v2['elite_candidates']) * 
                        nm(v2['m_econ']) * nm(v2['food_pump']) * nm(v2['youth_bulge'])) / v2['s_capacity'].fillna(1.0)
    v2['psi_v4'] = v2['psi_v4_raw'].rolling(12, center=True, min_periods=1).mean()
    
    v2.to_csv(DATA_PROCESSED / 'master_cliodynamics_v4.csv')
    print("DONE. Processed v4.")

if __name__ == "__main__":
    process_v4()
