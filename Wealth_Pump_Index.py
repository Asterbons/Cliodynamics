import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# 1. Load data
df = pd.read_csv('master_cliodynamics.csv')
df['date'] = pd.to_datetime(df['date'])
df.sort_values('date', inplace=True)
df.set_index('date', inplace=True)

# 2. Perform Seasonal Decomposition on Nominal Wages
# Using additive model as bonuses are roughly absolute additions
result = seasonal_decompose(df['wage_nominal'], model='additive', period=12)

# 3. Visualization of the components
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

result.observed.plot(ax=ax1, title='Original Nominal Wage Index (with November spikes)', color='blue')
result.trend.plot(ax=ax2, title='Trend Component (The "Clean" underlying wage growth)', color='green')
result.seasonal.plot(ax=ax3, title='Seasonal Component (The Weihnachtsgeld effect)', color='orange')
result.resid.plot(ax=ax4, title='Residuals (Random noise / Shocks)', color='red')

plt.tight_layout()
plt.savefig('seasonal_decomposition.png')

# 4. Calculate "Smooth Wealth Pump"
# We use the Trend component to calculate a pump index that isn't distorted by bonuses
df['wage_trend'] = result.trend
# We can fill NaNs at the edges with original values or just drop them for the final plot
df['wealth_pump_trend'] = (df['rent'] / df['wage_trend']) * 100

# 5. Final Comparison Plot: Raw vs Trend Wealth Pump
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['wealth_pump'], label='Raw Wealth Pump (Seasonal)', alpha=0.3, color='grey')
plt.plot(df.index, df['wealth_pump_trend'], label='Structural Wealth Pump (Trend)', color='red', linewidth=2)
plt.axhline(100, color='black', linestyle='--')
plt.title('Structural Wealth Pump in Germany (Seasonally Adjusted)')
plt.ylabel('Index (2020=100)')
plt.legend()
plt.grid(True)
plt.savefig('wealth_pump_trend.png')

print("Decomposition complete. Files saved.")