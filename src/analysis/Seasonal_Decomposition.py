import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'

df = pd.read_csv(DATA_PROCESSED / 'master_cliodynamics.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

analysis = seasonal_decompose(df['wage_nominal'], model='additive', period=12)

fig = make_subplots(rows=4, cols=1, 
                    subplot_titles=("Original (Номинальная ЗП)", "Trend (Чистый тренд)", 
                                    "Seasonal (Эффект премий)", "Residual (Шум/Аномалии)"))

fig.add_trace(go.Scatter(x=df.index, y=analysis.observed, name='Observed'), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=analysis.trend, name='Trend'), row=2, col=1)
fig.add_trace(go.Scatter(x=df.index, y=analysis.seasonal, name='Seasonal'), row=3, col=1)
fig.add_trace(go.Scatter(x=df.index, y=analysis.resid, name='Residual'), row=4, col=1)

fig.update_layout(height=900, title_text="Сезонная декомпозиция зарплат в ФРГ", showlegend=False)
fig.show()