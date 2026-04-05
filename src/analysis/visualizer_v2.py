import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'

df_full = pd.read_csv(DATA_PROCESSED / 'master_cliodynamics_v2.csv')
df = df_full.dropna(subset=['psi_index'])  # Только для PSI

# Отдельные данные для студентов (с 2015)
df_elite = df_full.dropna(subset=['elite_candidates'])

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=False,
    vertical_spacing=0.08,
    subplot_titles=(
        "Индекс Политического Стресса (PSI) - Германия", 
        "Кандидаты в элиту (Студенты)",
        "Насос богатства (Аренда/ЗП)"
    ),
    row_heights=[0.5, 0.25, 0.25]
)

# 1. Главный график - PSI
fig.add_trace(go.Scatter(
    x=df['date'], 
    y=df['psi_index'],
    name='PSI (Political Stress Index)',
    line=dict(color='firebrick', width=4),
    fill='tozeroy',
    fillcolor='rgba(178, 34, 34, 0.1)'
), row=1, col=1)

# Анотации для важных точек
max_psi_date = df.loc[df['psi_index'].idxmax()]['date']
max_psi_val = df['psi_index'].max()

fig.add_annotation(
    x=max_psi_date, y=max_psi_val,
    text="Пик стресса (Шок 2022)",
    showarrow=True, arrowhead=1, row=1, col=1
)

# 2. Кандидаты в элиту (полные данные с 2015)
fig.add_trace(go.Scatter(
    x=df_elite['date'], 
    y=df_elite['elite_candidates'],
    name='Кандидаты в элиту (Студенты)',
    line=dict(color='royalblue', width=2),
    fill='tozeroy',
    fillcolor='rgba(65, 105, 225, 0.1)'
), row=2, col=1)

# 3. Насос богатства
fig.add_trace(go.Scatter(
    x=df['date'], 
    y=df['wealth_pump'],
    name='Насос богатства (Аренда/ЗП)',
    line=dict(color='orange', width=2),
    fill='tozeroy',
    fillcolor='rgba(255, 165, 0, 0.1)'
), row=3, col=1)


# Настройка макета
fig.update_layout(
    template='plotly_white',
    height=900,
    hovermode="x unified",
    showlegend=True
)

fig.update_yaxes(title_text="Уровень стресса", row=1, col=1)

# Zoom Y-ось для студентов чтобы видеть изменения
elite_min = df_elite['elite_candidates'].min()
elite_max = df_elite['elite_candidates'].max()
elite_padding = (elite_max - elite_min) * 0.1
fig.update_yaxes(
    title_text="Студенты", 
    range=[elite_min - elite_padding, elite_max + elite_padding],
    row=2, col=1
)

fig.update_yaxes(title_text="Аренда/ЗП", row=3, col=1)

# Убираем shared x-axis чтобы каждый график имел свой диапазон дат
fig.update_xaxes(row=2, col=1)

fig.show()