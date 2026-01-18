import pandas as pd
import numpy as np

def fix_encoding(text):
    if isinstance(text, str):
        return text.replace('Ã¤', 'ä').replace('Ã¼', 'ü').replace('Ã¶', 'ö').replace('ÃŸ', 'ß')
    return text

def process_students(file_path):
    # Загружаем данные, пропуская метаданные (обычно первые строки до заголовка)
    df = pd.read_csv(file_path, sep=';', encoding='utf-8', on_bad_lines='skip')
    
    # Фильтруем, чтобы избежать дублирования (берем только "Insgesamt")
    # В твоем CSV это пустые значения в атрибутах или слово 'Insgesamt'
    df = df[
        (df['2_variable_attribute_label'].str.contains('Insgesamt', na=True)) & 
        (df['3_variable_attribute_label'].str.contains('Insgesamt', na=True))
    ].copy()

    # Чистим значения
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df = df.dropna(subset=['value'])

    # Парсим дату: 2022-10P6M -> 2022-10-01
    def parse_destatis_date(date_str):
        if not isinstance(date_str, str): return None
        year = date_str[:4]
        month = '10' if '-10P6M' in date_str else '04'
        return pd.to_datetime(f"{year}-{month}-01")

    df['date'] = df['time'].apply(parse_destatis_date)
    df = df.dropna(subset=['date'])

    # Группируем по дате (суммируем все выбранные нами специальности SF)
    df_daily = df.groupby('date')['value'].sum().reset_index()
    df_daily = df_daily.rename(columns={'value': 'elite_candidates'})
    
    return df_daily

# 1. Загружаем старый мастер-файл (или создаем из прошлых шагов)
try:
    master_df = pd.read_csv('master_cliodynamics.csv')
    master_df['date'] = pd.to_datetime(master_df['date'])
except FileNotFoundError:
    print("Ошибка: Сначала запусти прошлый скрипт для создания master_cliodynamics.csv")
    exit()

# 2. Обрабатываем студентов
df_students = process_students('data_students.csv')

# 3. Мерджим данные
# Используем outer, чтобы не потерять исторические данные по студентам, если их больше, чем по ЗП
final_df = pd.merge(master_df, df_students, on='date', how='left')
final_df = final_df.sort_values('date')

# 4. Интерполяция (Подводный камень: студенты обновляются раз в полгода)
# Мы заполняем пропуски между октябрем и апрелем линейно
final_df['elite_candidates'] = final_df['elite_candidates'].interpolate(method='linear')

# 5. Расчет Индекса Политического Стресса (PSI) - упрощенная модель
# PSI = (Wealth Pump * Elite Candidates) / 1000 (для масштаба)
final_df['psi_index'] = (final_df['wealth_pump'] * final_df['elite_candidates']) / 1000

# Сохраняем результат
final_df.to_csv('master_cliodynamics_v2.csv', index=False)

print("Финальный файл master_cliodynamics_v2.csv готов.")
print(final_df[['date', 'wealth_pump', 'elite_candidates', 'psi_index']].tail(10))