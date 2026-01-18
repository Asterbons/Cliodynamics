import pandas as pd
import numpy as np

def fix_encoding(text):
    """Исправляет кривую кодировку UTF-8"""
    if isinstance(text, str):
        return text.replace('Ã¤', 'ä').replace('Ã¼', 'ü').replace('Ã¶', 'ö').replace('ÃŸ', 'ß')
    return text

def parse_destatis_date(date_str):
    """
    Парсит формат Destatis: 2022-10P6M -> 2022-10-01
    10P6M = октябрь (зимний семестр)
    04P6M = апрель (летний семестр)
    """
    if not isinstance(date_str, str): 
        return None
    try:
        year = date_str[:4]
        # Определяем месяц по паттерну
        if '-10' in date_str or 'WS' in date_str:  # Wintersemester
            month = '10'
        elif '-04' in date_str or 'SS' in date_str:  # Sommersemester
            month = '04'
        else:
            # Fallback: пытаемся извлечь месяц из строки
            month = date_str[5:7] if len(date_str) > 6 else '01'
        return pd.to_datetime(f"{year}-{month}-01")
    except Exception as e:
        print(f"Ошибка парсинга даты '{date_str}': {e}")
        return None

def process_students(file_path):
    """
    Обрабатывает CSV от GENESIS Destatis со студентами
    
    Структура ожидаемых данных:
    - statistics_code: код таблицы (21311)
    - time: период (2022-10P6M)
    - 1_variable: специальность (STAF01)
    - 2_variable: обычно пол (DINSG)
    - 3_variable: обычно национальность (NATS)
    - 4_variable: тип вуза или другое измерение
    - value: количество студентов
    """
    
    # 1. ЧИТАЕМ CSV
    print(f"Читаем файл {file_path}...")
    
    # Сначала читаем первую строку чтобы понять структуру
    with open(file_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
    
    # Проверяем, разделены ли колонки точкой с запятой
    if ';' in first_line:
        df = pd.read_csv(file_path, sep=';', encoding='utf-8', low_memory=False)
    else:
        print("ОШИБКА: CSV не содержит разделителя ';'")
        print(f"Первая строка: {first_line[:200]}")
        return None
    
    print(f"Загружено {len(df)} строк, {len(df.columns)} колонок")
    print(f"Колонки: {list(df.columns)[:10]}")  # Показываем первые 10
    
    # 2. ПРОВЕРЯЕМ НАЛИЧИЕ НЕОБХОДИМЫХ КОЛОНОК
    required_cols = ['time', 'value']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"ОШИБКА: Отсутствуют обязательные колонки: {missing}")
        return None
    
    # 3. ОПРЕДЕЛЯЕМ КОЛОНКИ С АТРИБУТАМИ (пол, национальность и т.д.)
    # Ищем все колонки вида N_variable_attribute_label
    attr_cols = [col for col in df.columns if '_variable_attribute_label' in col]
    print(f"Найдены колонки атрибутов: {attr_cols}")
    
    # 4. ФИЛЬТРАЦИЯ: оставляем только итоговые значения (Insgesamt)
    # Это избегает двойного подсчета по полу, национальности и т.д.
    df_filtered = df.copy().reset_index(drop=True)
    
    for col in attr_cols:
        # Проверяем наличие данных в колонке
        if df_filtered[col].notna().any():
            # Проверяем, есть ли вообще "Insgesamt" в этой колонке
            has_insgesamt = df_filtered[col].str.contains('Insgesamt', na=False, case=False).any()
            
            if has_insgesamt:
                # Оставляем только строки с 'Insgesamt' или пустые (они тоже означают итог)
                prev_len = len(df_filtered)
                mask = (df_filtered[col].isna()) | (df_filtered[col].str.contains('Insgesamt', na=False, case=False))
                df_filtered = df_filtered[mask].reset_index(drop=True)
                print(f"  Фильтр по {col}: {prev_len} -> {len(df_filtered)} строк")
            else:
                print(f"  Пропуск {col}: 'Insgesamt' отсутствует в данных")
    
    # 5. ОЧИСТКА ЗНАЧЕНИЙ
    # Конвертируем в числа, убираем пропуски
    df_filtered['value'] = pd.to_numeric(df_filtered['value'], errors='coerce')
    initial_len = len(df_filtered)
    df_filtered = df_filtered.dropna(subset=['value'])
    print(f"Удалено {initial_len - len(df_filtered)} строк с невалидными значениями")
    
    # 6. ПАРСИНГ ДАТ
    df_filtered['date'] = df_filtered['time'].apply(parse_destatis_date)
    initial_len = len(df_filtered)
    df_filtered = df_filtered.dropna(subset=['date'])
    print(f"Удалено {initial_len - len(df_filtered)} строк с невалидными датами")
    
    if len(df_filtered) == 0:
        print("ОШИБКА: После фильтрации не осталось данных!")
        return None
    
    # 7. ГРУППИРОВКА ПО ДАТЕ
    # Суммируем всех студентов по элитным специальностям на каждую дату
    df_daily = df_filtered.groupby('date')['value'].sum().reset_index()
    df_daily = df_daily.rename(columns={'value': 'elite_candidates'})
    df_daily = df_daily.sort_values('date')
    
    print(f"\nИтого дат: {len(df_daily)}")
    print(f"Период: {df_daily['date'].min()} - {df_daily['date'].max()}")
    print(f"Всего студентов (среднее): {df_daily['elite_candidates'].mean():.0f}")
    print(f"\nПример данных:")
    print(df_daily.head())
    
    return df_daily


def main():
    """Основная функция обработки"""
    
    # 1. Загружаем мастер-файл
    try:
        master_df = pd.read_csv('master_cliodynamics.csv')
        master_df['date'] = pd.to_datetime(master_df['date'])
        print(f"\nЗагружен master_cliodynamics.csv: {len(master_df)} строк")
    except FileNotFoundError:
        print("ОШИБКА: Файл master_cliodynamics.csv не найден!")
        print("Сначала создайте базовый файл с данными по зарплатам.")
        return
    
    # 2. Обрабатываем студентов
    df_students = process_students('data_students_test.csv')
    
    if df_students is None:
        print("Обработка студентов не удалась!")
        return
    
    # 3. Объединяем данные
    print("\nОбъединение с мастер-файлом...")
    final_df = pd.merge(master_df, df_students, on='date', how='outer')
    final_df = final_df.sort_values('date')
    
    # 4. Интерполяция
    # Студенты обновляются 2 раза в год (апрель, октябрь)
    # Заполняем промежуточные месяцы линейной интерполяцией
    print("Интерполяция данных по студентам...")
    final_df['elite_candidates'] = final_df['elite_candidates'].interpolate(
        method='linear', 
        limit_direction='both'
    )
    
    # 5. Расчет PSI (Political Stress Index)
    # PSI = (Wealth Pump * Elite Candidates) / масштаб
    # Wealth Pump должен быть уже в master файле
    if 'wealth_pump' in final_df.columns:
        final_df['psi_index'] = (
            final_df['wealth_pump'] * final_df['elite_candidates']
        ) / 1000
        print("PSI индекс рассчитан")
    else:
        print("ВНИМАНИЕ: wealth_pump не найден в мастер-файле, PSI не рассчитан")
    
    # 6. Сохранение
    output_file = 'master_cliodynamics_v2.csv'
    final_df.to_csv(output_file, index=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Файл {output_file} успешно создан!")
    print(f"{'='*60}")
    print(f"\nПоследние 10 записей:")
    cols_to_show = ['date', 'elite_candidates']
    if 'wealth_pump' in final_df.columns:
        cols_to_show.append('wealth_pump')
    if 'psi_index' in final_df.columns:
        cols_to_show.append('psi_index')
    
    print(final_df[cols_to_show].tail(10).to_string(index=False))
    
    # Статистика
    print(f"\nСтатистика:")
    print(f"  Период: {final_df['date'].min()} - {final_df['date'].max()}")
    print(f"  Всего точек: {len(final_df)}")
    print(f"  Среднее число элитных кандидатов: {final_df['elite_candidates'].mean():.0f}")


if __name__ == "__main__":
    main()
