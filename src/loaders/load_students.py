import os
import sys
import requests
import io
import zipfile
import csv
from pathlib import Path
from dotenv import load_dotenv

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'

load_dotenv(PROJECT_ROOT / '.env')

token = os.getenv('DESTATIS_TOKEN')
base_url = "https://www-genesis.destatis.de/genesisWS/rest/2020/"

headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
    'username': token,
    'password': '' 
}

# Специальности претендующие на элитный статус: Jura, Politik, BWL/VWL/Wirtschaft, Soziologie, Philosophie, Geschichte, Medien
subject_codes = {
    'SF135': 'Rechtswissenschaft',           # Jura
    'SF129': 'Politikwissenschaft/Politologie',  # Politik
    'SF021': 'Betriebswirtschaftslehre',     # BWL
    'SF175': 'Volkswirtschaftslehre',        # VWL
    'SF182': 'Internationale Betriebswirtschaft/Management',  # Wirtschaft
    'SF184': 'Wirtschaftswissenschaften',    # Wirtschaft
    'SF149': 'Soziologie',
    'SF127': 'Philosophie',
    'SF068': 'Geschichte',
    'SF272': 'Alte Geschichte',
    'SF275': 'Wissenschaftsgeschichte/Technikgeschichte',
    'SF302': 'Medienwissenschaft',
    'SF303': 'Kommunikationswissenschaft/Publizistik',
}

# Для поиска кода специальности в ffcsv формате
def find_subject_code_position(header_row):
    """Находит позицию колонки с кодом специальности в заголовке"""
    for idx, field in enumerate(header_row):
        # Ищем колонку, которая может содержать код специальности
        # Обычно это что-то вроде "STAF01", "Studienfach", и т.д.
        if 'STAF' in field.upper() or 'FACH' in field.upper():
            return idx
    return None

def download_and_unzip(url, headers, payload):
    """Скачивает и распаковывает файл из API"""
    try:
        response = requests.post(url + 'data/tablefile', headers=headers, data=payload)
        if response.status_code == 200:
            if response.content.startswith(b'Error'):
                print(f"Ошибка в ответе сервера: {response.text[:100]}")
                return None
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                file_name = z.namelist()[0]
                with z.open(file_name) as f: 
                    return f.read().decode('utf-8')
        else:
            print(f"HTTP Error {response.status_code}")
            return None
    except Exception as e:
        print(f"Ошибка запроса: {e}")
        return None

all_data_rows = []
header_saved = False

print(f"Запуск теста для {len(subject_codes)} специальностей...")

# Скачиваем полный датасет один раз
print("Скачивание полного датасета...", end=" ", flush=True)

payload = {
    'name': '21311-0003',
    'startyear': '2015',
    'compress': 'true',
    'format': 'ffcsv',
    'language': 'de'
}

csv_text = download_and_unzip(base_url, headers, payload)

if csv_text:
    print("OK")
    lines = csv_text.splitlines()
    
    # Находим заголовок (первая строка, не начинающаяся с '21311')
    header_line = None
    data_start_idx = 0
    for idx, line in enumerate(lines):
        if line.startswith('21311'):
            data_start_idx = idx
            if idx > 0:
                header_line = lines[idx - 1]
            break
    
    if header_line:
        all_data_rows.append(header_line)
        print(f"Заголовок найден: {header_line[:100]}...")
    
    # Фильтруем данные по нужным кодам специальностей
    print(f"\nФильтрация по специальностям:")
    for code, name in subject_codes.items():
        # Ищем строки, содержащие код специальности
        # В формате ffcsv коды обычно разделены точкой с запятой
        matching_rows = [line for line in lines[data_start_idx:] 
                        if code in line]
        
        all_data_rows.extend(matching_rows)
        print(f"  {name} ({code}): {len(matching_rows)} строк")
    
    # Сохраняем результат
    if all_data_rows:
        with open(DATA_RAW / "data_students.csv", "w", encoding="utf-8") as f:
            f.write("\n".join(all_data_rows))
        print(f"\nФайл data_students.csv создан. Всего строк: {len(all_data_rows)}")
        print(f"(включая заголовок: {len(all_data_rows)-1} строк данных)")
    else:
        print("\nОтфильтрованные данные не найдены.")
else:
    print("FAIL - данные не получены")
