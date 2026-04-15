import os
import io
import zipfile
import requests
from pathlib import Path
from dotenv import load_dotenv

# Destatis table 21311-0012: Studentflow — first-semester entrants by Studienfach
# Annual pipeline inflow — used alongside enrolled students (21311-0003) to track
# the elite aspirant flow entering the degree pipeline each year.

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

# Same elite-status subject codes as load_students.py
subject_codes = {
    'SF135': 'Rechtswissenschaft',
    'SF129': 'Politikwissenschaft/Politologie',
    'SF021': 'Betriebswirtschaftslehre',
    'SF175': 'Volkswirtschaftslehre',
    'SF182': 'Internationale Betriebswirtschaft/Management',
    'SF184': 'Wirtschaftswissenschaften',
    'SF149': 'Soziologie',
    'SF127': 'Philosophie',
    'SF068': 'Geschichte',
    'SF272': 'Alte Geschichte',
    'SF275': 'Wissenschaftsgeschichte/Technikgeschichte',
    'SF302': 'Medienwissenschaft',
    'SF303': 'Kommunikationswissenschaft/Publizistik',
}


def download_and_unzip(payload):
    try:
        response = requests.post(base_url + 'data/tablefile', headers=headers, data=payload)
        if response.status_code == 200:
            if response.content.startswith(b'Error'):
                print(f"Ошибка в ответе сервера: {response.text[:200]}")
                return None
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    return f.read().decode('utf-8')
        else:
            print(f"HTTP Error {response.status_code}: {response.text[:200]}")
            return None
    except Exception as e:
        print(f"Ошибка запроса: {e}")
        return None


def main():
    print("Скачивание данных о студентах-первокурсниках (21311-0012)...")

    payload = {
        'name': '21311-0012',
        'startyear': '2015',
        'compress': 'true',
        'format': 'ffcsv',
        'language': 'de'
    }

    csv_text = download_and_unzip(payload)

    if not csv_text:
        print("FAIL — данные не получены")
        return

    print("OK — данные получены, фильтрация по специальностям...")

    lines = csv_text.splitlines()

    # Locate header row
    header_line = None
    data_start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('21311'):
            data_start_idx = i
            if i > 0:
                header_line = lines[i - 1]
            break

    all_rows = []
    if header_line:
        all_rows.append(header_line)
        print(f"Заголовок: {header_line[:120]}...")

    matched_total = 0
    for code, name in subject_codes.items():
        matching = [line for line in lines[data_start_idx:] if code in line]
        all_rows.extend(matching)
        matched_total += len(matching)
        print(f"  {name} ({code}): {len(matching)} строк")

    if matched_total == 0:
        print("\nПредупреждение: строки не найдены по кодам SF.")
        print("Сохраняем полный CSV для ручной проверки заголовков.")
        raw_path = DATA_RAW / 'data_studienanfaenger_raw_full.csv'
        with open(raw_path, 'w', encoding='utf-8') as f:
            f.write(csv_text)
        print(f"Полный CSV сохранён: {raw_path}")
        return

    out_path = DATA_RAW / 'data_studienanfaenger.csv'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_rows))

    print(f"\nФайл сохранён: {out_path}")
    print(f"Всего строк данных: {matched_total}")


if __name__ == '__main__':
    main()
