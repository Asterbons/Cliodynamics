import os
import requests
import io
import zipfile
import time
from dotenv import load_dotenv

load_dotenv()

token = os.getenv('DESTATIS_TOKEN')
base_url = "https://www-genesis.destatis.de/genesisWS/rest/2020/"

headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
    'username': token,
    'password': '' 
}

# 1. State Capacity: Государственный долг (Квартальный)
# Таблица 71321-0005: Schulden beim nicht-öffentlichen Bereich
payload_debt = {
    'name': '71321-0005',
    'startyear': '2015',
    'compress': 'true',
    'format': 'ffcsv',
    'language': 'de'
}

# 2. Holders: Руководящие позиции (Годовые данные)
# Таблица 12211-9014 (или подобная для Führungskräfte) часто меняется.
# Используем прокси: Занятые с высоким статусом из Mikrozensus
# Для теста берем общую таблицу занятости, фильтрацию сделаем на этапе процессинга,
# так как точные коды профессий через API часто сбоят.
payload_holders = {
    'name': '13311-0002', # Erwerbstätige nach Berufssegmenten
    'startyear': '2015',
    'compress': 'true',
    'format': 'ffcsv',
    'language': 'de'
    # Мы не используем selection, чтобы скачать все профессии и отфильтровать 'Führungskräfte' локально
}

def download_and_unzip(url, headers, payload, filename):
    print(f"Скачиваю {filename}...", end=" ")
    try:
        response = requests.post(url + 'data/tablefile', headers=headers, data=payload)
        if response.status_code == 200:
            if response.content.startswith(b'Error'):
                print("FAIL (Server Error)")
                return
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                file_name = z.namelist()[0]
                with z.open(file_name) as f: 
                    data = f.read().decode('utf-8')
                    with open(filename, "w", encoding="utf-8") as out:
                        out.write(data)
            print("OK")
        else:
            print(f"FAIL (Status {response.status_code})")
    except Exception as e:
        print(f"Error: {e}")

# Запуск
download_and_unzip(base_url, headers, payload_debt, "data_debt.csv")
# download_and_unzip(base_url, headers, payload_holders, "data_holders.csv") 
# Holders часто весят много, для теста пока возьмем только долг, 
# так как он критичнее для формулы (знаменатель).