import os
import requests
import io
import zipfile
import time
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_RAW.mkdir(parents=True, exist_ok=True)

token = os.getenv('DESTATIS_TOKEN')
base_url = "https://www-genesis.destatis.de/genesisWS/rest/2020/"

headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
    'username': token,
    'password': '' 
}

def download_table(table_code, filename, timescale=None):
    print(f"Downloading {table_code} ({timescale or 'default'}) -> {filename}...", end=" ")
    payload = {
        'name': table_code,
        'startyear': '2015',
        'compress': 'true',
        'format': 'ffcsv',
        'language': 'de'
    }
    if timescale:
        payload['timescale'] = timescale
        
    try:
        response = requests.post(base_url + 'data/tablefile', headers=headers, data=payload)
        if response.status_code == 200:
            if b"direkten Abruf" in response.content and b"gross" in response.content.lower():
                print("TOO LARGE. Retrying with job=true...", end=" ")
                payload['job'] = 'true'
                response = requests.post(base_url + 'data/tablefile', headers=headers, data=payload)
                print("JOB SUBMITTED. (Please use long-term loader for this).")
                return False
                
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    data = f.read().decode('utf-8')
                    with open(DATA_RAW / filename, "w", encoding="utf-8") as out:
                        out.write(data)
            print("DONE.")
            return True
        else:
            print(f"FAIL: {response.status_code}")
            return False
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return False

# Tables to download for PSI v4
tables = [
    ("61111-0004", "data_cpi_general.csv", "monatlich"), # Monthly CPI
    ("61111-0006", "data_food_prices.csv", "monatlich"), # Monthly Food
    ("81111-0001", "data_gdp_quarterly.csv", "vierteljährlich"), # Quarterly GDP
    ("12411-0005", "data_youth_annual.csv", "jährlich"),
    ("71211-0001", "data_tax_revenue.csv", "vierteljährlich"),
    ("74111-0001", "data_civil_servants.csv", "jährlich"),  # Civil servants (state capacity)
    ("12211-0009", "data_holders_raw.csv", "jährlich"),    # Holders (elite positions)
]

for code, fname, scale in tables:
    download_table(code, fname, scale)
    time.sleep(1)
