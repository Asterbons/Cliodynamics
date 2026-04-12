import os
import sys
import requests
import io
import zipfile
from pathlib import Path
from dotenv import load_dotenv

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'

load_dotenv(PROJECT_ROOT / '.env')

token = os.getenv('DESTATIS_TOKEN')
base_url = "https://www-genesis.destatis.de/genesisWS/rest/2020/"

# POST + Headers
headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
    'username': token,
    'password': '' 
}

# parameters
payload_wages = {
    'name': '62361-0001', # Nominallohnindex
    'startyear': '2020',
    'compress': 'true',
    'format': 'ffcsv',
    'language': 'de'
}
payload_price_index = {
    'name': '61111-0004', # price index(for Wealth Pump)
    'startyear': '2020',
    'compress': 'true',
    'format': 'ffcsv',
    'language': 'de'
}

def download_and_unzip(url, headers, payload):
    response = requests.post(url + 'data/tablefile', headers=headers, data=payload)
    
    if response.status_code == 200:
        # we get zip file
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            file_name = z.namelist()[0]
            with z.open(file_name) as f: 
                return f.read().decode('utf-8')
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

csv_data_wages = download_and_unzip(base_url, headers, payload_wages)
csv_data_price_index = download_and_unzip(base_url, headers, payload_price_index)

if csv_data_wages:
    with open(DATA_RAW / "data_wages.csv", "w", encoding="utf-8") as f:
        f.write(csv_data_wages)
    print("Success: data_wages.csv updated.")

if csv_data_price_index:
    with open(DATA_RAW / "data_price_index.csv", "w", encoding="utf-8") as f:
        f.write(csv_data_price_index)
    print("Success: data_price_index.csv updated.")