import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()

token = os.getenv('DESTATIS_TOKEN')
base_url = "https://www-genesis.destatis.de/genesisWS/rest/2020/"

headers = {
    'Content-Type': 'application/x-www-form-urlencoded',
    'username': token,
    'password': '' 
}

def search_tables(search_term, category='tables', max_results=20):
    """
    Поиск таблиц в GENESIS по ключевым словам
    """
    payload = {
        'term': search_term,
        'category': category,
        'pagelength': str(max_results),
        'language': 'de'
    }
    
    try:
        response = requests.post(
            base_url + 'find/find',
            headers=headers,
            data=payload
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get('Tables', [])
        else:
            print(f"Ошибка поиска: {response.status_code}")
            return []
    except Exception as e:
        print(f"Ошибка: {e}")
        return []


def print_tables(tables, title):
    """
    Красиво выводит результаты поиска
    """
    print("\n" + "="*80)
    print(title)
    print("="*80)
    
    if not tables:
        print("Ничего не найдено")
        return
    
    for i, table in enumerate(tables, 1):
        code = table.get('Code', 'N/A')
        content = table.get('Content', 'N/A')
        time_range = table.get('Time', 'N/A')
        
        print(f"\n{i}. Code: {code}")
        print(f"   Описание: {content}")
        print(f"   Период: {time_range}")


# ============================================================================
# ПОИСК НУЖНЫХ ТАБЛИЦ
# ============================================================================

print("🔍 ПОИСК ТАБЛИЦ В GENESIS DESTATIS")
print("="*80)

# 1. Elite Holders - ищем таблицы про статус занятости или профессии
print("\n[1] ELITE HOLDERS - Руководители и самозанятые")
searches = [
    "Erwerbstätige Stellung Beruf",
    "Selbstständige",
    "Führungskräfte",
    "Stellung im Beruf"
]

for search in searches:
    tables = search_tables(search, max_results=5)
    print_tables(tables, f"Поиск: '{search}'")

# 2. Frustrated Elites - безработные с образованием
print("\n\n[2] FRUSTRATED ELITES - Безработные с высшим образованием")
searches = [
    "Arbeitslose Qualifikation",
    "Arbeitslose Bildung",
    "Akademiker Arbeitslosigkeit"
]

for search in searches:
    tables = search_tables(search, max_results=5)
    print_tables(tables, f"Поиск: '{search}'")

# 3. Mass Mobilization - забастовки
print("\n\n[3] MASS MOBILIZATION - Забастовки и протесты")
searches = [
    "Streik",
    "Arbeitskampf",
    "Arbeitskämpfe"
]

for search in searches:
    tables = search_tables(search, max_results=5)
    print_tables(tables, f"Поиск: '{search}'")

print("\n" + "="*80)
print("✅ Поиск завершен")
print("="*80)
print("\nИНСТРУКЦИЯ:")
print("1. Найдите коды таблиц выше, которые подходят по описанию")
print("2. Обновите файл new_loader.py с правильными кодами")
print("3. Запустите new_loader.py для скачивания данных")
