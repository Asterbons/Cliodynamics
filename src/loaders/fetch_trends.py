import pandas as pd
from pytrends.request import TrendReq
import time
import random
from datetime import datetime

def fetch_mobilization_trends(start_date='2015-01-01', geo='DE'):
    keywords = ['Bahnstreik', 'Bauernprotest', 'demo heute', 'Demonstration']
    
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Mobile/15E148 Safari/604.1"
    ]

    pytrends = TrendReq(
        hl='de-DE', 
        tz=60, 
        requests_args={'headers': {'User-Agent': random.choice(user_agents)}}
    )
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    timeframe = f"{start_date} {end_date}"
    
    max_retries = 5
    for attempt in range(max_retries):
        try:
            print(f"Попытка {attempt + 1}...")
            pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo=geo, gprop='')
            df = pytrends.interest_over_time()
            
            if not df.empty:
                if 'isPartial' in df.columns:
                    df = df.drop(columns=['isPartial'])
                
                df['mobilization_index'] = df.mean(axis=1)
                df_monthly = df.resample('MS').mean()
                return df_monthly.reset_index()
            
            print("Google вернул пустой результат, пробую снова...")
            time.sleep(5)
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                wait_time = (attempt + 1) * random.uniform(20, 40)
                print(f"Блокировка 429. Ожидание {wait_time:.1f} сек...")
                time.sleep(wait_time)
                # Обновляем сессию и User-Agent при блокировке
                pytrends = TrendReq(
                    hl='de-DE', 
                    tz=60, 
                    requests_args={'headers': {'User-Agent': random.choice(user_agents)}}
                )
            else:
                print(f"Ошибка: {error_msg}")
                break
                
    return None

if __name__ == "__main__":
    print("Загрузка данных Google Trends с обходом блокировок...")
    trends_df = fetch_mobilization_trends()
    
    if trends_df is not None and not trends_df.empty:
        trends_df.to_csv('google_trends_mobilization.csv', index=False)
        print("Файл google_trends_mobilization.csv успешно создан.")
        print(trends_df.tail(15))
    else:
        print("Не удалось получить данные после всех попыток.")