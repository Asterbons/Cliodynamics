import pandas as pd

def fix_encoding(text):
    if isinstance(text, str):
        return text.replace('Ã¤', 'ä').replace('Ã¼', 'ü').replace('Ã¶', 'ö').replace('ÃŸ', 'ß')
    return text

def process_wages(file_path):
    df = pd.read_csv(file_path, sep=';', encoding='utf-8')
    
    cols = ['time', '1_variable_attribute_label', 'value_variable_label', 'value']
    df = df[cols].copy()
    
    df.loc[:, 'value'] = pd.to_numeric(df['value'].str.replace(',', '.'), errors='coerce')
    df = df.dropna(subset=['value'])
    
    df.loc[:, '1_variable_attribute_label'] = df['1_variable_attribute_label'].apply(fix_encoding)
    df.loc[:, 'value_variable_label'] = df['value_variable_label'].apply(fix_encoding)
    
    month_map = {
        'Januar': 1, 'Februar': 2, 'März': 3, 'April': 4, 'Mai': 5, 'Juni': 6,
        'Juli': 7, 'August': 8, 'September': 9, 'Oktober': 10, 'November': 11, 'Dezember': 12
    }
    
    df.loc[:, 'month_num'] = df['1_variable_attribute_label'].map(month_map)
    df = df.dropna(subset=['month_num'])
    
    df.loc[:, 'date'] = pd.to_datetime(
        df['time'].astype(str) + '-' + df['month_num'].astype(int).astype(str) + '-01'
    )
    
    return df.sort_values('date')

def process_prices(file_path):
    df = pd.read_csv(file_path, sep=';', encoding='utf-8')
    
    cols = ['time', '1_variable_attribute_label', '3_variable_attribute_label', '3_variable_code', 'value']
    df = df[cols].copy()
    
    df.loc[:, 'value'] = pd.to_numeric(df['value'].str.replace(',', '.'), errors='coerce')
    df = df.dropna(subset=['value'])
    
    df.loc[:, '1_variable_attribute_label'] = df['1_variable_attribute_label'].apply(fix_encoding)
    df.loc[:, '3_variable_attribute_label'] = df['3_variable_attribute_label'].apply(fix_encoding)
    
    month_map = {
        'Januar': 1, 'Februar': 2, 'März': 3, 'April': 4, 'Mai': 5, 'Juni': 6,
        'Juli': 7, 'August': 8, 'September': 9, 'Oktober': 10, 'November': 11, 'Dezember': 12
    }
    
    df.loc[:, 'month_num'] = df['1_variable_attribute_label'].map(month_map)
    df = df.dropna(subset=['month_num'])
    
    df.loc[:, 'date'] = pd.to_datetime(
        df['time'].astype(str) + '-' + df['month_num'].astype(int).astype(str) + '-01'
    )
    
    return df

df_wages = process_wages('data_wages.csv')
df_prices = process_prices('data.csv')

nominal_wages = df_wages[df_wages['value_variable_label'] == 'Nominallohnindex'][['date', 'value']].rename(columns={'value': 'wage_nominal'})
real_wages = df_wages[df_wages['value_variable_label'] == 'Reallohnindex'][['date', 'value']].rename(columns={'value': 'wage_real'})
rent_prices = df_prices[(df_prices['3_variable_attribute_label'] == 'Tatsächliche Nettokaltmiete') & (df_prices['3_variable_code'] == 'CC13A4')][['date', 'value']].rename(columns={'value': 'rent'})

master_df = pd.merge(nominal_wages, rent_prices, on='date', how='inner')
master_df = pd.merge(master_df, real_wages, on='date', how='inner')

master_df['wealth_pump'] = (master_df['rent'] / master_df['wage_nominal']) * 100

print(master_df.head())
master_df.to_csv('master_cliodynamics.csv', index=False)