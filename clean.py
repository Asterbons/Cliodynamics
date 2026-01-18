import pandas as pd

# Load the CSV data with correct separator and encoding
df = pd.read_csv('data.csv', sep=';', encoding='utf-8')

# Select only the relevant columns
cols_to_keep = ['time', '1_variable_attribute_label', '3_variable_attribute_label', 'value']
df_clean = df[cols_to_keep].copy()

# Convert 'value' column to float (handling German decimal comma)
df_clean['value'] = pd.to_numeric(df_clean['value'].str.replace(',', '.'), errors='coerce')
df_clean = df_clean.dropna(subset=['value'])

# Fix encoding artifacts
def fix_encoding(text):
    if isinstance(text, str):
        return text.replace('Ã¤', 'ä').replace('Ã¼', 'ü').replace('Ã¶', 'ö').replace('ÃŸ', 'ß')
    return text

df_clean['1_variable_attribute_label'] = df_clean['1_variable_attribute_label'].apply(fix_encoding)
df_clean['3_variable_attribute_label'] = df_clean['3_variable_attribute_label'].apply(fix_encoding)

# Define categories to filter by
categories = [
    'Gesamtindex', 
    'Tatsächliche Nettokaltmiete',
    'Nahrungsmittel'
]
df_final = df_clean[df_clean['3_variable_attribute_label'].isin(categories)]

# Map German month names to numbers
month_map = {
    'Januar': 1, 'Februar': 2, 'März': 3, 'April': 4, 'Mai': 5, 'Juni': 6,
    'Juli': 7, 'August': 8, 'September': 9, 'Oktober': 10, 'November': 11, 'Dezember': 12
}
df_final['month_num'] = df_final['1_variable_attribute_label'].map(month_map)

# Create a datetime object for sorting
df_final['date'] = pd.to_datetime(df_final['time'].astype(str) + '-' + df_final['month_num'].astype(str) + '-01')

# Sort by date and display the first few rows
df_final = df_final.sort_values('date')
print(df_final.head())