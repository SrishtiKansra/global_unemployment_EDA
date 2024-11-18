import pandas as pd

file_path = 'global_unemployment_data.csv'

def filter_data(file_path, countries):
    df = pd.read_csv(file_path)
    df['country_name'] = df['country_name'].str.strip()
    df.fillna(0, inplace=True)
    filtered_df = df[df['country_name'].isin(countries)]
    return filtered_df

countries = ['Afghanistan','India','Italy','Germany','France']
filtered_data =filter_data(file_path, countries)
print('These are the test countries.',filtered_data['country_name'].unique())
print(filtered_data)
