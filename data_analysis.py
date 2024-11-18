def prepare_data(df):
        year_columns = [str(year) for year in range(2014, 2025)]
        df = df.melt(id_vars=['country_name', 'indicator_name', 'sex', 'age_group', 'age_categories'],
                     value_vars=year_columns,
                     var_name='year',
                     value_name='unemployment_rate')

        df['year'] = df['year'].astype(int)
        return df.rename(columns = {'year':'ds','unemployment_rate':'y'})
