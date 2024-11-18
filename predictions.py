import pandas as pd
# def predict_future(model,start_year,end_year):
#     future = model.make_future_dataframe(periods = end_year-start_year+1,freq='Y')
#     forecast = model.predict(future)
#     future_data = forecast[['ds','yhat']].rename(columns={'ds':'year','yhat':'unemployment_rate'})
#     return future_data[future_data['year'].dt.year>=start_year]

def predict_future(model, start_year, end_year):
    future_years = pd.date_range(start=f'{start_year}-01-01', end=f'{end_year}-01-01', freq='Y')
    future = pd.DataFrame({'ds': future_years})
    forecast = model.predict(future)
    future_data = forecast[['ds', 'yhat']].rename(columns={'ds': 'year', 'yhat': 'unemployment_rate'})
    return future_data[future_data['year'].dt.year <= end_year]
