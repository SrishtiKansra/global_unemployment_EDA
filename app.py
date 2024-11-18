import streamlit as st
import pandas as pd
import plotly.express as px

from data_import import filter_data
from data_analysis import prepare_data
from model_training import train_model
from predictions import predict_future
from visualisation import plot_trends
from prophet import Prophet

file_path = 'global_unemployment_data.csv'

def get_all_countries(file_path):
    df = pd.read_csv(file_path)
    df['country_name'] = df['country_name'].str.strip()
    countries = df['country_name'].unique().tolist()
    return countries

countries = get_all_countries(file_path)
st.title('Global Unemployment Statistics')
selected_country =st.selectbox('Select a Country', countries)
cleaned_data = filter_data(file_path, countries)
future_data = pd.DataFrame()
st.write('Debug cleaned data',cleaned_data.columns)
st.write('Debug head',cleaned_data.head())

if selected_country != 'All Countries':
    country_data = cleaned_data[cleaned_data['country_name'] == selected_country]
    if not country_data.empty:
        prepared_data = prepare_data(country_data)
        st.write('Debug prepared data', prepared_data.head())
        model = train_model(prepared_data)
        future_data = predict_future(model, 2024, 2030)
        st.write(f'Unemployment Rate Prediction for {selected_country}')
    else:
        st.warning(f"No data available for {selected_country}")
else:
    prepared_data = prepare_data(cleaned_data)
    st.write('Debug prepared data else block', prepared_data.head())
    model = train_model(prepared_data)
    future_data = predict_future(model, 2024, 2030)
    st.write('Unemployment Rate Prediction for All Countries')

st.write("Debug: Future Data Head", future_data.head())

if not future_data.empty:
    future_data['year'] = future_data['year'].dt.year
    fig = px.line(future_data, x='year', y='unemployment_rate',
                  title=f'Unemployment Rate Prediction for {selected_country if selected_country != "All Countries" else "All Countries"}')
    fig.update_layout(xaxis_title='Year', yaxis_title='Unemployment Rate')
    st.plotly_chart(fig)
else:
    st.warning('No future data available to plot.')



