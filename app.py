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

selected_country = st.selectbox('Select a Country', countries)

gender = st.selectbox('Select Gender', ['All', 'Male', 'Female'])

age_category = st.selectbox('Select Age Category', ['All', 'Youth', 'Adults', 'Children'])

cleaned_data = filter_data(file_path, countries)


if gender != 'All':
    cleaned_data = cleaned_data[cleaned_data['sex'] == gender]

if age_category != 'All':
    cleaned_data = cleaned_data[cleaned_data['age_categories'] == age_category]

if selected_country != 'All Countries':
    country_data = cleaned_data[cleaned_data['country_name'] == selected_country]
    if not country_data.empty:
        prepared_data = prepare_data(country_data)
        model = train_model(prepared_data)
        future_data = predict_future(model, 2024, 2030)
        # st.write(f'Unemployment Rate Prediction for {selected_country}')

        past_data = country_data.melt(id_vars=['country_name', 'sex', 'age_categories', 'indicator_name'],
                                      value_vars=[str(year) for year in range(2014, 2024)],
                                      var_name='year', value_name='unemployment_rate')
        past_data['year'] = past_data['year'].astype(int)

        past_data_aggregated = past_data.groupby(['year', 'age_categories'])['unemployment_rate'].mean().reset_index()

        fig_past = px.line(past_data_aggregated, x='year', y='unemployment_rate', color='age_categories',
                           title=f'Average Unemployment Rate for {selected_country} (2014-2024) by Age Category')
        fig_past.update_layout(xaxis_title='Year', yaxis_title='Unemployment Rate')
        st.plotly_chart(fig_past)
        st.write('Ages for each categories are as follow: ')
        st.write('Children = Under 15')
        st.write('Youth = 15-24')
        st.write('Adult = 25+')

    else:
        st.warning(f"No data available for {selected_country}")
else:
    prepared_data = prepare_data(cleaned_data)
    model = train_model(prepared_data)
    future_data = predict_future(model, 2024, 2030)
    st.write('Unemployment Rate Prediction for All Countries')

if not future_data.empty:
    future_data['year'] = future_data['year'].dt.year
    fig_future = px.line(future_data, x='year', y='unemployment_rate',
                         title=f'Unemployment Rate Prediction for {selected_country if selected_country != "All Countries" else "All Countries"}')
    fig_future.update_layout(xaxis_title='Year', yaxis_title='Unemployment Rate')
    st.plotly_chart(fig_future)
else:
    st.warning('No future data available to plot.')