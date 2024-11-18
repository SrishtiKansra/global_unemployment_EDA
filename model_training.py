from prophet import Prophet
def train_model(df):
    df_prophet = df.rename(columns = {'ds':'ds','y':'y'})
    model = Prophet()
    model.fit(df_prophet)
    return model
