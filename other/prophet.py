import pandas as pd
from fbprophet import Prophet
import numpy as np
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_plotly, plot_cross_validation_metric
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
from adtk.data import validate_series
import matplotlib.pyplot as plt
from datetime import datetime

if __name__ == "__main__":

    df_2 = pd.read_csv('samples/1616116101_1-energy_aplus_inc.csv', parse_dates=True)
    df_size = len(df_2['ds'])

    df = pd.DataFrame()
    df['sss'] = df_2['ds'][:int(df_size*0.8)]
    df['ds'] = df_2['ds'][:int(df_size*0.8)]
    df['y'] = df_2['y'][:int(df_size*0.8)]
    
    df['ds'] = df['ds'].astype('datetime64[ms]')
    #print(df['ds'])
    df.head()
    print(df)
    exit()

    m = Prophet()
    m.fit(df)

    future = m.make_future_dataframe(periods=365)
    future.tail()

    forecast = m.predict(future)
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
    print(forecast.columns)

    #forecast['yhat'] = np.exp(forecast['yhat']) - 1
    forecast_size = len(forecast['ds'])
    forecastValue = forecast['trend'][int(df_size*0.8)-1]
    realValue = df['y'][int(df_size*0.8)-1]
    differenceValue = forecastValue - realValue

    print(df_size*0.8)

    print('Real Value: %.2f' % realValue)
    print('Forecast (yhat): %.2f' % forecastValue)
    print('Difference Value: %.2f' % differenceValue)
    print('Forecast (yhat_lower): %.2f' % (forecast['yhat_lower'][int(df_size*0.8)-1]))
    print('Forecast (yhat_upper): %.2f' % (forecast['yhat_upper'][int(df_size*0.8)-1]))
    """
    #fig1 = m.plot(forecast)
    #fig2 = m.plot_components(forecast)
    #plt.show()

    #df_cv = validate_series(df)
    #df_cv = cross_validation(m, initial='40 days', horizon = '20 days')
    #df_cv.head()
    #print(df_cv)
    df_p = performance_metrics(df_cv)
    df_p.head()
    print(df_p)
    """
    #fig = plot_cross_validation_metric(df_cv, metric='mape')
    #plt.show()
