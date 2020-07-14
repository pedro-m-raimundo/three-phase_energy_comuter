import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
from fbprophet.plot import plot_plotly, plot_cross_validation_metric
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, r2_score, recall_score, precision_score, mean_squared_error, mean_absolute_error, precision_recall_fscore_support
from sklearn.utils import check_array
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from keras.layers import Dense, LSTM
from keras.models import Sequential
import math
import sys
from datetime import datetime
import time

import logging
import json
import socket
import os
import tqdm

train_dataset = 0.8
look_back = 1

def menu():
    print("\npython test_script.py [Classifier] [Sample] (t)")
    print("\nClassifiers:")
    print("\tann -> Artificial Neural Networks")
    print("\tsvm -> Support Vector Machines")
    print("\ttree -> Decision Tree")
    print("\tlinreg -> Linear Regression")
    print("\tforest -> Random Forest")
    print("\tprophet -> Facebook Prophet")
    print("\nSamples:")
    print("\t0 -> Periodical")
    print("\t1 -> Continuous")
    print("\t2 -> Discontinuous")
    print("\n(t) -> select largest continuous stream of data")
    return

def loadDataset(algorithm, file, splitData):
    start_time = time.time()
    biggestSplittedData = {'startIndex': 0, 'length': 0}
    continuousDataLength = 0

    if splitData:
        df2 = pd.read_csv(file, engine='python', parse_dates=True)
        #print(df2['ds'][134])
        for i in range(1, len(df2['ds'])):
            continuousDataLength += 1
            if df2['ds'][i] != df2['ds'][i-1]+900000:
                if biggestSplittedData['length'] < continuousDataLength:
                    biggestSplittedData['startIndex'] = i
                    biggestSplittedData['length'] = continuousDataLength
                continuousDataLength = 0
        
        #print(df2['ds'][biggestSplittedData['startIndex']:biggestSplittedData['startIndex']+biggestSplittedData['length']])
        #print(biggestSplittedData['length'])
        df = pd.DataFrame()
        df['ds'] = df2['ds'][biggestSplittedData['startIndex']:biggestSplittedData['startIndex']+biggestSplittedData['length']]
        df['y'] = df2['y'][biggestSplittedData['startIndex']:biggestSplittedData['startIndex']+biggestSplittedData['length']]
        print(len(df))
        
    else:
        df = pd.read_csv(file, engine='python', parse_dates=True)

    if algorithm == "prophet":
        
        train_size = int(len(df)*train_dataset)
        val_size = len(df) - train_size
        train_val_size = train_size + val_size
        i = train_val_size
        H = 1

        df['ds'] = df['ds'].astype('datetime64[ms]')
        dataset_dates = df['ds'].values.astype('datetime64[ms]').reshape(-1, 1)

        # Fit prophet model
        m = Prophet()
        m.fit(df[i-train_val_size:i])
        
        # Create dataframe with the dates we want to predict
        future = m.make_future_dataframe(periods=30, freq='15min')

        # Predict
        forecast = m.predict(future)

        #print(forecast)
        
        forecastValue = forecast['yhat'][train_val_size+1]
        #print(forecast['ds'][train_val_size-1])
        #realValue = df['y'][train_val_size-1]
        #differenceValue = forecastValue - realValue

        #print('Real Value: %.2f' % realValue)
        #print('Forecast (yhat): %.2f' % forecastValue)
        #print('Difference Value: %.2f' % differenceValue)

        metric_df = forecast.set_index('ds')[['yhat']].join(df.set_index('ds').y).reset_index()
        metric_df.dropna(inplace=True)
        
        print("RMSE: %.2f" % math.sqrt(mean_squared_error(metric_df.y, metric_df.yhat)))
        print("MSE: %.2f" % mean_squared_error(metric_df.y, metric_df.yhat))
        print("MAE: %.2f" % mean_absolute_error(metric_df.y, metric_df.yhat))
        print("MAPE: %.2f" % mean_absolute_percentage_error(metric_df.y, metric_df.yhat))
        print("R2: %.2f" % r2_score(metric_df.y, metric_df.yhat))
        
        """
        df_cv = cross_validation(m, horizon = '15 days')
        df_cv.head()
        #print(df_cv)
        
        df_p = performance_metrics(df_cv)#, metric=['rmse'])
        df_p.head()
        print("RMSE:", df_p)
        """
        #fig = plot_cross_validation_metric(df_cv, metric='mape')
        #plt.show()

        print("Time:", time.time() - start_time)
        
        fig, ax = plt.subplots()

        xfmt = mdates.DateFormatter('%d-%m-%y %H:%M')
        ax.xaxis.set_major_formatter(xfmt)

        # shift train predictions for plotting
        trainPredictPlot = np.empty_like(metric_df.yhat)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(df)*train_size+look_back] = metric_df.yhat[look_back:len(df)*train_size+look_back]

        # shift test predictions for plotting
        testPredictPlot = np.empty_like(metric_df.yhat)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(df)*train_size+(look_back*2)+1:len(df)-1] = metric_df.yhat[len(df)*train_size+(look_back*2)+1:len(df)-1]
        
        ax.plot(dataset_dates, df['y'], label="Real data")
        ax.plot(dataset_dates, trainPredictPlot, label="Train predicted data")
        #ax.plot(dataset_dates, testPredictPlot, label="Test predicted data")
        ax.legend(loc='lower right')

        ax.set(xlabel="Date",
        ylabel="Energy Consumption (Wh)",
        title="Quarter Hour Total Energy Consumption")

        plt.show()
        """
        fig1 = m.plot(forecast)
        #fig2 = m.plot_components(forecast)

        axes = fig1.get_axes()
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Energy Consumption (Wh)')
        axes[0].set_title("Quarter Hour Total Energy Consumption")

        plt.show()
        """

        return forecastValue

    else:
        dataset = df['y'].values.astype('float32').reshape(-1, 1)
        dataset_dates = df['ds'].values.astype('datetime64[ms]').reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)

        train, test = train_test_split(dataset, test_size=1-train_dataset, shuffle=False)

        # reshape into X=t and Y=t+1
    
        X_train, y_train = create_dataset(train, look_back)
        X_test, y_test = create_dataset(test, look_back)

        if "ann" in algorithm:
            X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
            X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
            predict_train, predict_test = neuralNetwork(X_train, X_test, y_train, y_test, look_back, algorithm.replace("ann",""))

        elif algorithm == "svm":
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
            predict_train, predict_test = svmpy(X_train, X_test, y_train, y_test)
        
        elif algorithm == "tree":
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
            predict_train, predict_test = decisionTree(X_train, X_test, y_train, y_test)
        
        elif algorithm == "linreg":
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
            predict_train, predict_test = LinearReg(X_train, X_test, y_train, y_test)

        elif algorithm == "forest":
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
            predict_train, predict_test = RandomForest(X_train, X_test, y_train, y_test)

        else:
            print("ERROR: No known classifier!")
            menu()
            exit()
        
        #print("Accuracy (train): %0.2f (+/- %0.2f)" % (predict_train.mean(), predict_train.std() * 2))
        #print("Accuracy (test): %0.2f (+/- %0.2f)" % (predict_test.mean(), predict_test.std() * 2))

        # invert predictions
        predict_train = scaler.inverse_transform(predict_train)
        y_train = scaler.inverse_transform([y_train])
        predict_test = scaler.inverse_transform(predict_test)
        y_test = scaler.inverse_transform([y_test])

        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(y_train[0], predict_train[:,0]))
        #print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(y_test[0], predict_test[:,0]))
        print('RMSE: %.2f' % testScore)

        #print('Train MSE: %.2f MSE' % mean_squared_error(y_train[0], predict_train[:,0]))
        print('MSE: %.2f' %  mean_squared_error(y_test[0], predict_test[:,0]))

        #print('Train MAE: %.2f MAE' % mean_absolute_error(y_train[0], predict_train[:,0]))
        print('MAE: %.2f' %  mean_absolute_error(y_test[0], predict_test[:,0]))

        print('MAPE: %.2f' % mape_vectorized_v2(y_test[0], predict_test[:,0]))

        print('R2: %.2f' % r2_score(y_test[0], predict_test[:,0]))

        #print('Train MAPE: %.2f MAPE' % mean_absolute_percentage_error(y_train[0], predict_train[:,0]))
        #print('Test MAPE: %.2f MAPE' %  mean_absolute_percentage_error(y_test[0], predict_test[:,0]))

        # shift train predictions for plotting
        trainPredictPlot = np.empty_like(dataset)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(predict_train)+look_back, :] = predict_train

        # shift test predictions for plotting
        testPredictPlot = np.empty_like(dataset)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(predict_train)+(look_back*2)+1:len(dataset)-1, :] = predict_test

        predictedValue = testPredictPlot[-1]#[len(predict_train)+(look_back*2)+1]
        realValue = scaler.inverse_transform(dataset)[-1]#df['y'][len(predict_train)+(look_back*2)+1]
        differenceValue = predictedValue - realValue

        #print('Predicted value for the next 15 minutes: %.2f' % (predictedValue))
        #print('Real Value for the next 15 minutes: %.2f' % (realValue))
        #print('Difference: %.2f' % (differenceValue))

        # plot baseline and predictions

        print("Time:", time.time() - start_time)
        
        fig, ax = plt.subplots()

        xfmt = mdates.DateFormatter('%d-%m-%y %H:%M')
        ax.xaxis.set_major_formatter(xfmt)
        
        ax.plot(dataset_dates, scaler.inverse_transform(dataset), label="Real data")
        ax.plot(dataset_dates, trainPredictPlot, label="Train predicted data")
        ax.plot(dataset_dates, testPredictPlot, label="Test predicted data")
        ax.legend(loc='upper right')

        ax.set(xlabel="Date",
        ylabel="Energy Consumption (Wh)",
        title="Quarter Hour Total Energy Consumption")

        plt.grid()
        plt.show()
        
        return predictedValue[0]


def mape_vectorized_v2(a, b):
    mask = a != 0
    return (np.fabs(a - b)/a)[mask].mean()*100


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def neuralNetwork(X_train, X_test, y_train, y_test, look_back, algorithm):
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))

    activationTypes = ['relu', 'sigmoid']
    optimizerTypes = ['adam', 'sgd']

    if algorithm[0] == "r":
        model.add(Dense(1, activation=activationTypes[0]))
    elif algorithm[0] == "s":
        model.add(Dense(1, activation=activationTypes[1]))
    else:
        model.add(Dense(1))

    if algorithm[1] == "a":
        model.compile(loss='mean_squared_error', optimizer=optimizerTypes[0], metrics=['sparse_categorical_accuracy'])
    else:
        model.compile(loss='mean_squared_error', optimizer=optimizerTypes[1], metrics=['sparse_categorical_accuracy'])

    model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=2)

    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %0.2f" % (scores[1]))

    # make predictions
    predict_train = model.predict(X_train)
    predict_test = model.predict(X_test)

    return predict_train, predict_test


def svmpy(X_train, X_test, y_train, y_test):
    model = SVR()
    model.fit(X_train, y_train)
    
    scores = cross_val_score(model, X_test, y_test, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # make predictions
    predict_train = model.predict(X_train).reshape(-1,1)
    predict_test = model.predict(X_test).reshape(-1,1)

    return predict_train, predict_test


def decisionTree(X_train, X_test, y_train, y_test):
    model = DecisionTreeRegressor(random_state=0)
    model.fit(X_train, y_train)

    scores = cross_val_score(model, X_test, y_test, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # make predictions
    predict_train = model.predict(X_train).reshape(-1,1)
    predict_test = model.predict(X_test).reshape(-1,1)

    return predict_train, predict_test


def LinearReg(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)

    scores = cross_val_score(model, X_test, y_test, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # make predictions
    predict_train = model.predict(X_train).reshape(-1,1)
    predict_test = model.predict(X_test).reshape(-1,1)

    return predict_train, predict_test


def RandomForest(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    model.fit(X_train, y_train)

    scores = cross_val_score(model, X_test, y_test, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    # make predictions
    predict_train = model.predict(X_train).reshape(-1,1)
    predict_test = model.predict(X_test).reshape(-1,1)

    return predict_train, predict_test


def getPrediction(splitData):

    samples = [
        'samples/periodic_data.csv',
        'samples/688AB5004D91_02-energy_aplus_inc.csv',
        'samples_new/688AB5015D8A%3A02-energy_aplus_inc.csv',
    ]

    #algorithms = [
    #    "ann-a", "ann-s",
    #    "annra", "annrs",
    #    "annsa", "annss",
    #    "svm", "tree",
    #    "linreg", "forest",
    #    "prophet",
    #]

    #value = loadDataset(algorithms[0], file, splitData)
    value = loadDataset(sys.argv[1], samples[int(sys.argv[2])], splitData)

    #print("Value %.2f" % value)

    #print("Minimum Value: %.2f" % minValue)
    #print("Phase: %d" % value.index(minValue))

    return #value.index(minValue)

if __name__ == "__main__":
    if len(sys.argv) > 2:
        getPrediction(True if len(sys.argv) > 3 and sys.argv[3] == "t" else False)

    elif len(sys.argv[1]) > 1 and sys.argv[1] == "help":
        menu()

    
    