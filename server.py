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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, recall_score, precision_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from keras.layers import Dense, LSTM
from keras.models import Sequential
import math
import sys
from datetime import datetime
import time

import statistics
import logging
import json
import socket
import os
import tqdm

train_dataset = 0.8

def menu():
    print("Classifiers:")
    print("\tann -> Artificial Neural Networks")
    print("\tsvm -> Support Vector Machines")
    print("\ttree -> Decision Tree")
    print("\tlinreg -> Linear Regression")
    print("\tforest -> Random Forest")
    print("\tprophet -> Facebook Prophet")
    return

def loadDataset(algorithm, file, splitData = False):
    start_time = time.time()
    biggestSplittedData = {'startIndex': 0, 'length': 0}
    continuousDataLength = 0

    if splitData:
        df2 = pd.read_csv(file, engine='python', parse_dates=True)
        print(df2['ds'][134])
        for i in range(1, len(df2['ds'])):
            continuousDataLength += 1
            if df2['ds'][i] != df2['ds'][i-1]+900000:
                if biggestSplittedData['length'] < continuousDataLength:
                    biggestSplittedData['startIndex'] = i
                    biggestSplittedData['length'] = continuousDataLength
                continuousDataLength = 0
        
        print(df2['ds'][biggestSplittedData['startIndex']:biggestSplittedData['startIndex']+biggestSplittedData['length']])
        print(biggestSplittedData['length'])
        df = pd.DataFrame()
        df['ds'] = df2['ds'][biggestSplittedData['startIndex']:biggestSplittedData['startIndex']+biggestSplittedData['length']]
        df['y'] = df2['y'][biggestSplittedData['startIndex']:biggestSplittedData['startIndex']+biggestSplittedData['length']]
        print(df)
        
    else:
        df = pd.read_csv(file, engine='python', parse_dates=True)

    if (df.empty):
        return 0

    if algorithm == "prophet":
        train_size = int(len(df)*train_dataset)
        val_size = len(df) - train_size
        train_val_size = train_size + val_size
        i = train_val_size
        H = 1

        df['ds'] = df['ds'].astype('datetime64[ms]')

        # Fit prophet model
        m = Prophet()
        m.fit(df[i-train_val_size:i])

        # Create dataframe with the dates we want to predict
        future = m.make_future_dataframe(periods=30, freq='15min')

        # Predict
        forecast = m.predict(future)

        print(forecast)
        """
        fig1 = m.plot(forecast)
        fig2 = m.plot_components(forecast)
        plt.show()"""

        forecastValue = forecast['yhat'][train_val_size+1]
        print(forecast['ds'][train_val_size-1])
        realValue = df['y'][train_val_size-1]
        differenceValue = forecastValue - realValue

        print('Real Value: %.2f' % realValue)
        print('Forecast (yhat): %.2f' % forecastValue)
        print('Difference Value: %.2f' % differenceValue)
        #print('Forecast (yhat_lower): %.2f' % (forecast['yhat_lower'][train_val_size-1]))
        #print('Forecast (yhat_upper): %.2f' % (forecast['yhat_upper'][train_val_size-1]))

        """
        df_cv = cross_validation(m, initial='40 days', horizon = '1 day')
        df_cv.head()
        print(df_cv)

        df_p = performance_metrics(df_cv)
        df_p.head()
        print(df_p)
        """

        return abs(differenceValue), forecastValue
    
    else:
        dataset = df['y'].values.astype('float32').reshape(-1, 1)
        dataset_dates = df['ds'].values.astype('datetime64[ms]').reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)

        train, test = dataset[0:int(len(dataset)*train_dataset),:], dataset[int(len(dataset)*train_dataset):len(dataset),:]

        # reshape into X=t and Y=t+1
        look_back = 1
        X_train, y_train = create_dataset(train, look_back)
        X_test, y_test = create_dataset(test, look_back)

        if algorithm == "ann":
            X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
            X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
            predict_train, predict_test = neuralNetwork(X_train, X_test, y_train, y_test, look_back)

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
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(y_test[0], predict_test[:,0]))
        print('Test Score: %.2f RMSE' % (testScore))

        # shift train predictions for plotting
        trainPredictPlot = np.empty_like(dataset)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(predict_train)+look_back, :] = predict_train

        # shift test predictions for plotting
        testPredictPlot = np.empty_like(dataset)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(predict_train)+(look_back*2)+2:len(dataset), :] = predict_test

        predictedValue = testPredictPlot[-1]#[len(predict_train)+(look_back*2)+1]
        realValue = scaler.inverse_transform(dataset)[-1]#df['y'][len(predict_train)+(look_back*2)+1]
        #differenceValue = predictedValue - realValue

        print('Predicted value for the next 15 minutes: %.2f' % (predictedValue))
        print('Real Value for the next 15 minutes: %.2f' % (realValue))
        #print('Difference: %.2f' % (differenceValue))

        # plot baseline and predictions
        """
        fig, ax = plt.subplots()

        xfmt = mdates.DateFormatter('%d-%m-%y %H:%M')
        ax.xaxis.set_major_formatter(xfmt)
        
        ax.plot(dataset_dates, scaler.inverse_transform(dataset))
        ax.plot(dataset_dates, trainPredictPlot)
        ax.plot(dataset_dates, testPredictPlot)

        ax.set(xlabel="Date",
        ylabel="Energy Consumption (Wh)",
        title="Quarter Hour Total Energy Consumption")

        plt.show()
        """

        return predictedValue[0]

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

def neuralNetwork(X_train, X_test, y_train, y_test, look_back):
    model = Sequential()

    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)

    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %0.2f" % (scores*100))

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

def getPrediction(files):
    if len(files) != 3:
        print("ERROR: Only 3 files are needed to perfomr this action!")
        return

    else:
        algorithms = ["ann", "svm", "tree", "linreg", "forest", "prophet"]

        value = [0, 0, 0]
        prediction = [0, 0, 0]
        i = 0
        print("#######################################")

        while i < len(files):
            value[i] = loadDataset(algorithms[3], files[i])
            i += 1
            print("#######################################")

        minValue = min(value)
        maxValue = max(value)

        print("Minimum Value: %.2f" % minValue)
        print("Minimum Phase: %d" % value.index(minValue))
        print("Maximum Value: %.2f" % maxValue)
        print("Maximum Phase: %d" % value.index(maxValue))

        mean_value = statistics.mean(value)

        print("Mean Value: %.2f" % mean_value)

        differenceValue = [0, 0, 0]

        differenceValue[0] = value[0] - mean_value
        differenceValue[1] = value[1] - mean_value
        differenceValue[2] = value[2] - mean_value

        print("Difference: %.2f" % differenceValue[value.index(minValue)])

        return value.index(maxValue)

if __name__ == "__main__":

    SERVER_HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
    SERVER_PORT = 65432        # Port to listen on (non-privileged ports are > 1023)
    BUFFER_SIZE = 4096
    SEPARATOR = "<SEPARATOR>"
    filename = ""

    if not os.path.exists("server_files/"):
        os.mkdir("server_files/")
    
    # create the server socket
    # TCP socket
    s = socket.socket()

    # bind the socket to our local address
    s.bind((SERVER_HOST, SERVER_PORT))

    # enabling our server to accept connections
    # 5 here is the number of unaccepted connections that
    # the system will allow before refusing new connections
    s.listen(10)
    print(f"[*] Listening as {SERVER_HOST}:{SERVER_PORT}")

    while True:
        client_socket = None
        try:
            # accept connection if there is any
            client_socket, address = s.accept() 
            # if below code is executed, that means the sender is connected
            print(f"[+] {address} is connected.")

            k = 0
            listFiles = []
            while k < 3:
                size = client_socket.recv(16).decode("utf-8") # Note that you limit your filename length to 255 bytes.
                if not size:
                    break
                size = int(size, 2)
                filename = client_socket.recv(size).decode("utf-8")
                filename = filename.replace("samples/","server_files/")
                filesize = client_socket.recv(32).decode("utf-8")
                filesize = int(filesize, 2)
                file_to_write = open(filename, 'wb')
                chunksize = 4096
                while filesize > 0:
                    if filesize < chunksize:
                        chunksize = filesize
                    data = client_socket.recv(chunksize)
                    file_to_write.write(data)
                    filesize -= len(data)

                file_to_write.close()
                print('File received successfully')
                listFiles.append(filename)

            phase = getPrediction(listFiles)
            print(phase)

            # close the client socket
            client_socket.close()

        except KeyboardInterrupt:
            if client_socket:
                client_socket.close()
            break

    # close the server socket
    s.close()
