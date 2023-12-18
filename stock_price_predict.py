# -*- coding:utf-8 -*-

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

plt.rcParams['font.family'] ='Malgun Gothic'
plt.rcParams['axes.unicode_minus'] =False


# 종목 코드에 해당하는 주식 데이터 가져오기
def get_stock_data(stock_code, start_date, end_date):
    stock_data = yf.download(stock_code, start=start_date, end=end_date)
    return stock_data

# 데이터 전처리 함수
def preprocess_data(stock_data):
    data = stock_data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# 시퀀스 데이터로 변환
def create_sequences(data, look_back=1):
    x, y = [], []
    for i in range(len(data) - look_back):
        x.append(data[i:(i + look_back), 0])
        y.append(data[i + look_back, 0])
    return np.array(x), np.array(y)

# LSTM 모델 정의
def build_lstm_model(look_back):
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(1, look_back)))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
real = '실제 가격'
predic = '예측 가격'
title = '주식 가격 예측'
date = '날짜'
endval = '종가'

# 예측값과 실제값을 비교하여 그래프로 표시
def plot_predictions(test_data, predicted_data):
    plt.plot(test_data, label=real)
    plt.plot(predicted_data, label=predic)
    plt.title(title)
    plt.xlabel(date)
    plt.ylabel(endval)
    plt.title(stock_name[stock])
    plt.legend()
    plt.show()

# 예측 실행
def predict_stock_price(stock_code, start_date, end_date, look_back=1, epochs=50):
    # 주식 데이터 가져오기
    stock_data = get_stock_data(stock_code, start_date, end_date)
    # 데이터 전처리
    scaled_data, scaler = preprocess_data(stock_data)
    # 시퀀스 데이터 생성
    x, y = create_sequences(scaled_data, look_back)
    # 훈련 데이터와 테스트 데이터로 분할
    train_size = int(len(x) * 0.80)
    test_size = len(x) - train_size
    x_train, x_test = x[0:train_size], x[train_size:len(x)]
    y_train, y_test = y[0:train_size], y[train_size:len(y)]
    # LSTM 모델 빌드
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
    model = build_lstm_model(look_back)
    # 모델 훈련
    model.fit(x_train, y_train, epochs=epochs, batch_size=1, verbose=2)
    # 테스트 데이터로 예측
    test_predict = model.predict(x_test)
    test_predict = scaler.inverse_transform(test_predict)
    y_test = scaler.inverse_transform([y_test])
    # 예측 결과 출력
    rmse = np.sqrt(mean_squared_error(y_test[0], test_predict[:, 0]))
    print(f'평균 제곱근 오차(RMSE): {rmse}')
    # 예측 결과 시각화
    plot_predictions(y_test[0], test_predict[:, 0])

#주식 가격 예측
stock_list = {'1':'AAPL','2':'NVDA','3':'005930.KS'}
stock_name = {'1':'APPLE','2':'NVIDIA','3':'삼성전자'}
stock = input('''주식 가격 예측
1. 애플
2. NVIDIA
3. 삼성전자

''')
predict_stock_price(stock_list[stock], '2023-01-01', '2023-11-06', look_back=5, epochs=50)
