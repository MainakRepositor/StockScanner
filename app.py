import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import pandas_datareader as data

# from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# ********** #
st.title('STOCK TREND PREDICTION')

start = '2010-01-01'
end = '2020-12-31'
try:
    stock_input = st.text_input('ENTER A STOCK CODE', 'AAPL')
    st.markdown('Try adding different Stock Code. You can find Stock Code from Yahoo Finance')
    st.markdown('https://finance.yahoo.com/trending-tickers')
    df = data.DataReader(stock_input, 'yahoo', start, end)

    # DESCRIBING DATA
    st.subheader('DATA FROM 2010-2020')
    st.write(df.describe())

    # DATA VISUALIZATION
    st.subheader('CLOSING PRICE TIME CHART')
    fig = plt.figure(figsize=(16,8))
    plt.plot(df['Close'], 'b')
    st.pyplot(fig)

    st.subheader('WITH 100 DAYS MOVING AVERAGE')
    ma100 = df['Close'].rolling(100).mean()
    fig = plt.figure(figsize=(16,8))
    plt.plot(df['Close'], 'b')
    plt.plot(ma100, color='r')
    st.pyplot(fig)

    st.subheader('WITH 100 & 200 DAYS MOVING AVERAGE')
    ma100 = df['Close'].rolling(100).mean()
    ma200 = df['Close'].rolling(200).mean()
    fig = plt.figure(figsize=(16,8))
    plt.plot(df['Close'], 'b')
    plt.plot(ma100, color='r')
    plt.plot(ma200, color='g')
    st.pyplot(fig)


    # ************* #
    split = int(len(df)*0.70)

    train_df = pd.DataFrame(df['Close'][:split])
    test_df = pd.DataFrame(df['Close'][split:])

    scalar = MinMaxScaler(feature_range=(0,1))
    train_df_arr = scalar.fit_transform(train_df)

    X_train = []
    Y_train = []

    for i in range(100, train_df_arr.shape[0]):
        X_train.append(train_df_arr[i-100:i])
        Y_train.append(train_df_arr[i,0])

    X_train, Y_train = np.array(X_train), np.array(Y_train)
    model = pickle.load(open('model.pkl', 'rb'))

    past100_df = train_df.tail(100)
    test_df = past100_df.append(test_df, ignore_index=True)
    test_df_arr = scalar.fit_transform(test_df)

    X_test = []
    Y_test = []

    for i in range(100, test_df_arr.shape[0]):
        X_test.append(test_df_arr[i-100:i])
        Y_test.append(test_df_arr[i, 0])

    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    Y_pred = model.predict(X_test)
    scale = scalar.scale_[0]
    Y_test = Y_test / scale
    Y_pred = Y_pred / scale

    st.subheader('ORIGINAL VS PREDICTION')
    fig = plt.figure(figsize=(15,7))
    plt.plot(Y_test, color='b', label='Original')
    plt.plot(Y_pred, color='r', label='Predicted')
    plt.legend()
    st.pyplot(fig)

except Exception as e:
    st.title(e)
    st.markdown('You can find Stock Code from Yahoo Finance')