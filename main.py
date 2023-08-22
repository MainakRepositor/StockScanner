import numpy as np
import pandas as pd

# For reading stock data from yahoo
import yfinance as yf
from datetime import datetime
from pandas_datareader import data as pdr
yf.pdr_override()

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, LSTM
 
import webbrowser
import streamlit as st
st.set_page_config(layout="wide")

print("**************************************************")
st.title('STOCK TREND PREDICTION')


# The tech stocks we'll use for this analysis
stock = st.text_input('ENTER A STOCK CODE', 'TSLA')
st.markdown('You can various find Stock Symbols at [Yahoo Finance](https://finance.yahoo.com/trending-tickers)')
st.markdown("")

col1, col2 = st.columns(2)
with col1:
    start = st.date_input("START DATE: ", datetime(2000, 1, 1))
with col2:
    end = st.date_input("END DATE: ", datetime.now())

# Set up End and Start times for data grab
try:
    df = yf.download(stock, start, end)

    print(type(df), df.empty)
    if df.empty: raise Exception("##### NO SUCH STOCK SYMBOL FOUND! TRY ENTERING THE CORRECT STOCK SYMBOL.")
    st.subheader('STOCK PRICE DATA')
    st.table(df.head())
    st.table(df.tail())

    if st.button("Analyze"):
            # Summary Stats
            st.subheader('DESCRIPTIVE STATISTICS ABOUT THE DATA')
            st.table(df.describe())

            # Let's see a historical view of the closing price
            st.subheader('CLOSING PRICE GRAPH')
            st.line_chart(df['Adj Close'])
            print("PLOTTED CLOSING GRAPH")

            # Now let's plot the total volume of stock being traded each day
            st.subheader('SALES VOLUME GRAPH')
            st.line_chart(df['Volume'])
            print("PLOTTED VOLUME GRAPH")


            st.subheader('MOVING AVERAGES OF CLOSING PRICE')

            ma_day = [10, 20, 50]
            for ma in ma_day:
                column_name = f"MA for {ma} days"
                df[column_name] = df['Adj Close'].rolling(ma).mean()
                            
            col1, col2, col3 = st.columns(3, gap="large")

            col1.markdown(f'{ma_day[0]} DAYS MA')
            col1.line_chart(df[[f'MA for {ma_day[0]} days']])

            col2.markdown(f'{ma_day[1]} DAYS MA')
            col2.line_chart(df[[f'MA for {ma_day[1]} days']])

            col3.markdown(f'{ma_day[2]} DAYS MA')
            col3.line_chart(df[[f'MA for {ma_day[2]} days']])

            print("PLOTTED MA GRAPHS")

            # We'll use pct_change to find the percent change for each day
            st.subheader('DAILY RETURNS')
            df['Daily Return'] = df['Adj Close'].pct_change()

            col1, col2 = st.columns(2)

            col1.markdown('TIMELINE PLOT')
            col1.line_chart(df[['Daily Return']])

            col2.markdown('HISTOGRAM')
            col2.bar_chart(df[['Daily Return']])
            print("PLOTTED DAILY RETURN GRAPHS")


            with st.spinner('PREPROCESSING THE DATA'):
                # Get the stock quote
                df = pdr.get_data_yahoo('AAPL', start=start, end=datetime.now())
                # Create a new dataframe with only the 'Close column 
                data = df.filter(['Close'])
                # Convert the dataframe to a numpy array
                dataset = data.values
                # Get the number of rows to train the model on
                training_data_len = int(np.ceil( len(dataset) * .95 ))

                scaler = MinMaxScaler(feature_range=(0,1))
                scaled_data = scaler.fit_transform(dataset)

                # Create the training data set 
                # Create the scaled training data set
                train_data = scaled_data[0:int(training_data_len), :]
                # Split the data into x_train and y_train data sets
                x_train = []
                y_train = []

                for i in range(60, len(train_data)):
                    x_train.append(train_data[i-60:i, 0])
                    y_train.append(train_data[i, 0])
                
                # Convert the x_train and y_train to numpy arrays 
                x_train, y_train = np.array(x_train), np.array(y_train)

                # Reshape the data
                x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

            st.checkbox("PREPROCESSED THE DATA", value=True, disabled=True)

            with st.spinner('TRAINING THE DEEP LEARNING MODEL (MIGHT TAKE 15-20 SECONDS)'):
                # Build the LSTM model
                model = Sequential()
                model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
                model.add(LSTM(64, return_sequences=False))
                model.add(Dense(25))
                model.add(Dense(1))

                # Compile the model
                model.compile(optimizer='adam', loss='mean_squared_error')

                # Train the model
                model.fit(x_train, y_train, epochs=1)

            st.checkbox("TRAINING THE DEEP LEARNING MODEL", value=True, disabled=True)

            with st.spinner('TESTING THE DEEP LEARNING MODEL'):
                test_data = scaled_data[training_data_len - 60: , :]
                # Create the data sets x_test and y_test
                x_test = []
                y_test = dataset[training_data_len:, :]
                for i in range(60, len(test_data)):
                    x_test.append(test_data[i-60:i, 0])
                    
                # Convert the data to a numpy array
                x_test = np.array(x_test)

                # Reshape the data
                x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

            st.checkbox("TESTING THE DEEP LEARNING MODEL", value=True, disabled=True)

            with st.spinner('PREDICTING THE CLOSING PRICE'):
                # Get the models predicted price values 
                predictions = model.predict(x_test)
                predictions = scaler.inverse_transform(predictions)

                # Get the root mean squared error (RMSE)
                rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
            
            st.subheader('PREDICTING THE CLOSING PRICE')
            
            # Plot the data
            train = data.iloc[5300:training_data_len, :]
            valid = data.iloc[training_data_len:, :]
            valid['Predictions'] = predictions

            # Visualize the data
            data = pd.DataFrame({'Train':train['Close'], 'Val':valid['Close'], 'Predictions':valid['Predictions']})
            st.line_chart(data)
            print("DONE!")
    
except Exception as e:
    st.subheader("")
    st.markdown(e)


