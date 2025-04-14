import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from keras.models import load_model
import yfinance as yf
import streamlit as st

# Streamlit Title
st.title("Stock Price Analysis Using LSTM, RF, SVM Models")

# 1. Enter Stock Ticker
ticker_symbol = st.text_input('Enter Stock Ticker:', " ")

if ticker_symbol:
    # Define the start and end dates
    start_date = "2000-01-01"
    end_date = "2025-02-27"

    # Fetch historical stock data
    stock_data = yf.download(ticker_symbol, start=start_date, end=end_date)

    if not stock_data.empty:
        
        # Calculate Moving Averages
        stock_data['100-Day MA'] = stock_data['Close'].rolling(window=100).mean()
        stock_data['200-Day MA'] = stock_data['Close'].rolling(window=200).mean()

        # 2. Plot Graph of 100 and 200 Days Moving Averages
        st.subheader(f"{ticker_symbol} Stock Prices with Moving Averages")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(stock_data['Close'], label='Close Price', color='blue')
        ax.plot(stock_data['100-Day MA'], label='100-Day MA', color='red')
        ax.plot(stock_data['200-Day MA'], label='200-Day MA', color='green')
        ax.set_title(f"{ticker_symbol} Stock Price with 100-Day and 200-Day Moving Averages")
        ax.legend()
        st.pyplot(fig)

        # Preprocess Data for Prediction
        scaler = MinMaxScaler(feature_range=(0, 1))
        data = stock_data['Close'].values.reshape(-1, 1)
        scaled_data = scaler.fit_transform(data)

        # Prepare Data for Prediction
        x_data = []
        y_data = scaled_data[100:]
        for i in range(100, len(scaled_data)):
            x_data.append(scaled_data[i-100:i])

        x_data = np.array(x_data).reshape(len(x_data), -1)
        y_data = np.array(y_data).flatten()

        # Split into Training and Testing Sets
        split_idx = int(len(x_data) * 0.8)
        x_train, x_test = x_data[:split_idx], x_data[split_idx:]
        y_train, y_test = y_data[:split_idx], y_data[split_idx:]

        # LSTM Model
        try:
            lstm_model = load_model('LSTM_model.keras')
            lstm_predictions = lstm_model.predict(x_test)
            lstm_predictions = scaler.inverse_transform(lstm_predictions.reshape(-1, 1))
            y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

            # LSTM Metrics
            lstm_mae = mean_absolute_error(y_test_actual, lstm_predictions)
            lstm_mse = mean_squared_error(y_test_actual, lstm_predictions)
            lstm_rmse = np.sqrt(lstm_mse)
            lstm_r2 = r2_score(y_test_actual, lstm_predictions)

            # Display LSTM Metrics
            st.subheader("LSTM Model Accuracy")
            st.write(f"Mean Absolute Error (MAE): {lstm_mae:.2f}")
            st.write(f"Mean Squared Error (MSE): {lstm_mse:.2f}")
            st.write(f"Root Mean Squared Error (RMSE): {lstm_rmse:.2f}")
            st.write(f"R-squared (R²) Score: {lstm_r2:.2f}")

            # Plot Actual vs Predicted Prices for LSTM
            st.subheader("Actual vs Predicted Stock Prices (LSTM)")
            fig_lstm, ax_lstm = plt.subplots(figsize=(10, 5))
            ax_lstm.plot(y_test_actual, label='Actual Price', color='blue')
            ax_lstm.plot(lstm_predictions, label='Predicted Price (LSTM)', color='red')
            ax_lstm.set_title(f"{ticker_symbol} Stock Price Prediction: LSTM")
            ax_lstm.legend()
            st.pyplot(fig_lstm)

        except Exception as e:
            st.error("LSTM model file not found or an error occurred during prediction.")
            st.error(e)

        # Random Forest Model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(x_train, y_train)
        rf_predictions = rf_model.predict(x_test)
        rf_predictions = scaler.inverse_transform(rf_predictions.reshape(-1, 1))

        # Random Forest Metrics
        rf_mae = mean_absolute_error(y_test_actual, rf_predictions)
        rf_mse = mean_squared_error(y_test_actual, rf_predictions)
        rf_rmse = np.sqrt(rf_mse)
        rf_r2 = r2_score(y_test_actual, rf_predictions)

        # Display Random Forest Metrics
        st.subheader("Random Forest Model Accuracy")
        st.write(f"Mean Absolute Error (MAE): {rf_mae:.2f}")
        st.write(f"Mean Squared Error (MSE): {rf_mse:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rf_rmse:.2f}")
        st.write(f"R-squared (R²) Score: {rf_r2:.2f}")

        # Plot Actual vs Predicted Prices for Random Forest
        st.subheader("Actual vs Predicted Stock Prices (Random Forest)")
        fig_rf, ax_rf = plt.subplots(figsize=(10, 5))
        ax_rf.plot(y_test_actual, label='Actual Price', color='blue')
        ax_rf.plot(rf_predictions, label='Predicted Price (RF)', color='orange')
        ax_rf.set_title(f"{ticker_symbol} Stock Price Prediction: Random Forest")
        ax_rf.legend()
        st.pyplot(fig_rf)

        # SVM Model
        svr_model = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
        svr_model.fit(x_train, y_train)
        svr_predictions = svr_model.predict(x_test)
        svr_predictions = scaler.inverse_transform(svr_predictions.reshape(-1, 1))

        # SVM Metrics
        svr_mae = mean_absolute_error(y_test_actual, svr_predictions)
        svr_mse = mean_squared_error(y_test_actual, svr_predictions)
        svr_rmse = np.sqrt(svr_mse)
        svr_r2 = r2_score(y_test_actual, svr_predictions)

        # Display SVM Metrics
        st.subheader("SVM Model Accuracy")
        st.write(f"Mean Absolute Error (MAE): {svr_mae:.2f}")
        st.write(f"Mean Squared Error (MSE): {svr_mse:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {svr_rmse:.2f}")
        st.write(f"R-squared (R²) Score: {svr_r2:.2f}")

        # Plot Actual vs Predicted Prices for SVM
        st.subheader("Actual vs Predicted Stock Prices (SVM)")
        fig_svr, ax_svr = plt.subplots(figsize=(10, 5))
        ax_svr.plot(y_test_actual, label='Actual Price', color='blue')
        ax_svr.plot(svr_predictions, label='Predicted Price (SVM)', color='green')
        ax_svr.set_title(f"{ticker_symbol} Stock Price Prediction: SVM")
        ax_svr.legend()
        st.pyplot(fig_svr)

    else:
        st.error("No data found for the entered ticker symbol. Please try another.")

