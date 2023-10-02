import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
# Replace with actual module and function names
from algorithms import lstm_model, gradient_boosting_model, ets_model, predict_lstm, predict_gb, predict_ets
from tensorflow.keras.models import load_model
import joblib
import os

# Load Data


@st.cache
def load_data(filepath):
    data = pd.read_csv(filepath)
    data['# Date'] = pd.to_datetime(data['# Date'])
    return data

# Main Function to run the app


def main():
    st.title("Receipts Forecasting App")

    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to", ["Home", "Data Exploration", "Model Predictions", "Future Predictions"])

    if page == "Home":
        st.header("Welcome to Receipts Forecasting App!")
        st.write("Navigate to different pages using the sidebar.")

    if page == "Data Exploration":
        st.header("Data Exploration")

        # Load and display data
        data = load_data('data_daily.csv')
        st.write(data.head())

        # Display plots
        st.subheader("Time Series Plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=data, x='# Date', y='Receipt_Count', ax=ax)
        st.pyplot(fig)

    if page == "Model Predictions":
        st.header("Model Predictions")
        # Run models and display results

        try:
            model_lstm = load_model('lstm_model.h5')
            scaler_lstm = joblib.load('lstm_scaler.pkl')
            mae_lstm = joblib.load('lstm_mae.pkl')
            rmse_lstm = joblib.load('lstm_rmse.pkl')
        except:
            mae_lstm, rmse_lstm, model_lstm, scaler_lstm = lstm_model(
                "data_daily.csv")

        st.write(f"LSTM Model - MAE: {mae_lstm}, RMSE: {rmse_lstm}")

        mae_gb, rmse_gb, model_gb = gradient_boosting_model(
            'data_daily.csv')
        st.write(f"Gradient Boosting Model - MAE: {mae_gb}, RMSE: {rmse_gb}")

        mae_ets, rmse_ets, model_ets = ets_model('data_daily.csv')
        st.write(f"ETS Model - MAE: {mae_ets}, RMSE: {rmse_ets}")

        if page == "Future Predictions":
            st.header("Future Predictions")

        dates_2022 = pd.date_range(
            start='2022-01-01', end='2022-12-31', freq='D')
        future_df = pd.DataFrame(dates_2022, columns=['Date'])

        # Assuming 'scaler' is the MinMaxScaler used during training
        future_df['LSTM_Predictions'] = predict_lstm(
            model_lstm, scaler_lstm, 365)
        future_df['GB_Predictions'] = predict_gb(model_gb, 365)
        future_df['ETS_Predictions'] = predict_ets(model_ets, 365)

        st.write(
            "Here you can view the models' predictions for future receipts in 2022.")
        st.write(future_df)

        csv = future_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="predictions_2022.csv">Download Predictions CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)


# Run the app
if __name__ == "__main__":
    main()
