import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from algorithms import lstm_model, gradient_boosting_model, ets_model, predict_lstm, predict_gb, predict_ets
from tensorflow.keras.models import load_model
import joblib
import os

# Load Data


@st.cache
def load_data(filepath):
    data = pd.read_csv(filepath)
    data['Date'] = pd.to_datetime(data['Date'])
    return data


def main():
    st.title("Receipts Forecasting App")
    st.sidebar.header("Navigation")

    page = st.sidebar.radio(
        "Go to", ["Home", "Data Exploration", "Model and Future Predictions"])

    if page == "Home":
        st.header("Welcome to Receipts Forecasting App!")
        st.write("Navigate to different pages using the sidebar.")

        st.subheader("About the Models")
        st.write("""
        - **LSTM Model:** 
            * Type: Recurrent Neural Network.
            * Suitable For: Time series forecasting, especially when order dependence in the sequence is crucial.
            * How it Works: Capable of learning long-term dependencies using mechanisms called gates, allowing it to remember or forget information over long sequences.
            * [Learn More](https://en.wikipedia.org/wiki/Long_short-term_memory)
            
        - **Gradient Boosting Model:** 
            * Type: Ensemble Learning Method.
            * Suitable For: Regression and classification problems.
            * How it Works: Builds trees one at a time; each tree helps to correct the mistakes of the previous ones.
            * [Learn More](https://en.wikipedia.org/wiki/Gradient_boosting)
            
        - **ETS Model:** 
            * Type: Exponential Smoothing State Space Model.
            * Suitable For: Univariate time series forecasting.
            * How it Works: Uses weighted averages of past observations with an exponentially decreasing weight to make forecasts.
            * [Learn More](https://otexts.com/fpp2/ets.html)
        """)

        st.subheader("Future Work")
        st.write("""
        - Refine existing models for enhanced prediction accuracy.
        - Incorporate advanced models like PatchTST from the research paper "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers."
        - Expand the dataset for improved model training and reliability.
        - Enable user interaction and feedback to understand user requirements and improve model performance.
        """)

    elif page == "Data Exploration":
        st.header("Data Exploration")
        data = load_data('data_daily.csv')
        st.dataframe(data.head())

        st.subheader("Time Series Plot")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=data, x='Date', y='Receipt_Count', ax=ax)
        st.pyplot(fig)

    elif page == "Model and Future Predictions":
        st.header("Model and Future Predictions")

        try:
            model_lstm = load_model('lstm_model.h5')
            scaler_lstm = joblib.load('lstm_scaler.pkl')
            mae_lstm = joblib.load('lstm_mae.pkl')
            rmse_lstm = joblib.load('lstm_rmse.pkl')
        except Exception as e:
            st.write(f"Error loading LSTM model: {e}")
            mae_lstm, rmse_lstm, model_lstm, scaler_lstm = lstm_model(
                "data_daily.csv")

        st.subheader("Model Predictions")
        st.write(f"LSTM Model - MAE: {mae_lstm}, RMSE: {rmse_lstm}")
        mae_gb, rmse_gb, model_gb = gradient_boosting_model('data_daily.csv')
        st.write(f"Gradient Boosting Model - MAE: {mae_gb}, RMSE: {rmse_gb}")
        mae_ets, rmse_ets, model_ets = ets_model('data_daily.csv')
        st.write(f"ETS Model - MAE: {mae_ets}, RMSE: {rmse_ets}")

        st.subheader("Future Predictions")
        if os.path.exists('predictions_2022.csv'):
            future_df = pd.read_csv('predictions_2022.csv')
            st.dataframe(future_df)

            # Monthly Predictions
            future_df['Date'] = pd.to_datetime(future_df['Date'])
            future_df_monthly = future_df.resample(
                'M', on='Date').sum().reset_index()

            st.subheader("Monthly Predictions")

            csv = future_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="predictions_2022.csv">Download Predictions CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)

            # Plotting for yearly predictions
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=future_df, x='Date',
                         y='LSTM_Predictions', ax=ax, label='LSTM Predictions')
            sns.lineplot(data=future_df, x='Date', y='GB_Predictions',
                         ax=ax, label='Gradient Boosting Predictions')
            sns.lineplot(data=future_df, x='Date',
                         y='ETS_Predictions', ax=ax, label='ETS Predictions')
            plt.title('Yearly Predictions for 2022')
            plt.ylabel('Receipt_Count')
            plt.xlabel('Date')
            plt.legend()
            st.pyplot(fig)

            # Plotting
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(data=future_df_monthly, x='Date',
                         y='LSTM_Predictions', ax=ax, label='LSTM Predictions')
            sns.lineplot(data=future_df_monthly, x='Date', y='GB_Predictions',
                         ax=ax, label='Gradient Boosting Predictions')
            sns.lineplot(data=future_df_monthly, x='Date',
                         y='ETS_Predictions', ax=ax, label='ETS Predictions')
            plt.title('Monthly Predictions for 2022')
            plt.ylabel('Receipt_Count')
            plt.xlabel('Month')
            plt.legend()
            st.pyplot(fig)

        else:
            if st.button("Run Models"):
                dates_2022 = pd.date_range(
                    start='2022-01-01', end='2022-12-31', freq='D')
                future_df = pd.DataFrame(dates_2022, columns=['Date'])

                future_df['LSTM_Predictions'] = predict_lstm(
                    model_lstm, scaler_lstm, 365)
                future_df['GB_Predictions'] = predict_gb(model_gb, 365)
                future_df['ETS_Predictions'] = predict_ets(model_ets, 365)

                future_df.to_csv('predictions_2022.csv', index=False)

                st.dataframe(future_df)

                # Monthly Predictions
                future_df['Date'] = pd.to_datetime(future_df['Date'])
                future_df_monthly = future_df.resample(
                    'M', on='Date').sum().reset_index()

                st.subheader("Monthly Predictions")

                csv = future_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="predictions_2022.csv">Download Predictions CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)

                # Plotting for yearly predictions
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.lineplot(data=future_df, x='Date',
                             y='LSTM_Predictions', ax=ax, label='LSTM Predictions')
                sns.lineplot(data=future_df, x='Date', y='GB_Predictions',
                             ax=ax, label='Gradient Boosting Predictions')
                sns.lineplot(data=future_df, x='Date',
                             y='ETS_Predictions', ax=ax, label='ETS Predictions')
                plt.title('Yearly Predictions for 2022')
                plt.ylabel('Receipt_Count')
                plt.xlabel('Date')
                plt.legend()
                st.pyplot(fig)

                # Plotting
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.lineplot(data=future_df_monthly, x='Date',
                             y='LSTM_Predictions', ax=ax, label='LSTM Predictions')
                sns.lineplot(data=future_df_monthly, x='Date', y='GB_Predictions',
                             ax=ax, label='Gradient Boosting Predictions')
                sns.lineplot(data=future_df_monthly, x='Date',
                             y='ETS_Predictions', ax=ax, label='ETS Predictions')
                plt.title('Monthly Predictions for 2022')
                plt.ylabel('Receipt_Count')
                plt.xlabel('Month')
                plt.legend()
                st.pyplot(fig)


if __name__ == "__main__":
    main()
