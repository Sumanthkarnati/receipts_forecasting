import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf


data = pd.read_csv('data_daily.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)


def plot_time_series():
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Receipt_Count'], label='Receipt Count')
    plt.title('Daily Scanned Receipts in 2021')
    plt.xlabel('Date')
    plt.ylabel('Receipt Count')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    st.pyplot()


def plot_acf():
    lag_acf = acf(data['Receipt_Count'], nlags=40, fft=False)
    plt.figure(figsize=(12, 6))
    plt.plot(lag_acf, marker='o')
    plt.axhline(y=0, linestyle='--', color='gray')
    plt.axhline(
        y=-1.96/np.sqrt(len(data['Receipt_Count'])), linestyle='--', color='gray')
    plt.axhline(
        y=1.96/np.sqrt(len(data['Receipt_Count'])), linestyle='--', color='gray')
    plt.title('Autocorrelation Function (ACF)')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.tight_layout()
    st.pyplot()


def plot_forecasted_2022():
    plt.figure(figsize=(12, 6))
    plt.bar(forecasted_2022.keys(), forecasted_2022.values(), color='teal')
    plt.title('Forecasted Scanned Receipts in 2022')
    plt.xlabel('Month')
    plt.ylabel('Receipt Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot()


def main():
    st.title("Scanned Receipts Forecasting App")

    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Go to", ["Home", "Time Series Analysis", "Forecasting", "New Data Prediction"])

    if page == "Home":
        st.header("Welcome!")
        st.write("This app allows you to view the time series analysis, forecasting results, and make new predictions for the number of scanned receipts.")

    elif page == "Time Series Analysis":
        st.header("Time Series Analysis of Daily Scanned Receipts in 2021")
        st.write(
            "This section provides a visual analysis of the daily scanned receipts data for the year 2021.")
        plot_time_series()
        st.write(
            "Autocorrelation Function (ACF) is used to identify the seasonality in the time series data.")
        plot_acf()

    elif page == "Forecasting":
        st.header("Forecasting Scanned Receipts for 2022")
        st.write("This section shows the forecasted monthly total of scanned receipts for the year 2022 using an Exponential Smoothing model.")
        plot_forecasted_2022()

        month = st.selectbox(
            "Select a Month to View Forecasted Receipts", list(forecasted_2022.keys()))
        st.write(
            f"Forecasted Receipt Count for {month} 2022: {forecasted_2022[month]}")

        st.write("Optimal Parameters:")
        st.write(f"Alpha (Level Smoothing Parameter): 0.19")
        st.write(f"Beta (Trend Smoothing Parameter): 0.13")

    elif page == "New Data Prediction":
        st.header("Predict New Data")
        st.write("You can enter new data here and get the forecasted receipts.")


if __name__ == "__main__":
    main()
