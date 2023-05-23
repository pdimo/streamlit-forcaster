import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from plotly import graph_objects as go


# Function to load data from a CSV file
def load_data(file):
    """Load data from a CSV file"""
    data = pd.read_csv(file, thousands=',')
    return data

# Function to resample time series data
def resample_data(data, time_column, metric_column, period):
    """
    Resamples the data based on the selected period.

    data: The input dataframe.
    time_column: The name of the column with time series data.
    metric_column: The name of the column with the metric to forecast.
    period: The period to resample the data. Can be 'D' (daily), 'W' (weekly), 'M' (monthly), or 'Y' (yearly).
    """
    # Convert the time column to datetime
    data[time_column] = pd.to_datetime(data[time_column])

    # Set the time column as the index
    data.set_index(time_column, inplace=True)

    # Resample the data
    data_resampled = data.resample(period).sum().reset_index()

    # Filter out rows where the metric_column is 0
    data_resampled = data_resampled[data_resampled[metric_column] != 0]

    # If the period is monthly or weekly, make sure the data starts from the first day of the month or week, respectively
    if period == 'M' or period == 'W' or period == 'Y':
        # Find the first date that is the start of a month or week
        first_valid_date = data_resampled[time_column].iloc[0]
        while first_valid_date != first_valid_date.to_period(period).to_timestamp():
            first_valid_date += pd.DateOffset(days=1)

        # Trim the DataFrame to start from the first valid date
        data_resampled = data_resampled[data_resampled[time_column] >= first_valid_date]

        # Find the last date that is the end of a month or week
        last_valid_date = data_resampled[time_column].iloc[-1]
        while last_valid_date != (last_valid_date + pd.offsets.MonthEnd(1)).to_period(period).to_timestamp() - pd.DateOffset(days=1):
            last_valid_date -= pd.DateOffset(days=1)

        # Trim the DataFrame to end at the last valid date
        data_resampled = data_resampled[data_resampled[time_column] <= last_valid_date]

        # Check if the last month or week is complete, if not, remove it
        if period == 'M':
            last_month = data_resampled[time_column].dt.to_period('M').iloc[-1]
            if (data_resampled[time_column].dt.to_period('M') == last_month).sum() < last_month.days_in_month:
                data_resampled = data_resampled[data_resampled[time_column].dt.to_period('M') != last_month]
        elif period == 'W':
            last_week = data_resampled[time_column].dt.to_period('W').iloc[-1]
            if (data_resampled[time_column].dt.to_period('W') == last_week).sum() < 1:
                data_resampled = data_resampled[data_resampled[time_column].dt.to_period('W') != last_week]

    # Keep only the selected columns
    data_resampled = data_resampled[[time_column, metric_column]]

    return data_resampled


# Function to fit and forecast data using Prophet model
def prophet_model(data, time_column, metric_column, seasonality_mode, changepoint_prior_scale, seasonality_prior_scale, holidays_prior_scale, mcmc_samples, forecasting_periods, period):
    """Fit and forecast data using Prophet model"""
    # Preparing data for Prophet model
    data = data[[time_column, metric_column]]
    data = data.rename(columns={time_column: 'ds', metric_column: 'y'})

    # Defining Prophet model
    model = Prophet(seasonality_mode=seasonality_mode, 
                    changepoint_prior_scale=changepoint_prior_scale,
                    seasonality_prior_scale=seasonality_prior_scale,
                    holidays_prior_scale=holidays_prior_scale,
                    mcmc_samples=mcmc_samples)

    # Fitting model
    model.fit(data)

    # Making future dataframe for specified periods
    if period == 'D':
        future = model.make_future_dataframe(periods=forecasting_periods)
    elif period == 'W':
        future = model.make_future_dataframe(periods=forecasting_periods*7)
    elif period == 'M':
        future = model.make_future_dataframe(periods=forecasting_periods*30)
    elif period == 'Y':
        future = model.make_future_dataframe(periods=forecasting_periods*365)

    # Forecasting
    forecast = model.predict(future)
    
    return forecast

# Function to visualize forecasted data
def visualize_forecast(forecast, data):
    fig = go.Figure()

    # Plot the original data
    fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], name='Original', line_color='deepskyblue'))

    # Plot the forecasted data
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Predicted', line=dict(color='red', dash='dash')))

    # Add confidence band
    fig.add_trace(go.Scatter(
        name='Confidence Interval',
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        mode='lines',
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        name='Confidence Interval',
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        mode='lines',
        line=dict(width=0),
        fillcolor='rgba(255, 0, 0, 0.2)',
        fill='tonexty',
        showlegend=False
    ))

    # Add vertical line
    fig.add_shape(type='line',
                  x0=data['ds'].iloc[-1], y0=0, # the line will start at the last date of the original data
                  x1=data['ds'].iloc[-1], y1=1, # and will end at the last date of the original data
                  yref='paper', # this makes the y coordinates be in relative coordinates (0 to 1 covers the whole y range)
                  line=dict(color='green', dash='dash'))

    # Reorder data so that original and predicted data appear on top of bounds
    fig.data = fig.data[-2:] + fig.data[:-2]

    return fig


def calculate_inverse_mape(data, forecast):
    """Calculate Inverse Mean Absolute Percentage Error (Inverse MAPE)"""
    y_true = data['y']
    y_pred = forecast['yhat'][:len(data)]
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    inverse_mape = 100 - mape
    return inverse_mape


def main():
    st.title("Time Series Forecasting")

    file = st.sidebar.file_uploader("Upload Time Series Data", type=['csv', 'xlsx'])

    if file is not None:
        data = load_data(file)

        st.sidebar.write("Data Loaded Successfully!")
        #st.dataframe(data)

        time_column = st.sidebar.selectbox("Select Time Series Column", data.columns, help="This column should contain the time series data.")
        metric_column = st.sidebar.selectbox("Select Metric Column", data.columns, help="This column should contain the metric to forecast.")

        period = st.sidebar.selectbox("Select Resampling Period", ('D', 'W', 'M', 'Y'), 
                              format_func=lambda x: {'D':'Daily', 'W':'Weekly', 'M':'Monthly', 'Y':'Yearly'}[x], 
                              help="The period to resample the data. Can be 'D' (daily), 'W' (weekly), 'M' (monthly), or 'Y' (yearly).")

        # Resample data based on selected period
        data = resample_data(data, time_column, metric_column, period)

        st.write("Data after resampling")
        st.dataframe(data)

        # forecasting_periods slider
        if period == 'D':
            forecasting_periods = st.sidebar.slider('Select Forecasting Interval (Days)', min_value=30, max_value=730, value=365, step=1)
        elif period == 'W':
            forecasting_periods = st.sidebar.slider('Select Forecasting Interval (Weeks)', min_value=4, max_value=104, value=52, step=1)
        elif period == 'M':
            forecasting_periods = st.sidebar.slider('Select Forecasting Interval (Months)', min_value=1, max_value=24, value=12, step=1)
        elif period == 'Y':
            forecasting_periods = st.sidebar.slider('Select Forecasting Interval (Years)', min_value=1, max_value=10, value=5, step=1)

        # User selects seasonality mode
        seasonality_mode = st.sidebar.selectbox(
            "Select Seasonality Mode", 
            ('additive', 'multiplicative'), 
            help="Choose 'additive' if the seasonal effect is consistent (e.g., sales increase by the same amount every December). Choose 'multiplicative' if the seasonal effect changes with the level of the time series (e.g., sales double every December)."
        )

        if st.sidebar.button('Run Model'):
            # Run Prophet model with default parameters
            forecast = prophet_model(data, time_column, metric_column, 
                                     seasonality_mode=seasonality_mode, 
                                     changepoint_prior_scale=0.05,
                                     seasonality_prior_scale=10.0,
                                     holidays_prior_scale=10.0,
                                     mcmc_samples=0,
                                     forecasting_periods=forecasting_periods,
                                     period=period)
            st.write("Forecasting Complete!")

            # Prepare original data for visualization
            viz_data = data[[time_column, metric_column]]
            viz_data.columns = ['ds', 'y']

           # Calculate Inverse MAPE
            inverse_mape = calculate_inverse_mape(viz_data, forecast)
            st.write(f"Forecast Accuracy (inverse MAPE): {inverse_mape:.2f}%")

            # Interpret Inverse MAPE
            if inverse_mape > 90:
                st.write("The forecast is highly accurate.")
            elif inverse_mape > 80:
                st.write("The forecast is reasonably accurate.")
            elif inverse_mape > 50:
                st.write("The forecast is somewhat accurate, but could be improved.")
            else:
                st.write("The forecast is not very accurate. Consider using a different model or adjusting the model parameters.")
            
            fig = visualize_forecast(forecast, viz_data)

            st.plotly_chart(fig)

if __name__ == '__main__':
    main()
