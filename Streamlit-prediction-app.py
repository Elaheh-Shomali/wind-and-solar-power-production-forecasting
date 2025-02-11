import datetime
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
import plotly.graph_objects as go

# Load trained models
with open('models/rnn_forecast_wind.sav', 'rb') as f:
    model_wind = pickle.load(f)

with open('models/rnn_forecast_solar.sav', 'rb') as f:
    model_solar = pickle.load(f)

# Load production data
Wind_PATH = './data/wind_renewables_production_cl.csv'
Solar_PATH = './data/solar_renewables_production_cl.csv'

production_wind = np.loadtxt(Wind_PATH, delimiter=",", skiprows=1, usecols=5, dtype=str)
production_solar = np.loadtxt(Solar_PATH, delimiter=",", skiprows=1, usecols=5, dtype=str)

production_wind = np.array([float(val) if val != '' else np.nan for val in production_wind])
production_solar = np.array([float(val) if val != '' else np.nan for val in production_solar])

# Configurations
WINDOW_SIZE = 20
START_DATETIME = datetime.datetime(2020, 4, 1, 0, 0)

st.title("Renewable Energy Production Forecast")
st.write("Enter a date and hour to forecast energy production for wind and solar.")

# User input
date_col, time_col = st.columns(2)
selected_date = date_col.date_input(
    "Select Date", 
    value=datetime.date(2021, 4, 20),
    min_value=datetime.date(2020, 4, 1),
    max_value=datetime.date(2023, 6, 30)
)
selected_time = time_col.time_input("Select Time", value=datetime.time(13, 0))

requested_datetime = datetime.datetime(
    year=selected_date.year,
    month=selected_date.month,
    day=selected_date.day,
    hour=selected_time.hour,
    minute=selected_time.minute
)

# Check if the input date is after June 30, 2023.
max_valid_datetime = datetime.datetime(2023, 6, 30, 23, 59, 59)
if requested_datetime > max_valid_datetime:
    st.error("The date is not available")
    st.stop()

st.write(f"Forecasting production for: {requested_datetime}")

time_delta = requested_datetime - START_DATETIME
requested_index = int(time_delta.total_seconds() // 3600)

st.write(f"Corresponding index in the series: {requested_index}")

# Forecasting functions for wind and solar
col1, col2 = st.columns(2)

def forecast_wind():
    if requested_index < WINDOW_SIZE:
        col1.error(f"Not enough historical data available to forecast for {requested_datetime}.")
    else:
        input_window = production_wind[requested_index - WINDOW_SIZE : requested_index]
        input_array = np.array(input_window).reshape(1, WINDOW_SIZE, 1)
        prediction = model_wind.predict(input_array)
        col1.success(f"Predicted wind production: {prediction[0][0]:.2f} MW")

def forecast_solar():
    if requested_index < WINDOW_SIZE:
        col2.error(f"Not enough historical data available to forecast for {requested_datetime}.")
    else:
        input_window = production_solar[requested_index - WINDOW_SIZE : requested_index]
        input_array = np.array(input_window).reshape(1, WINDOW_SIZE, 1)
        prediction = model_solar.predict(input_array)
        col2.success(f"Predicted solar production: {prediction[0][0]:.2f} MW")

with col1:
    st.subheader("Wind Energy Forecast")
    forecast_wind()

with col2:
    st.subheader("Solar Energy Forecast")
    forecast_solar()

# Interactive Forecast Plot using Plotly (displayed one below the other)
if st.button("Show Forecast Plots"):
    def plot_forecast_interactive(production_data, model, energy_type, forecast_color):
        # Prepare the dataset and get the forecast
        ds = tf.data.Dataset.from_tensor_slices(production_data)
        ds = ds.window(WINDOW_SIZE, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(WINDOW_SIZE))
        ds = ds.batch(32).prefetch(1)
        forecast = model.predict(ds).squeeze()

        # Generate corresponding date lists
        actual_dates = [START_DATETIME + datetime.timedelta(hours=i) for i in range(len(production_data))]
        forecast_dates = [START_DATETIME + datetime.timedelta(hours=i) for i in range(WINDOW_SIZE, WINDOW_SIZE + len(forecast))]

        # Build an interactive Plotly figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=actual_dates,
            y=production_data,
            mode='lines',
            name=f"Actual {energy_type} Production",
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast,
            mode='lines',
            name=f"Forecasted {energy_type} Production",
            line=dict(color=forecast_color, dash='dash', width=2)
        ))

        # Customize the layout: interactive zooming and panning are enabled by default
        fig.update_layout(
            title=f"Actual vs Forecasted {energy_type} Energy Production",
            xaxis_title="Date",
            yaxis_title="Production (MW)",
            xaxis=dict(rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1d", step="day", stepmode="backward"),
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(step="all")
                ])
            ), rangeslider=dict(visible=True), type="date")
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Wind Energy Forecast Plot")
    plot_forecast_interactive(production_wind, model_wind, "Wind", "red")
    
    st.subheader("Solar Energy Forecast Plot")
    plot_forecast_interactive(production_solar, model_solar, "Solar", "orange")
