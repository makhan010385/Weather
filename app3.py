import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib

# Load model and encoder
model = joblib.load("weather_multi_model.joblib")
le_location = joblib.load("weather_location_encoder.joblib")

st.title("⛅ 5-Day Weather Forecast from Historical Trends")

# User Input
location = st.text_input("Enter Location (case-sensitive)", "Jabalpur")
date_input = st.date_input("Select Starting Date", datetime.today())

# Check if location exists
if location not in le_location.classes_:
    st.error("❌ Location not found in historical data.")
else:
    location_code = le_location.transform([location])[0]
    start_date = pd.to_datetime(date_input)

    forecast = []

    for i in range(1, 6):
        future_date = start_date + timedelta(days=i)
        day_of_year = future_date.timetuple().tm_yday
        year = future_date.year
        input_vector = np.array([[location_code, day_of_year, year]])
        prediction = model.predict(input_vector)[0]

        forecast.append({
            "Date": future_date.strftime("%Y-%m-%d"),
            "Max_Temp (°C)": round(prediction[0], 2),
            "Min_Temp (°C)": round(prediction[1], 2),
            "Rainfall (mm)": round(prediction[2], 2),
            "Max_Humidity (%)": round(prediction[3], 2),
            "Min_Humidity (%)": round(prediction[4], 2)
        })

    # Show results
    forecast_df = pd.DataFrame(forecast)
    st.success("✅ Forecast Generated")
    st.dataframe(forecast_df)

    # Download option
    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Forecast CSV", data=csv, file_name="5_day_weather_forecast.csv")
