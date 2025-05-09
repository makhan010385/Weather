import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data (must include: Location, Year, SMW, Max_Temp, etc.)
df = pd.read_csv("Model_CSV.csv")

# Drop rows with missing targets or location
df = df.dropna(subset=["Location", "Year", "SMW", "Max_Temp"])

# Convert SMW to DayOfYear approx. (assume 7 days per week)
df["DayOfYear"] = df["SMW"] * 7

# Encode Location
le_location = LabelEncoder()
df["Location_Code"] = le_location.fit_transform(df["Location"])

# Features and Targets
features = ["Location_Code", "DayOfYear", "Year"]
targets = ["Max_Temp", "Min_Temp", "Rainfall", "Max_Humidity", "Min_Humidity"]

X = df[features]
Y = df[targets].apply(pd.to_numeric, errors="coerce").fillna(0)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, Y)

# Save model and encoder
joblib.dump(model, "weather_multi_model.joblib")
joblib.dump(le_location, "weather_location_encoder.joblib")

print("✅ Model trained and saved successfully.")
