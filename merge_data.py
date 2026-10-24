
import pandas as pd

# 1. Load the two files
generation = pd.read_csv(
    "data/Plant_1_Generation_Data.csv"
)

weather = pd.read_csv(
    "data/Plant_1_Weather_Sensor_Data.csv"
)

# 2. Clean column names
generation.columns = generation.columns.str.strip()
weather.columns = weather.columns.str.strip()

# 3. Parse timestamps
generation["DATE_TIME"] = pd.to_datetime(
    generation["DATE_TIME"],
    dayfirst=True,
    errors="coerce"
)

weather["DATE_TIME"] = pd.to_datetime(
    weather["DATE_TIME"],
    dayfirst=True,
    errors="coerce"
)

# 4. Remove rows with invalid timestamps
generation = generation.dropna(subset=["DATE_TIME"])
weather = weather.dropna(subset=["DATE_TIME"])

# 5. Sum AC power across inverters
plant_power = (
    generation.groupby("DATE_TIME", as_index=False)
    ["AC_POWER"].sum()
)

# 6. Keep weather measurements for Plant 1
weather_columns = [
    "DATE_TIME",
    "AMBIENT_TEMPERATURE",
    "MODULE_TEMPERATURE",
    "IRRADIATION"
]

plant_weather = weather[weather_columns].copy()

# If several weather rows share a timestamp,
# average the sensor readings for that timestamp.
plant_weather = (
    plant_weather.groupby("DATE_TIME", as_index=False)
    .mean(numeric_only=True)
)

# 7. Merge on the exact timestamp
df = pd.merge(
    plant_weather,
    plant_power,
    on="DATE_TIME",
    how="inner",
    validate="one_to_one"
)

# 8. Add time features
df["HOUR"] = df["DATE_TIME"].dt.hour
df["DAY"] = df["DATE_TIME"].dt.day
df["MONTH"] = df["DATE_TIME"].dt.month

# 9. Sort and save
df = df.sort_values("DATE_TIME")

df.to_csv(
    "data/solar_plant1_merged.csv",
    index=False
)

# 10. Inspect the result
print("Merged shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate timestamps:",
      df["DATE_TIME"].duplicated().sum())

print("\nTime range:")
print(df["DATE_TIME"].min(), "to", df["DATE_TIME"].max())

print("\nSaved: data/solar_plant1_merged.csv")
