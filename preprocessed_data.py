from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the merged data
df = pd.read_csv("C:/Users/Hafsa Rashid/Desktop/AiModel/merged_data.csv")

# View basic info
print(df.info())
print(df.head(100))

# One-hot encode item and weather_condition
df_encoded = pd.get_dummies(
    df, columns=['item', 'weather_condition'], drop_first=True)

print(df_encoded.columns)  # See all feature columns


# Columns to scale
scale_cols = ['temperature_avg', 'precipitation', 'humidity']

scaler = StandardScaler()
df_encoded[scale_cols] = scaler.fit_transform(df_encoded[scale_cols])

X = df_encoded.drop(columns=['sales_count', 'date'])  # Features
y = df_encoded['sales_count']                         # Target
