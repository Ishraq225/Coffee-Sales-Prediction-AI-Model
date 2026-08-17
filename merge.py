import pandas as pd

# Use full paths
sales_df = pd.read_csv(
    "C:/Users/Hafsa Rashid/Desktop/AiModel/sales_receipts.csv")
weather_df = pd.read_csv(
    "C:/Users/Hafsa Rashid/Desktop/AiModel/weather_data.csv")

# Convert 'date' columns to datetime
sales_df['date'] = pd.to_datetime(sales_df['date'])
weather_df['date'] = pd.to_datetime(weather_df['date'])

# Merge on 'date'
merged_df = pd.merge(sales_df, weather_df, on='date')

print(merged_df.head(100))
merged_df.to_csv("merged_data.csv", index=False)
print("✅ Merged data saved to 'merged_data.csv'")
