import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor
import warnings

# Suppress ConvergenceWarning from PoissonRegressor if it's not a critical issue
warnings.filterwarnings("ignore", category=UserWarning,
                        module="sklearn.linear_model._glm.glm")

# --- START OF SCRIPT ---
print("🚀 Sales Prediction AI Model - Weather Impact Analysis")
print("=" * 60)

# 📊 Step 1: Loading and Processing Data...
print("\n📊 Step 1: Loading and Processing Data...")
try:
    merged_df = pd.read_csv("merged_data.csv")
    initial_shape = merged_df.shape

    # Data Cleaning
    # Ensure all relevant columns for feature engineering are handled in dropna
    merged_df.dropna(subset=['sales_count', 'temperature_avg', 'precipitation',
                     'humidity', 'weather_condition', 'item', 'date'], inplace=True)

    # Convert 'date' column to datetime objects
    merged_df['date'] = pd.to_datetime(merged_df['date'])

    # Get date range
    min_date = merged_df['date'].min().strftime('%Y-%m-%d')
    max_date = merged_df['date'].max().strftime('%Y-%m-%d')

    print(f"✅ Data loaded successfully! Shape: {initial_shape}")
    print(f"📅 Date range: {min_date} to {max_date}")

except FileNotFoundError:
    print("Error: merged_data.csv not found. Please ensure the file is in the same directory.")
    exit()
except Exception as e:
    print(f"An error occurred during data loading: {e}")
    exit()

# 📈 Step 2: Exploratory Data Analysis...
print("\n📈 Step 2: Exploratory Data Analysis...")

# Sales Statistics
avg_sales = merged_df['sales_count'].mean()
min_sales = merged_df['sales_count'].min()
max_sales = merged_df['sales_count'].max()
std_sales = merged_df['sales_count'].std()

print("\n📊 Sales Statistics:")
print(f"Average daily sales: {avg_sales:.1f}")
print(f"Sales range: {min_sales} - {max_sales}")
print(f"Standard deviation: {std_sales:.1f}")

# Sales by Weather Condition
sales_by_weather = merged_df.groupby('weather_condition')[
    'sales_count'].agg(['mean', 'count'])
print("\n🌤 Sales by Weather Condition:")
# Using to_string() for better console formatting
print(sales_by_weather.to_string())

# 🔧 Step 3: Feature Engineering...
print("\n🔧 Step 3: Feature Engineering...")

# Feature Engineering
merged_df['day_of_week'] = merged_df['date'].dt.day_name()
merged_df['is_weekend'] = (merged_df['date'].dt.weekday >= 5).astype(int)
# NEW: Extract month as integer
merged_df['month'] = merged_df['date'].dt.month

# One-hot encoding for categorical columns
weather_dummies = pd.get_dummies(
    merged_df['weather_condition'], prefix='weather')
day_dummies = pd.get_dummies(merged_df['day_of_week'], prefix='day')
# NEW: One-hot encode 'item'
item_dummies = pd.get_dummies(merged_df['item'], prefix='item')
# NEW: One-hot encode 'month'
month_dummies = pd.get_dummies(merged_df['month'], prefix='month')

# Final feature set
# Combine all new and old features. 'sales_count' is the target variable.
feature_df = pd.concat([
    merged_df[['sales_count', 'temperature_avg',
               'precipitation', 'humidity', 'is_weekend']],
    weather_dummies,
    day_dummies,
    item_dummies,  # Include item dummies
    month_dummies  # Include month dummies
], axis=1)

# Store all feature column names (excluding 'sales_count') for later use in prediction
all_features_columns = feature_df.drop('sales_count', axis=1).columns
total_features = len(all_features_columns)

print(f"✅ Features created! Total features: {total_features}")

# ⚙ Step 4: Data Preprocessing...
print("\n⚙ Step 4: Data Preprocessing...")

# Prepare features and target
X = feature_df.drop('sales_count', axis=1).values
y = feature_df['sales_count'].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42)

print("✅ Data preprocessed!")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# 🤖 Step 5: Building AI Models...
print("\n🤖 Step 5: Building AI Models...")
print("Training Poisson Regression Model...")

# Train using Poisson Regression
# Added a small alpha for regularization (L2 penalty) to help with generalization.
# Increased max_iter to ensure convergence for more complex models.
model = PoissonRegressor(alpha=0.1, max_iter=1000)
model.fit(X_train, y_train)
print("✅ Poisson Regression Model trained.")

# 📊 Step 6: Model Performance Comparison (Adapted for single model)
print("\n📊 Step 6: Model Performance Comparison")
print("=" * 60)

# Predict on test set
y_pred = model.predict(X_test)

# Metrics (provided by user, added robustness for R2)


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def r2(y_true, y_pred):
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    # Handle case where y_true are all identical (no variance)
    if ss_total == 0:
        # If y_true are all same, R2 is 1 if prediction is perfect, else 0.
        return 1.0 if np.all(y_true == y_pred) else 0.0
    return 1 - np.sum((y_true - y_pred) ** 2) / ss_total


# Calculate metrics
train_mae = mae(y_train, model.predict(X_train))
test_mae = mae(y_test, y_pred)
train_rmse = rmse(y_train, model.predict(X_train))
test_rmse = rmse(y_test, y_pred)
train_r2 = r2(y_train, model.predict(X_train))
test_r2 = r2(y_test, y_pred)

# Print header for the performance table
print(f"{'Model':<20} {'Train MAE':>10} {'Test MAE':>10} {'Train RMSE':>12} {'Test RMSE':>12} {'Train R²':>10} {'Test R²':>10}")
print("-" * 90)
# Print the single model's performance
print(f"{'Poisson Regression':<20} {train_mae:>10.2f} {test_mae:>10.2f} {train_rmse:>12.2f} {test_rmse:>12.2f} {train_r2:>10.3f} {test_r2:>10.3f}")
print("-" * 90)

print("\n🏆 Best Model: Poisson Regression")
print(f"Test MAE: {test_mae:.2f}")
print(f"Test R²: {test_r2:.3f}")

# 🎯 Step 7: Model Predictions and Business Recommendations
print("\n🎯 Step 7: Model Predictions and Business Recommendations")
print("=" * 60)

print("\n📥 Enter Weather Conditions for Prediction:")

# Input validation for temperature
while True:
    try:
        user_temp = float(input("🌡 Enter average temperature (°C): "))
        if user_temp > 65:
            print(
                "❌ Error: Average temperature cannot exceed 65°C. Please enter a lower value.")
        elif user_temp < 0:  # Adding a lower bound for realism, assuming positive temps
            print("❌ Error: Temperature cannot be negative. Please enter a valid value.")
        else:
            break
    except ValueError:
        print("❌ Invalid input! Please enter a numerical value for temperature.")

# Input validation for precipitation
while True:
    try:
        user_precip = float(
            input("🌧 Enter precipitation level (e.g., 0.0 for none): "))
        if user_precip < 0:
            print(
                "❌ Error: Precipitation level cannot be negative. Please enter 0.0 or a positive value.")
        else:
            break
    except ValueError:
        print("❌ Invalid input! Please enter a numerical value for precipitation.")

# Input validation for humidity
while True:
    try:
        user_humidity = float(input("💧 Enter humidity level (0-100): "))
        if not (0 <= user_humidity <= 100):
            print(
                "❌ Error: Humidity level must be between 0 and 100. Please enter a valid value.")
        else:
            break
    except ValueError:
        print("❌ Invalid input! Please enter a numerical value for humidity.")

# Categorical inputs (no strict numerical validation needed, but formatting)
user_weather_input = input(
    "⛅ Enter weather condition (e.g., Sunny, Partly Cloudy, Cloudy): ").strip().title()
user_day_input = input(
    "📅 Enter day of the week (e.g., Monday): ").strip().capitalize()

# Input validation for month
while True:
    try:
        user_month_input = int(
            input("🗓 Enter month (e.g., 4 for April, 7 for July): "))
        if not (1 <= user_month_input <= 12):
            print(
                "❌ Error: Month must be an integer between 1 and 12. Please enter a valid month.")
        else:
            break
    except ValueError:
        print("❌ Invalid input! Please enter an integer for the month.")

user_coffee_type_input = input(
    "☕ Coffee type? (Cold Coffee/Hot Coffee): ").strip()

# Prepare user input vector for prediction
user_is_weekend = 1 if user_day_input in ['Saturday', 'Sunday'] else 0

# Create a DataFrame to hold the user's input, initialized with zeros for all possible feature columns
# This ensures the input vector matches the training data's column order and presence of dummies
user_data_processed = pd.DataFrame(0, index=[0], columns=all_features_columns)

# Populate continuous numerical features
user_data_processed['temperature_avg'] = user_temp
user_data_processed['precipitation'] = user_precip
user_data_processed['humidity'] = user_humidity
user_data_processed['is_weekend'] = user_is_weekend

# Populate one-hot encoded features based on user input
# Robustly check if the column exists for weather, day, item, month
weather_col_name = f"weather_{user_weather_input}"
if weather_col_name in all_features_columns:
    user_data_processed[weather_col_name] = 1
else:
    print(
        f"⚠ Warning: Weather condition '{user_weather_input}' not recognized by the model. Setting its feature to 0. Prediction may be less accurate.")

day_col_name = f"day_{user_day_input}"
if day_col_name in all_features_columns:
    user_data_processed[day_col_name] = 1
else:
    print(
        f"⚠ Warning: Day of week '{user_day_input}' not recognized by the model. Setting its feature to 0. Prediction may be less accurate.")

item_col_name = f"item_{user_coffee_type_input}"
if item_col_name in all_features_columns:
    user_data_processed[item_col_name] = 1
else:
    print(f"❌ Error: Coffee type '{user_coffee_type_input}' is a critical feature and not recognized by the model. Cannot make a reliable prediction. Please enter 'Cold Coffee' or 'Hot Coffee'.")
    exit()  # Exit if critical feature is missing

month_col_name = f"month_{user_month_input}"
if month_col_name in all_features_columns:
    user_data_processed[month_col_name] = 1
else:
    print(f"⚠ Warning: Month '{user_month_input}' not recognized by the model (e.g., if only limited months were in training data). Setting its feature to 0. Prediction may be less accurate.")


# Convert the processed user input DataFrame to a numpy array for scaling and prediction
user_features_raw = user_data_processed.values

# Standardize user features using the same scaler fitted on the training data
user_features_scaled = scaler.transform(user_features_raw)

# Predict sales count
user_pred = model.predict(user_features_scaled)

print(f"\n📊 Prediction Results for {user_coffee_type_input}:")
print(f"Most likely sales: {int(np.round(user_pred[0]))} cups")

# Note to clarify about "Bayesian" output not being directly available from sklearn's PoissonRegressor
print("\n--- Note on Credible Intervals and Probabilities ---")
print("The scikit-learn Poisson Regression model provides a point estimate.")
print("To obtain '90% Credible Intervals' or 'Probability of selling >X cups',")
print("a dedicated Bayesian modeling approach (e.g., using PyMC3/CmdStanPy) would be required.")
print("This goes beyond the scope of the current PoissonRegressor implementation.")
# --- END OF SCRIPT ---
