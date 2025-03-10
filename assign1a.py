# %% [markdown]
# ## Import Libraries
import pandas as pd
import polars as pl
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# %% [markdown]
# ## Load Dataset Functions
def load_pandas_data(file_path):
    return pd.read_parquet(file_path)

def load_polars_data(file_path):
    return pl.read_parquet(file_path)

# %% [markdown]
# ## Technical Indicators Function
def add_technical_indicators(df):
    df = df.copy()

    # Moving Average (SMA)
    df["SMA_10"] = df["close"].rolling(window=10, min_periods=1).mean()

    # Relative Strength Index (RSI)
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-10)  # Avoid division by zero
    df["RSI_14"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df["BB_Middle"] = df["close"].rolling(window=20, min_periods=1).mean()
    std_dev = df["close"].rolling(window=20, min_periods=1).std()
    df["BB_Upper"] = df["BB_Middle"] + (std_dev * 2)
    df["BB_Lower"] = df["BB_Middle"] - (std_dev * 2)

    # Moving Average Convergence Divergence (MACD)
    short_ema = df["close"].ewm(span=12, adjust=False).mean()
    long_ema = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = short_ema - long_ema

    return df.dropna()

# %% [markdown]
# ## Load Data and Apply Technical Indicators
parquet_file = "/Users/sanchita/Desktop/Assignment_1/all_stocks_5yr.parquet"
df_pandas = load_pandas_data(parquet_file)
df_pandas = add_technical_indicators(df_pandas)

# %% [markdown]
# ## Feature Engineering
features = ["SMA_10", "RSI_14", "BB_Middle", "BB_Upper", "BB_Lower", "MACD"]
target = "close"

X = df_pandas[features]
y = df_pandas[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# ## LightGBM Model - Fast & Efficient
lgb_model = lgb.LGBMRegressor(
    boosting_type='gbdt',
    n_estimators=500,
    learning_rate=0.1,
    max_depth=-1,
    num_leaves=32,
    n_jobs=-1,
    random_state=42
)

print("Training LightGBM...")
lgb_model.fit(X_train, y_train)

# %% [markdown]
# ## XGBoost Model
xgb_model = xgb.XGBRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    n_jobs=-1,
    random_state=42
)

print("Training XGBoost...")
xgb_model.fit(X_train, y_train)

# %% [markdown]
# ## Linear Regression Model
lr_model = LinearRegression()

print("Training Linear Regression...")
lr_model.fit(X_train, y_train)

# %% [markdown]
# ## Evaluation
y_pred_lgb = lgb_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)
y_pred_lr = lr_model.predict(X_test)

mae_lgb = mean_absolute_error(y_test, y_pred_lgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
mae_lr = mean_absolute_error(y_test, y_pred_lr)

# Display Model Performance
print("\nModel Performance (Mean Absolute Error):")
print(f"LightGBM: {mae_lgb:.4f}")
print(f"XGBoost: {mae_xgb:.4f}")
print(f"Linear Regression: {mae_lr:.4f}")

# %%
# Save Predictions for Streamlit
output_df = pd.DataFrame({
    "Actual": y_test.values,
    "LightGBM_Predicted": y_pred_lgb,
    "XGBoost_Predicted": y_pred_xgb,
    "LinearRegression_Predicted": y_pred_lr
})

# %%
# Save as CSV and Parquet for flexibility
output_csv = "/Users/sanchita/Desktop/Assignment_1/model_predictions.csv"
output_parquet = "/Users/sanchita/Desktop/Assignment_1/model_predictions.parquet"

output_df.to_csv(output_csv, index=False)
output_df.to_parquet(output_parquet, index=False)

print(f"Predictions saved successfully:\n- CSV: {output_csv}\n- Parquet: {output_parquet}")

# %%
