import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# --- Load Data ---
@st.cache_data
def load_benchmark_data():
    return pd.read_csv("/Users/sanchita/Desktop/Assignment_1/benchmark_results.csv")

@st.cache_data
def load_prediction_data():
    return pd.read_parquet("/Users/sanchita/Desktop/Assignment_1/model_predictions.parquet")

# --- STREAMLIT APP ---
st.title("📊 Stock Price Prediction Dashboard")

# --- SECTION A: Benchmark Results ---
st.header("⏱️ Benchmark Results")
benchmark_data = load_benchmark_data()

# Filters
compression_filter = st.selectbox("Select Compression Type:", benchmark_data["Compression"].unique())
scale_filter = st.selectbox("Select Scale Factor:", benchmark_data["Scale Factor"].unique())

filtered_data = benchmark_data[
    (benchmark_data["Compression"] == compression_filter) &
    (benchmark_data["Scale Factor"] == scale_filter)
]

st.dataframe(filtered_data)

# Visualization - Read/Write Times
st.subheader("📈 Read/Write Times")
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(["CSV Read", "Parquet Read", "Parquet Write"], 
       [filtered_data["CSV Read Time (s)"].values[0],
        filtered_data["Parquet Read Time (s)"].values[0],
        filtered_data["Parquet Write Time (s)"].values[0]])

ax.set_ylabel("Time (s)")
st.pyplot(fig)

# --- SECTION B: Stock Price Predictions ---
st.header("💹 Stock Price Predictions")

# Load Model Predictions
prediction_data = load_prediction_data()

# Dropdown for Model Selection
model_choice = st.selectbox("Select Prediction Model:", ["LightGBM", "XGBoost", "Linear Regression"])

# Map the selected model to its respective column in the prediction file
model_mapping = {
    "LightGBM": "LightGBM_Predicted",
    "XGBoost": "XGBoost_Predicted",
    "Linear Regression": "LinearRegression_Predicted"
}

selected_model_column = model_mapping[model_choice]

# Visualization - Predicted vs Actual
st.subheader(f"📊 {model_choice} Predictions vs Actual Prices")

plt.figure(figsize=(10, 6))
plt.plot(prediction_data["Actual"], label="Actual", alpha=0.8)
plt.plot(prediction_data[selected_model_column], label=f"{model_choice} Prediction", alpha=0.8)
plt.title(f"{model_choice} Predictions vs Actual Prices")
plt.legend()
st.pyplot(plt)

# --- Evaluation Metrics ---
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(prediction_data["Actual"], prediction_data[selected_model_column])
rmse = mean_squared_error(prediction_data["Actual"], prediction_data[selected_model_column], squared=False)
r2 = r2_score(prediction_data["Actual"], prediction_data[selected_model_column])

st.subheader(f"📋 {model_choice} Performance Metrics")
st.write(f"**MAE:** {mae:.4f}")
st.write(f"**RMSE:** {rmse:.4f}")
st.write(f"**R² Score:** {r2:.4f}")
