# Assignment1Results

CSIS 4260 – Assignment 1

Project Title: Stock Price Analysis and Prediction

Overview

This project is divided into three main parts, combining research, benchmarking, and coding to analyze and predict stock prices for S&P 500 companies. The provided dataset includes daily stock prices for 505 companies from 2013-02-08 to 2018-02-07 (619,040 rows).

Virtual Environment Setup

To ensure reproducibility, a virtual environment was created using venv.

Setup Instructions

Create a virtual environment:

python -m venv assignment_env


The requirements.txt file includes all the necessary dependencies for this project:

pandas

pyarrow

polars

lightgbm

xgboost

scikit-learn

streamlit

Part 1: Storing and Retrieving Data

Description

This section benchmarks CSV vs. Parquet file formats to assess:

Read Time

Write Time

File Size

Libraries Chosen

Pandas was used for data manipulation because of its versatility and efficient data handling.

PyArrow was chosen for Parquet file handling due to its performance and flexibility with compression options like snappy, gzip, and brotli.

Key Results

At small scales (1x), CSV was simpler to handle but slower.

At larger scales (10x and 100x), Parquet with snappy compression offered the best balance of file size and read/write performance.

Parquet with brotli compression achieved the smallest file size but had slightly longer read/write times.

Recommendation: Use Parquet with snappy compression for optimal performance and storage efficiency at larger scales.

Execution

Run the following command to execute Part 1:

script1.py

Part 2: Data Manipulation and Prediction Models

Description

This section enhances the dataset by adding four technical indicators:

Simple Moving Average (SMA)

Relative Strength Index (RSI)

Bollinger Bands

Moving Average Convergence Divergence (MACD)

Libraries Chosen

Pandas was used for data manipulation due to its intuitive syntax.

Polars was considered for performance comparisons but was not ultimately used for modeling.

LightGBM, XGBoost, and Linear Regression were selected for model comparison to predict closing stock prices.

Key Results

LightGBM performed best in terms of accuracy and speed.

XGBoost showed competitive accuracy but slower training.

Linear Regression was fast but less accurate.

Recommendation: Use LightGBM for best performance in predicting stock prices based on Mean Absolute Error (MAE) and overall efficiency.

Execution

Run the following command to execute Part 2:

script1a.py

Part 3: Dashboard for Results

Description

This Streamlit dashboard presents:

Benchmark Results with compression type and scale factor filters.

Prediction Results showing predicted vs. actual stock prices for LightGBM, XGBoost, and Linear Regression models.

Libraries Chosen

Streamlit was selected for its simplicity, ease of deployment, and user-friendly interface.

Visualizations were built using Matplotlib for improved flexibility in graph creation.

Execution

Run the following command to launch the dashboard:

streamlit run dashboard.py

Screenshots

A separate word file is included containing relevant screenshots of the Streamlit dashboard and key benchmark results for visual reference.

Conclusion

This project effectively compares data storage formats, builds predictive models for stock prices, and visualizes results using an interactive dashboard. The analysis shows that Parquet with snappy compression is optimal for large-scale data handling. LightGBM emerged as the best-performing model for stock price prediction based on Mean Absolute Error (MAE) and R² score. Streamlit effectively showcased both benchmarking and prediction insights for improved user experience.

