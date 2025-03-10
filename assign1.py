# %% Import libraries
import pandas as pd
import pyarrow.parquet as pq
import time
import os

# %% Define file paths for CSV and Parquet formats
csv_file = "/Users/sanchita/Desktop/Assignment_1/all_stocks_5yr.csv"  # Assuming you have the dataset file
parquet_file = "all_stocks_5yr.parquet"

# %% Benchmark function
def benchmark_csv_vs_parquet(csv_path, parquet_path, compression=None, scale_factor=1):
    """
    Function to benchmark CSV vs. Parquet in terms of read/write speed and file size at different scales.
    Parameters:
        csv_path (str): Path to the CSV file.
        parquet_path (str): Path to store the Parquet file.
        compression (str or None): Compression type for Parquet (e.g., "snappy", "gzip", "brotli").
        scale_factor (int): Scale factor for benchmarking (1x, 10x, 100x).
    Returns:
        dict: Dictionary containing read/write times and file sizes for both formats at the given scale.
    """
    # Measure time taken to read the CSV file
    start_time = time.time()
    df = pd.read_csv(csv_path)
    csv_read_time = time.time() - start_time

    # Scale the dataset by duplicating it for 10x and 100x benchmarking
    df = pd.concat([df] * scale_factor, ignore_index=True)

    # Measure time taken to write the DataFrame to Parquet with specified compression
    start_time = time.time()
    df.to_parquet(parquet_path, engine="pyarrow", compression=compression)
    parquet_write_time = time.time() - start_time

    # Measure time taken to read the Parquet file
    start_time = time.time()
    df_parquet = pd.read_parquet(parquet_path)
    parquet_read_time = time.time() - start_time

    # Get file sizes in megabytes (MB)
    csv_size = os.path.getsize(csv_path) / (1024 * 1024)  # Convert bytes to MB
    parquet_size = os.path.getsize(parquet_path) / (1024 * 1024)  # Convert bytes to MB

    return {
        "Scale Factor": scale_factor,
        "CSV Read Time (s)": csv_read_time,
        "Parquet Read Time (s)": parquet_read_time,
        "Parquet Write Time (s)": parquet_write_time,
        "CSV Size (MB)": csv_size,
        "Parquet Size (MB)": parquet_size,
        "Compression": compression
    }

# %% List of compression types and scale factors
compression_types = ["None", "snappy", "gzip", "brotli"]
scale_factors = [1, 10, 100]  # Scaling dataset size to 1x, 10x, 100x
results = []  # Store benchmark results

# %% Run benchmarks
for compression in compression_types:
    for scale in scale_factors:
        result = benchmark_csv_vs_parquet(csv_file, parquet_file, compression, scale)
        results.append(result)

# %% Convert results to DataFrame and display
benchmark_df = pd.DataFrame(results)
benchmark_df.to_csv("benchmark_results.csv", index=False)
print(benchmark_df)  # Display the benchmark results

# %%
