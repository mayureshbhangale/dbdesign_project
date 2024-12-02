import pandas as pd
import csv
import os
import re
import pickle

# Paths - Update these paths as needed
csv_file_path = '../../../mqp-data/flights-benchmark/flights.csv'  # Path to your dataset
query_file_path = '../sql/sample_queries.sql'                     # Path to your SQL queries
output_pickle_path = '../sql/sample_ground_truth.pkl'             # Path to save the generated ground truth as pickle
output_times_pickle_path = '../sql/sample_ground_truth_times.pkl' # Path to save query processing times

# Function to inspect the first few lines of the CSV
def inspect_csv(file_path, num_lines=5):
    print(f"\nInspecting the first {num_lines} lines of the CSV file:")
    with open(file_path, 'r') as f:
        for _ in range(num_lines):
            line = f.readline()
            print(line.strip())

# Inspect the CSV to ensure it's correctly formatted
inspect_csv(csv_file_path)

# Load the dataset with comma separator and header row
print(f"\nLoading dataset from {csv_file_path}...")
try:
    df = pd.read_csv(csv_file_path, sep=',', header=0, low_memory=False)
except FileNotFoundError:
    print(f"Error: CSV file not found at {csv_file_path}")
    exit(1)
except Exception as e:
    print(f"Error loading CSV file: {e}")
    exit(1)

print("Dataset loaded successfully.")
print("\nDataset preview:")
print(df.head())

# Display columns and their data types
print("\nDataFrame columns and dtypes:")
print(df.dtypes)

# Convert numeric columns to appropriate types
numeric_columns = [
    "YEAR", "MONTH", "DAY", "DAY_OF_WEEK", "FLIGHT_NUMBER", 
    "SCHEDULED_DEPARTURE", "DEPARTURE_TIME", "DEPARTURE_DELAY", 
    "TAXI_OUT", "WHEELS_OFF", "SCHEDULED_TIME", "ELAPSED_TIME", 
    "AIR_TIME", "DISTANCE", "WHEELS_ON", "TAXI_IN", 
    "SCHEDULED_ARRIVAL", "ARRIVAL_TIME", "ARRIVAL_DELAY", 
    "DIVERTED", "CANCELLED", "AIR_SYSTEM_DELAY", "SECURITY_DELAY", 
    "AIRLINE_DELAY", "LATE_AIRCRAFT_DELAY", "WEATHER_DELAY"
]

for col in numeric_columns:
    if col in df.columns:
        # Convert to numeric, coerce errors to NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("\nNumeric columns converted.")
print("\nDataFrame dtypes after conversion:")
print(df.dtypes)

# Load and process the SQL queries
print(f"\nLoading SQL queries from {query_file_path}...")
try:
    with open(query_file_path, 'r') as f:
        raw_queries = f.readlines()
except FileNotFoundError:
    print(f"Error: SQL file not found at {query_file_path}")
    exit(1)
except Exception as e:
    print(f"Error reading SQL file: {e}")
    exit(1)

# Combine multi-line queries into single lines
queries = []
current_query = ""
for line in raw_queries:
    stripped_line = line.strip()
    if stripped_line.startswith("--") or not stripped_line:
        continue  # skip comments and empty lines
    current_query += " " + stripped_line
    if stripped_line.endswith(";"):
        queries.append(current_query[:-1].strip())  # Remove the trailing semicolon
        current_query = ""

print(f"\nTotal queries loaded: {len(queries)}")

# Prepare ground truth
print("\nGenerating ground truth cardinalities...")
results = {}
ground_truth_times = {}
for query in queries:
    if not query.startswith("SELECT COUNT(*) FROM flights WHERE"):
        print(f"Skipping unsupported query: {query}")
        results[query] = "UNSUPPORTED"
        continue
    try:
        # Extract the WHERE clause
        where_clause = query.replace("SELECT COUNT(*) FROM flights WHERE ", "").strip()
        # Replace standalone '=' with '==', while avoiding '>=', '<=', etc.
        where_clause = re.sub(r'(?<![!<>])=(?![=])', '==', where_clause)
        print(f"\nProcessed WHERE clause: {where_clause}")
        # Use df.query to filter
        filtered_df = df.query(where_clause, engine='python')
        cardinality = len(filtered_df)
        results[query] = cardinality
        print(f"Processed query: {query} -> {cardinality}")
    except Exception as e:
        print(f"Error processing query: {query}\n{e}")
        results[query] = "ERROR"

# Write ground truth to pickle files
print(f"\nWriting ground truth to {output_pickle_path}...")
os.makedirs(os.path.dirname(output_pickle_path), exist_ok=True)
try:
    with open(output_pickle_path, 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)
    with open(output_times_pickle_path, 'wb') as f:
        pickle.dump(ground_truth_times, f, pickle.HIGHEST_PROTOCOL)
    print(f"Ground truth generation completed. Saved to {output_pickle_path} and {output_times_pickle_path}.")
except Exception as e:
    print(f"Error writing pickle file: {e}")
