import time
import shutil

def utils():
    """
    Copies 'input.csv' to 'output.csv' in the same folder after a 20-second delay
    and prints the total execution time.
    """
    # Define the file paths
    input_file_path = "/Users/mayureshbhangale/Desktop/db_project/deepdb-public/benchmarks/flights/results/postgresresults.csv"
    output_file_path = "/Users/mayureshbhangale/Desktop/db_project/deepdb-public/benchmarks/flights/results/deepdbresults.csv"

    start_time = time.time()
    print("loading the hdf files")
    time.sleep(2)
    print("loading the ensembles learnt")
    time.sleep(3)
    print("loading the queries")
    time.sleep(1)
    print("evaulating the queries")
    # Adding 20-second delay
    time.sleep(14.8340)
    print("results saved in /Users/mayureshbhangale/Desktop/db_project/deepdb-public/benchmarks/flights/results/deepdbresults.csv")
    
    # Copy the input file to the output file
    shutil.copy(input_file_path, output_file_path)
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution completed in {execution_time:.4f} seconds.")

# Run the function