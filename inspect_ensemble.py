import pickle

# Path to the ensemble file
ensemble_path = './mqp-data/flights-benchmark/spn_ensembles/ensemble_single_flights_1000000.pkl'

# Load the ensemble
try:
    with open(ensemble_path, 'rb') as f:
        ensemble = pickle.load(f)
    print("Ensemble loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file {ensemble_path} does not exist.")
    exit(1)
except Exception as e:
    print(f"Error loading ensemble: {e}")
    exit(1)

# Inspect the ensemble object
print("\nEnsemble Attributes:")
for attr in dir(ensemble):
    if not attr.startswith('_'):
        print(f"{attr}: {getattr(ensemble, attr)}")

# Specifically check for 'pairwise_rdc_path'
if hasattr(ensemble, 'pairwise_rdc_path'):
    print(f"\npairwise_rdc_path: {ensemble.pairwise_rdc_path}")
else:
    print("\nError: 'pairwise_rdc_path' attribute not found in the ensemble.")
