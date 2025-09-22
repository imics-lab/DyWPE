"""
Example script for running DyWPE benchmark and ablation studies with actual dataset (SelfRegulationSCP2).

Simply import other dataloaders and run the example.
"""

import torch
from src.ablation.benchmark import PositionalEncodingBenchmark
from src.ablation.complete_ablation import run_core_ablation_studies
import os
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.io import arff
from torch.utils.data import TensorDataset, DataLoader

# Directory where datasets will be downloaded and extracted
DATA_DIR = 'datasets'

# Ensure the dataset directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def download_dataset(dataset_name, url):
    """
    Downloads and extracts a zip file containing the dataset.
    """
    zip_path = os.path.join(DATA_DIR, f"{dataset_name}.zip")
    extract_path = os.path.join(DATA_DIR, dataset_name)

    # Download the dataset
    print(f"Downloading {dataset_name} from {url}...")
    urllib.request.urlretrieve(url, zip_path)

    # Extract the zip file
    print(f"Extracting {dataset_name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # Remove the zip file after extraction
    os.remove(zip_path)
    print(f"Dataset {dataset_name} extracted to {extract_path}.")
    return extract_path

def load_arff_data(file_path):
    """
    Loads ARFF file and converts it to a pandas DataFrame.
    """
    print(f"Loading ARFF file: {file_path}")
    data, meta = arff.loadarff(file_path)
    df = pd.DataFrame(data)
    return df

def preprocess_data(train_paths, test_paths, batch_size=64):
    """
    Preprocesses the SelfRegulationSCP1 data:
    - Loads and combines multiple dimensions from ARFF files.
    - Normalizes the features for each dimension.
    - Stacks features from different dimensions.
    - Converts them into PyTorch tensors.
    - Creates DataLoaders for training, validation, and testing.
    """

    # Load all training and test dimensions
    train_dfs = [load_arff_data(path) for path in train_paths]
    test_dfs = [load_arff_data(path) for path in test_paths]

    # Separate features and labels for all dimensions
    train_features = [df.drop(columns=['cortical']) for df in train_dfs]
    test_features = [df.drop(columns=['cortical']) for df in test_dfs]

    # Create a label mapping for the two unique class labels
    label_mapping = {
        b'negativity': 0,
        b'positivity': 1
    }

    # Apply the label mapping to the training and test sets
    train_labels = train_dfs[0]['cortical'].apply(lambda x: label_mapping[x]).values
    test_labels = test_dfs[0]['cortical'].apply(lambda x: label_mapping[x]).values

    # Normalize the features using StandardScaler for each dimension
    scalers = [StandardScaler() for _ in range(6)]  # 6 dimensions
    train_features_normalized = [scalers[i].fit_transform(train_features[i]) for i in range(6)]
    test_features_normalized = [scalers[i].transform(test_features[i]) for i in range(6)]

    # Stack all dimensions along a new axis (multivariate time-series)
    X_train = np.stack(train_features_normalized, axis=-1)
    X_test_full = np.stack(test_features_normalized, axis=-1)

    # Split the test data into validation and test sets
    X_valid, X_test, y_valid, y_test = train_test_split(X_test_full, test_labels, test_size=0.50, random_state=42)
    y_train = train_labels

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.int64)

    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    y_valid = torch.tensor(y_valid, dtype=torch.int64)

    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.int64)

    # Output dataset shapes
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_valid shape: {X_valid.shape}, y_valid shape: {y_valid.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Return both the DataLoaders and the raw tensors
    return train_loader, valid_loader, test_loader, X_train, X_valid, X_test, y_train, y_valid, y_test

# Example usage for downloading, extracting, and preprocessing the SelfRegulationSCP1 dataset
if __name__ == "__main__":
    # URL for the dataset
    dataset_name = 'SelfRegulationSCP2'
    dataset_url = 'https://timeseriesclassification.com/aeon-toolkit/SelfRegulationSCP2.zip'

    # Download and extract the dataset
    extract_path = download_dataset(dataset_name, dataset_url)

    # Paths for the ARFF files
    train_arff_paths = [
        os.path.join(extract_path, 'SelfRegulationSCP2Dimension1_TRAIN.arff'),
        os.path.join(extract_path, 'SelfRegulationSCP2Dimension2_TRAIN.arff'),
        os.path.join(extract_path, 'SelfRegulationSCP2Dimension3_TRAIN.arff'),
        os.path.join(extract_path, 'SelfRegulationSCP2Dimension4_TRAIN.arff'),
        os.path.join(extract_path, 'SelfRegulationSCP2Dimension5_TRAIN.arff'),
        os.path.join(extract_path, 'SelfRegulationSCP2Dimension6_TRAIN.arff'),
        os.path.join(extract_path, 'SelfRegulationSCP2Dimension7_TRAIN.arff')
    ]

    test_arff_paths = [
        os.path.join(extract_path, 'SelfRegulationSCP2Dimension1_TEST.arff'),
        os.path.join(extract_path, 'SelfRegulationSCP2Dimension2_TEST.arff'),
        os.path.join(extract_path, 'SelfRegulationSCP2Dimension3_TEST.arff'),
        os.path.join(extract_path, 'SelfRegulationSCP2Dimension4_TEST.arff'),
        os.path.join(extract_path, 'SelfRegulationSCP2Dimension5_TEST.arff'),
        os.path.join(extract_path, 'SelfRegulationSCP2Dimension6_TEST.arff'),
        os.path.join(extract_path, 'SelfRegulationSCP2Dimension7_TEST.arff')
    ]

    # Preprocess the data
    train_loader, valid_loader, test_loader, X_train, X_valid, X_test, y_train, y_valid, y_test = preprocess_data(train_arff_paths, test_arff_paths)

    n_classes = len(torch.unique(y_train))

    # Output the number of classes
    print(f"Number of classes: {n_classes}")

    
    # Dataset parameters - adjust these to match your datasets
    input_timesteps = 256  
    in_channels = 6        
    patch_size = 128        
    num_classes = 2       
    
    
    # =================================================================
    # BENCHMARK CONFIGURATION (Ready to use)
    # =================================================================
    
    # Initialize benchmark
    benchmark = PositionalEncodingBenchmark()

    model_params = {
        'input_timesteps': 1152,
        'in_channels': 6,
        'patch_size': 128,
        'embedding_dim': 32,
        'num_transformer_layers': 4,
        'num_heads': 4,
        'dim_feedforward': 128,
        'dropout': 0.2,
        'num_classes': 2
    }

    print("\nStarting benchmark...")
    models = benchmark.run_full_benchmark(
        model_params,
        train_loader,
        valid_loader,
        test_loader,
        encodings=['dywpe'],  # Can add more: ['rope', 'fixed', 'learned']
        n_epochs=100
    )
    
    # Get results
    benchmark.print_summary()
    benchmark.plot_comparison('final_results_with_test.png')

    # Run core ablation studies
    results = run_core_ablation_studies(
        dataset_name='SRSCP2',
        model_params=model_params,
        train_loader=None,  
        valid_loader=None,  
        test_loader=None,   
        num_epochs=100
    )

    print("Core ablation studies completed!")
    
    print(f"\aAblation completed!")


   
    

