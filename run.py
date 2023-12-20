import pandas as pd
import torch
import os
from scipy.spatial import cKDTree
from sklearn.model_selection import train_test_split
from pycaret.regression import *
from tqdm import tqdm
import numpy as np

# Check for GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_gpu = torch.cuda.is_available()
num_cores = os.cpu_count()
torch.set_num_threads(num_cores)
print(f'Using {device} for computation')

if not use_gpu:
    print(f"Number of CPU cores available: {num_cores}")
    print(f"Number of threads set for PyTorch: {torch.get_num_threads()}")

def load_data(lipid_path : str, gene_path : str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load lipid and gene data from the given file paths.

    Parameters:
    lipid_path (str): Path to the lipid data file.
    gene_path (str): Path to the gene data file.

    Returns:
    tuple: A tuple containing two pandas DataFrames (lipids_data, genes_data).
    """
    lipids_data = pd.read_parquet(lipid_path, engine='pyarrow')
    genes_data = pd.read_parquet(gene_path, engine='pyarrow')
    return lipids_data, genes_data

def create_kdtree(genes_data : pd.DataFrame) -> cKDTree:
    """
    Create a KDTree for the gene data.

    Parameters:
    genes_data (DataFrame): The gene data.

    Returns:
    cKDTree: A KDTree object for the gene data.
    """
    genes_coords = genes_data[['y_ccf', 'z_ccf']].values
    return cKDTree(genes_coords)

def logarithmic_weight(dists : torch.Tensor) -> torch.Tensor:
    """
    Apply a logarithmic weighting scheme to distances.

    Parameters:
    dists (Tensor): Distances tensor.

    Returns:
    Tensor: Weighted distances tensor.
    """
    adjusted_dists = dists + 1e-6
    return -torch.log(adjusted_dists)

def impute_low_importance_lipids(lipids_data: pd.DataFrame, threshold: float = 0.00011) -> pd.DataFrame:
    """
    Imputes low importance lipids from the lipids dataset based on a given standard deviation threshold.
    
    Args:
    lipids_data (pd.DataFrame): DataFrame containing lipid data.
    threshold (float, optional): The threshold for standard deviation below which lipids are considered of low importance. Defaults to 0.00011.

    Returns:
    pd.DataFrame: The imputed DataFrame with low importance lipids removed.
    """
    # Select only the lipids with a standard deviation > threshold
    lipids_values = lipids_data.iloc[:, 13:]
    std_dict = lipids_values[lipids_values > threshold].std(axis=0).to_dict()
    low_imp_lipids = [lipid for lipid, std in std_dict.items() if std < threshold]
    
    # Impute the low importance lipids by removing them
    imputed_data = lipids_data.drop(columns=low_imp_lipids)
    return imputed_data

def aggregate_data(lipids_coords : np.ndarray, genes_kdtree : cKDTree, genes_data : pd.DataFrame, neighbors_num : int) -> torch.Tensor:
    """
    Aggregate gene data based on lipid coordinates.

    Parameters:
    lipids_coords (array): Coordinates of the lipids.
    genes_kdtree (cKDTree): KDTree for the gene data.
    genes_data (Tensor): Gene data as a PyTorch tensor.
    neighbors_num (int): Number of neighbors to consider for aggregation.

    Returns:
    Tensor: Aggregated gene data tensor.
    """
    distances, indices = genes_kdtree.query(lipids_coords, k=neighbors_num)
    distances = torch.tensor(distances).to(device)
    indices = torch.tensor(indices, dtype=torch.long).to(device)
    weighted_sum = torch.zeros((len(lipids_coords), genes_data.shape[1]), device=device)

    for i in tqdm(range(len(lipids_coords)), desc='Aggregating data'):
        dists = distances[i]
        gene_indices = indices[i]
        weights = logarithmic_weight(dists)
        normalized_weights = weights / weights.sum()
        weighted_data = genes_data[gene_indices] * normalized_weights[:, None]
        weighted_sum[i] = weighted_data.sum(axis=0)

    return weighted_sum

def prepare_data_for_modeling(aggregated_gene_data : pd.DataFrame, lipids_data : pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare feature and target dataframes for modeling.

    Parameters:
    aggregated_gene_data (DataFrame): Aggregated gene data.
    lipids_data (DataFrame): Lipid data.

    Returns:
    tuple: A tuple containing the feature and target DataFrames.
    """
    aggregated_gene_data = np.log1p(aggregated_gene_data)
    lipids_data = lipids_data.iloc[:, 13:]
    return aggregated_gene_data, lipids_data

def train_and_evaluate_models(features_df: pd.DataFrame, target_df: pd.DataFrame, use_gpu: bool) -> None:
    """
    Train and evaluate models for each lipid and save individual performance data.

    Parameters:
    features_df (DataFrame): Feature dataframe.
    target_df (DataFrame): Target dataframe.
    use_gpu (bool): Whether to use GPU for training.
    """
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_df, target_df, test_size=0.3, random_state=42)

    for i in tqdm(range(len(y_train.columns)), desc='Processing Lipids'):
        lipid_name = y_train.columns[i]
        train_data = pd.concat([X_train, y_train.iloc[:, i]], axis=1)
        test_data = pd.concat([X_test, y_test.iloc[:, i]], axis=1)

        # Set up PyCaret
        setup(data=train_data, target=y_train.columns[i], test_data=test_data, fold=5, session_id=42, use_gpu=use_gpu, preprocess=False, n_jobs=-1, fold_shuffle=True)
        
        # Create and evaluate the model
        model = create_model('catboost')
        predict_model(model)
        metrics = pull()
        
        # Extract R2 and feature importances
        r2 = metrics.loc[metrics['Model'] == 'CatBoost Regressor', 'R2'].iloc[0]
        feature_importance_df = pd.DataFrame({'Feature': model.feature_names_, 'Importance': model.feature_importances_})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        
        # Prepare data for saving
        summary_df = pd.DataFrame({'Lipid': [lipid_name], 'R2': [r2]})
        with open(f'results/models_performances/lipid_{i}.csv', 'w') as f:
            f.write('# Summary\n')
            summary_df.to_csv(f, index=False)
            f.write('\n# Feature Importances\n')
            feature_importance_df.to_csv(f, index=False)

        # Finalize and save the model
        model = finalize_model(model)
        save_model(model, f'results/models/lipid_{i}')

    # Optional: Print overall metrics
    print(f'Individual model results saved in "results/models" directory.')

def main():
    lipid_path = 'data/section12/lipids_section_12.parquet'
    gene_path = 'data/section12/genes_section_12.parquet'

    lipids_data, genes_data = load_data(lipid_path, gene_path)
    lipids_data = impute_low_importance_lipids(lipids_data)
    
    genes_kdtree = create_kdtree(genes_data)
    lipids_coords = lipids_data[['y_ccf', 'z_ccf']].values
    genes_tensor = torch.tensor(genes_data.iloc[:, 46:-50].values).to(device)

    aggregated_gene_data = aggregate_data(lipids_coords, genes_kdtree, genes_tensor, neighbors_num=1000)
    aggregated_gene_data = pd.DataFrame(aggregated_gene_data.to('cpu').numpy(), columns=genes_data.iloc[:, 46:-50].columns)
    features_df, target_df = prepare_data_for_modeling(aggregated_gene_data, lipids_data)
    
    # Can be spammy due to a LightGBM issue when using GPU
    train_and_evaluate_models(features_df, target_df, use_gpu=use_gpu)
    print("Modeling completed. Results saved to 'results' folder.")

if __name__ == "__main__":
    main()

