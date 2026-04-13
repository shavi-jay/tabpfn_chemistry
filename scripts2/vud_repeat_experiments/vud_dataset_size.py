from src2.data import ADMET_ID_TO_ENDPOINT, POTENCY_ID_TO_ENDPOINT, load_admet_data, load_potency_data, get_feature_pipeline
from src2.tabpfn_utils import create_tabpfn

import numpy as np
import pandas as pd

from src2.utils import set_seed
from src2.vud_repeat import vud_repeat_decomposition
from tqdm import tqdm

import os

def run_vud_experiment(
    dataset_name,
    save_path,
    num_u_samples=5,
    num_test_points=20,
):
    # Output is a table with columns: test_point_index, train_size, total_uncertainty, aleatoric_uncertainty_estimate
    set_seed(0)
    
    save_path = os.path.join(save_path, f"{dataset_name}.csv")
    
    # Load Dataset
    # Dataset name in the format "admet_i" or "potency_i"
    dataset_type, endpoint_id_str = dataset_name.split("_")
    endpoint_id = int(endpoint_id_str)
    if dataset_type == "admet":
        train_df, test_df = load_admet_data(endpoint_id)
    elif dataset_type == "potency":
        train_df, test_df = load_potency_data(endpoint_id)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    feature_pipeline = get_feature_pipeline()
    
    # Process features
    X_train = feature_pipeline.fit_transform(train_df["smiles"])
    y_train = train_df["target"].values
    X_test = feature_pipeline.transform(test_df["smiles"])
    y_test = test_df["target"].values
    
    # Subsample test points to num_test_points
    
    test_point_ids = np.random.permutation(np.arange(len(X_test)))[:num_test_points]
    X_test = X_test[test_point_ids]
    y_test = y_test[test_point_ids]
        
    # Increase the train set size by factor of 2 starting from 2, up to the full train set size
    train_sizes = [2**i for i in range(1, int(np.log2(len(X_train))) + 1)] + [len(X_train)]
    
    model = create_tabpfn(random_state=0)

    uncertainty_results = {
        "test_point_index": [],
        "train_size": [],
        "total_uncertainty": [],
        "aleatoric_uncertainty_estimates": [],
    }
    
    train_ids = np.random.permutation(np.arange(len(X_train)))

    for train_size in tqdm(train_sizes):
        
        X_train_sub = X_train[train_ids[:train_size]]
        y_train_sub = y_train[train_ids[:train_size]]
        
        total_uncertainty, aleatoric_uncertainty_estimates = vud_repeat_decomposition(
            model=model,
            in_context_features=X_train_sub,
            in_context_labels=y_train_sub,
            test_features=X_test,
            num_u_samples=num_u_samples,
        )
        # Convert tensors to numpy
        uncertainty_results["test_point_index"].extend(test_point_ids)
        uncertainty_results["train_size"].extend([train_size] * len(X_test))
        uncertainty_results["total_uncertainty"].extend(total_uncertainty)
        uncertainty_results["aleatoric_uncertainty_estimates"].extend(aleatoric_uncertainty_estimates)
        
        pd.DataFrame(uncertainty_results).to_csv(save_path, index=False)
            
if __name__ == "__main__":
    save_path = os.path.join("results", "vud_repeat", "dataset_size_2")
    os.makedirs(save_path, exist_ok=True)
    
    for endpoint_id in ADMET_ID_TO_ENDPOINT.keys():
        dataset_name = f"admet_{endpoint_id}"
        print(f"Running VUD experiment for dataset: {dataset_name}")
        run_vud_experiment(dataset_name, save_path)
        
    for endpoint_id in POTENCY_ID_TO_ENDPOINT.keys():
        dataset_name = f"potency_{endpoint_id}"
        print(f"Running VUD experiment for dataset: {dataset_name}")
        run_vud_experiment(dataset_name, save_path)