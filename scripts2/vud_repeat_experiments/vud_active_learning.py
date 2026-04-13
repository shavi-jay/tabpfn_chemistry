from src2.data import ADMET_ID_TO_ENDPOINT, POTENCY_ID_TO_ENDPOINT, load_admet_data, load_potency_data, get_feature_pipeline
from src2.tabpfn_utils import create_tabpfn

import numpy as np
import pandas as pd

from src2.utils import set_seed
from src2.vud_repeat import vud_repeat_decomposition, get_total_uncertainty
from tqdm import tqdm

from typing import Literal

import os

import argparse

# Active learning via uncertainty sampling using VUD repeat decomposition.

def test_rmse(X_train, y_train, seen_ids, X_test, y_test):
    model = create_tabpfn(random_state=0)
    model.fit(X_train[seen_ids], y_train[seen_ids])
    test_predictions = model.predict(X_test, output_type="mean")
    return np.sqrt(np.mean((test_predictions - y_test) ** 2))

def run_active_learning_experiment(
    dataset_name,
    save_path,
    uncertainty_method: Literal["epistemic", "total", "random"],
    num_u_samples=5,
    train_data_increment=10,
    max_train_size=200,
    active_learning_seed=0,
    tabpfn_seed=0,
):
    set_seed(active_learning_seed)
    
    save_path = os.path.join(save_path, dataset_name, f"{active_learning_seed}.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
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
    
    # Intialise training set with 10 random points
    train_ids = np.random.permutation(np.arange(len(X_train)))
    initial_train_size = min(10, len(X_train))
    seen_ids = train_ids[:initial_train_size]
    unseen_ids = train_ids[initial_train_size:]
    
    results = []
    
    # Initial model fit and evaluation
    initial_test_rmse = test_rmse(X_train, y_train, seen_ids, X_test, y_test)
    
    result = {
        "step": 0,
        "train_size": len(seen_ids),
        "new_train_ids": seen_ids.tolist(),
        "test_rmse": initial_test_rmse,
    }
    results.append(result)

    
    # Active learning loop
    for i, train_size in enumerate(range(initial_train_size, min(max_train_size, len(X_train)), train_data_increment)):
        print(f"Training with {train_size} points")
        X_train_sub = X_train[seen_ids]
        y_train_sub = y_train[seen_ids]
        
        if uncertainty_method == "epistemic":
            unseen_total_uncertainty, unseen_aleatoric_uncertainty = vud_repeat_decomposition(
                model=create_tabpfn(random_state=tabpfn_seed),
                in_context_features=X_train_sub,
                in_context_labels=y_train_sub,
                test_features=X_train[unseen_ids],
                num_u_samples=num_u_samples,
            )
            uncertainty_scores = unseen_total_uncertainty - unseen_aleatoric_uncertainty
        elif uncertainty_method == "total":
            uncertainty_scores = get_total_uncertainty(
                model=create_tabpfn(random_state=tabpfn_seed),
                in_context_features=X_train_sub,
                in_context_labels=y_train_sub,
                test_features=X_train[unseen_ids]
            )
        elif uncertainty_method == "random":
            uncertainty_scores = np.random.rand(len(unseen_ids))
        else:
            raise ValueError(f"Unknown uncertainty method: {uncertainty_method}")
        
        # Select top-k most uncertain points to add to training set
        top_k_indices = np.argsort(uncertainty_scores)[-train_data_increment:]
        new_seen_ids = unseen_ids[top_k_indices]
        seen_ids = np.concatenate([seen_ids, new_seen_ids])
        unseen_ids = np.setdiff1d(unseen_ids, new_seen_ids)
        
        # After each active learning loop, evaluate model on test set and save results
        test_rmse_score = test_rmse(X_train, y_train, seen_ids, X_test, y_test)
        result = {
            "step": i+1,
            "train_size": len(seen_ids),
            "test_rmse": test_rmse_score,
            "new_train_ids": new_seen_ids.tolist()
        }
        results.append(result)
        
    # Full model fit and evaluation at the end
    final_test_rmse = test_rmse(X_train, y_train, train_ids, X_test, y_test)
    result = {
        "step": len(results),
        "train_size": len(train_ids),
        "test_rmse": final_test_rmse,
        "new_train_ids": train_ids.tolist()
    }
    results.append(result)
    
    # Save results to CSV
    pd.DataFrame(results).to_csv(save_path, index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run active learning experiment")
    
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name in the format 'admet_i' or 'potency_i'")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save results CSV")
    parser.add_argument("--uncertainty_method", type=str, required=True, choices=["epistemic", "total", "random"], help="Uncertainty method to use for active learning")
    parser.add_argument("--active_learning_seed", type=int, default=0, help="Random seed for active learning")
    
    args = parser.parse_args()
    run_active_learning_experiment(
        dataset_name=args.dataset_name,
        save_path=args.save_path,
        uncertainty_method=args.uncertainty_method,
        active_learning_seed=args.active_learning_seed,
    )