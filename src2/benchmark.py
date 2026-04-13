
import numpy as np

from src2.data import ADMET_ID_TO_ENDPOINT, POTENCY_ID_TO_ENDPOINT, load_admet_data, load_potency_data, get_feature_pipeline

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from src2.tabpfn_utils import create_tabpfn

from xgboost import XGBRegressor

from tqdm import tqdm
import os
import pandas as pd


def create_random_forest(random_state=0):
    return RandomForestRegressor(n_estimators=500, random_state=random_state)

def create_xgboost(random_state=0):
    return XGBRegressor(random_state=random_state)

MODEL_CREATORS = {
    "random_forest": create_random_forest,
    "xgboost": create_xgboost,
    "tabpfn": create_tabpfn,
}

METRICS = {
    "mae": mean_absolute_error,
    "rmse": root_mean_squared_error,
}

def run_model(
    train_data, test_data, model_name, metrics=["rmse", "mae"], random_state=0
):
    feature_pipeline = get_feature_pipeline()
    features = feature_pipeline.transform(train_data["smiles"])
    model = MODEL_CREATORS[model_name](random_state=random_state)
    model.fit(features, y=np.array(train_data["target"]))
    
    test_features = feature_pipeline.transform(test_data["smiles"])
    test_predictions = model.predict(test_features)
    
    results = {}
    for metric in metrics:
        results[metric] = METRICS[metric](np.array(test_data["target"]), test_predictions)
    
    return results

def benchmark_model(model_name, dataset, endpoint_id, random_state=0, metrics=["rmse", "mae"]):
    if dataset == "admet":
        train_data, test_data = load_admet_data(endpoint_id)
    elif dataset == "potency":
        train_data, test_data = load_potency_data(endpoint_id)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    results = run_model(train_data, test_data, model_name, metrics=metrics, random_state=random_state)
    return results

def benchmark_all_models(save_file_name, num_seeds=10, metrics=["rmse", "mae"]):
    datasets = ["admet", "potency"]
    model_names = list(MODEL_CREATORS.keys())
    
    results_log = []
    
    save_path = os.path.join("results", "benchmark_2", save_file_name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    for dataset in datasets:
        if dataset == "admet":
            endpoint_ids = ADMET_ID_TO_ENDPOINT.keys()
        else:
            endpoint_ids = POTENCY_ID_TO_ENDPOINT.keys()
        
        for endpoint_id in tqdm(endpoint_ids, desc=f"{dataset} endpoints", leave=True):
            for model_name in model_names:
                for random_state in tqdm(range(num_seeds), desc=f"{model_name} - {dataset} - {endpoint_id}", leave=True):
                    results = benchmark_model(model_name, dataset, endpoint_id, random_state=random_state, metrics=metrics)
                    results_log.append({
                        "dataset": dataset,
                        "endpoint_id": endpoint_id,
                    "model_name": model_name,
                    "random_state": random_state,
                    **results,
                })
            
                pd.DataFrame(results_log).to_csv(save_path, index=False)
                
if __name__ == "__main__":
    benchmark_all_models(save_file_name="benchmark_results_2.csv", num_seeds=10)