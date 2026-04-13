# Patch scipy sparse matrices for Python 3.13 compatibility
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix
for _cls in (csc_matrix, coo_matrix, csr_matrix):
    if not hasattr(_cls, '__class_getitem__'):
        _cls.__class_getitem__ = classmethod(lambda c, _: c) # type: ignore
        
import os
import pandas as pd

from molpipeline import Pipeline
from molpipeline.any2mol import AutoToMol
from molpipeline.mol2any import MolToConcatenatedVector, MolToMorganFP, MolToRDKitPhysChem

ADMET_ID_TO_ENDPOINT = {
    0: "HLM",
    1: "MLM",
    2: "KSOL",
    3: "MDR1-MDCKII",
    4: "LogD",
}

POTENCY_ID_TO_ENDPOINT = {
    0: "pIC50 (SARS-CoV-2 Mpro)",
    1: "pIC50 (MERS-CoV Mpro)"
}


def load_admet_data(endpoint_id, base_path="data"):
    admet_path = os.path.join(base_path, "ADMET.csv")
    
    df = pd.read_csv(admet_path)
    
    train_data = df[df["Set"] == "Train"]
    test_data = df[df["Set"] == "Test"]
    
    if endpoint_id not in ADMET_ID_TO_ENDPOINT:
        raise ValueError(f"Invalid ADMET endpoint ID: {endpoint_id}")
    
    endpoint_name = ADMET_ID_TO_ENDPOINT[endpoint_id]
    
    X_train = train_data["CXSMILES"]
    y_train = train_data[endpoint_name]
    X_test = test_data["CXSMILES"]
    y_test = test_data[endpoint_name]
    
    train_df = pd.DataFrame({"smiles": X_train, "target": y_train})
    test_df = pd.DataFrame({"smiles": X_test, "target": y_test})
    
    # Remove rows with missing target values
    train_df = train_df.dropna(subset=["target"]).reset_index(drop=True)
    test_df = test_df.dropna(subset=["target"]).reset_index(drop=True)
    
    return train_df, test_df


def load_potency_data(endpoint_id, base_path="data"):
    potency_path = os.path.join(base_path, "Potency.csv")
    
    df = pd.read_csv(potency_path)
    
    train_data = df[df["Set"] == "Train"]
    test_data = df[df["Set"] == "Test"]
    
    if endpoint_id not in POTENCY_ID_TO_ENDPOINT:
        raise ValueError(f"Invalid Potency endpoint ID: {endpoint_id}")
    
    endpoint_name = POTENCY_ID_TO_ENDPOINT[endpoint_id]
    
    X_train = train_data["CXSMILES"]
    y_train = train_data[endpoint_name]
    X_test = test_data["CXSMILES"]
    y_test = test_data[endpoint_name]
    
    train_df = pd.DataFrame({"smiles": X_train, "target": y_train})
    test_df = pd.DataFrame({"smiles": X_test, "target": y_test})
    
    # Remove rows with missing target values
    train_df = train_df.dropna(subset=["target"]).reset_index(drop=True)
    test_df = test_df.dropna(subset=["target"]).reset_index(drop=True)
    
    return train_df, test_df

def get_feature_pipeline():
    return Pipeline([
        ("auto_to_mol", AutoToMol()),
        ("morgan_physchem", MolToConcatenatedVector(
            [
                ("RDKitPhysChem", MolToRDKitPhysChem(
                    standardizer=None,  # we avoid standardization at this point
                )),
                ("MorganFP", MolToMorganFP(
                    n_bits=2048,
                    radius=2,
                    return_as="dense",
                    counted=True,
                )),
            ],
        )),
    ])

if __name__ == "__main__":
    # For all endpoints, check for any missing values in the target column
    for endpoint_id, endpoint_name in ADMET_ID_TO_ENDPOINT.items():
        train_df, test_df = load_admet_data(endpoint_id)
        print(f"ADMET Endpoint: {endpoint_name}")
        print(f"Train set - Missing values in target: {train_df['target'].isna().sum()}")
        print(f"Test set - Missing values in target: {test_df['target'].isna().sum()}")
        print()
    for endpoint_id, endpoint_name in POTENCY_ID_TO_ENDPOINT.items():
        train_df, test_df = load_potency_data(endpoint_id)
        print(f"Potency Endpoint: {endpoint_name}")
        print(f"Train set - Missing values in target: {train_df['target'].isna().sum()}")
        print(f"Test set - Missing values in target: {test_df['target'].isna().sum()}")
        print()
        
    # Check if any targets are negative
    for endpoint_id, endpoint_name in ADMET_ID_TO_ENDPOINT.items():
        train_df, test_df = load_admet_data(endpoint_id)
        print(f"ADMET Endpoint: {endpoint_name}")
        print(f"Train set - Negative target values: {(train_df['target'] < 0).sum()}")
        print(f"Test set - Negative target values: {(test_df['target'] < 0).sum()}")
        print()
    for endpoint_id, endpoint_name in POTENCY_ID_TO_ENDPOINT.items():
        train_df, test_df = load_potency_data(endpoint_id)
        print(f"Potency Endpoint: {endpoint_name}")
        print(f"Train set - Negative target values: {(train_df['target'] < 0).sum()}")
        print(f"Test set - Negative target values: {(test_df['target'] < 0).sum()}")
        print()
        
    # Print dataset sizes
    for endpoint_id, endpoint_name in ADMET_ID_TO_ENDPOINT.items():
        train_df, test_df = load_admet_data(endpoint_id)
        print(f"ADMET Endpoint: {endpoint_name}")
        print(f"Train set size: {len(train_df)}")
        print(f"Test set size: {len(test_df)}")
        print()
    for endpoint_id, endpoint_name in POTENCY_ID_TO_ENDPOINT.items():
        train_df, test_df = load_potency_data(endpoint_id)
        print(f"Potency Endpoint: {endpoint_name}")
        print(f"Train set size: {len(train_df)}")
        print(f"Test set size: {len(test_df)}")
        print()