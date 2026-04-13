from src2.tabpfn_utils import create_tabpfn
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm

# Implement Variational Uncertainty Decomposition


# Step 0: Compute entropy and sample from predictive distribution

def entropy_from_logits(logits: torch.Tensor):
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy_terms = torch.nan_to_num(probs * log_probs, nan=0.0)
    return -(entropy_terms).sum(dim=-1).detach().cpu().numpy()

def sample_from_fitted_model(model, logits: torch.Tensor, num_samples: int):
    samples = torch.stack([
        model.raw_space_bardist_.sample(logits)
        for _ in range(num_samples)
    ], dim=1)
    return samples

def get_total_uncertainty(
    model,
    in_context_features,
    in_context_labels,
    test_features,
):
    model.fit(in_context_features, y=in_context_labels)
    predictions = model.predict(test_features, output_type="full")
    logits = predictions["logits"]
    return entropy_from_logits(logits)
    
def vud_repeat_decomposition(
    model,
    in_context_features,
    in_context_labels,
    test_features,
    num_u_samples=5,
):
    # Fit Model to In-Context Data
    model.fit(in_context_features, y=in_context_labels)
    
    # Get Predictive Distribution for Test Points
    predictions = model.predict(test_features, output_type="full")
    logits = predictions["logits"]
    total_uncertainty = entropy_from_logits(logits)
    
    # Get U Samples
    samples = sample_from_fitted_model(model, logits, num_u_samples)
    
    aleatoric_uncertainty_estimates = []
    
    for i, test_point in tqdm(enumerate(test_features)):
        u_samples = samples[i]
        
        cum_uncertainty_estimate = 0.0
        
        for u_sample in u_samples:
            # Create new in-context dataset with test point and u_sample
            augmented_features = np.concatenate([in_context_features, [test_point]], axis=0)
            augmented_labels = np.concatenate([in_context_labels, [u_sample]], axis=0)

            
            # Fit new model to augmented dataset
            augmented_model = create_tabpfn(random_state=0)
            augmented_model.fit(augmented_features, y=augmented_labels)
            
            # Get predictive distribution for test point
            augmented_logits = augmented_model.predict([test_point], output_type="full")["logits"]
            
            # Compute entropy for this augmented model
            aleatoric_uncertainty_estimate = entropy_from_logits(augmented_logits)
            cum_uncertainty_estimate += aleatoric_uncertainty_estimate.item()
            
        # Average over u_samples to get final aleatoric uncertainty estimate
        aleatoric_uncertainty_estimate = cum_uncertainty_estimate / num_u_samples
        aleatoric_uncertainty_estimates.append(aleatoric_uncertainty_estimate)
        
        # print(f"Point {i}: TU = {total_uncertainty[i].item():.4f}, AU = {aleatoric_uncertainty_estimate:.4f}")
        
    return total_uncertainty, aleatoric_uncertainty_estimates
    
# if __name__ == "__main__":
    # dataset = "admet"
    # endpoint_id = 0
    # seed = 0
    
    # set_seed(seed)
    
    # train_data, test_data = load_admet_data(endpoint_id)
    
    # feature_pipeline = get_feature_pipeline()
    # train_features = feature_pipeline.transform(train_data["smiles"])
    
    # tabpfn_model = create_tabpfn(random_state=0)
        
    # test_features = feature_pipeline.transform(test_data["smiles"].iloc[:10])
    
    # total_uncertainty, aleatoric_uncertainty_estimates = vud_repeat_decomposition(
    #     tabpfn_model,
    #     in_context_features=train_features,
    #     in_context_labels=np.array(train_data["target"]),
    #     test_features=test_features,
    #     num_u_samples=5,
    # )
    
    # tabpfn_model.fit(train_features, y=np.array(train_data["target"]))

    # predictions = tabpfn_model.predict(test_features, output_type="full")
    
    # logits = predictions["logits"]
    # samples = torch.stack([
    #     tabpfn_model.raw_space_bardist_.sample(logits)
    #     for _ in range(10)
    # ], dim=1)
    
    # print(samples)


